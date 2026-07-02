#pragma once

#include "bonsai/config/booster_config.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/objective.hpp"
#include "bonsai/parallel.hpp"
#include "bonsai/sampler.hpp"
#include "bonsai/types.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

namespace bonsai
{

class IBooster
{
  public:
    virtual ~IBooster()                                             = default;
    virtual void   update_one_iter(Dataset const &train)            = 0;
    virtual float  eval(features_view X, floats_view labels) const  = 0;
    virtual void   predict(features_view X, floats_out y_hat) const = 0;
    virtual size_t n_iters() const                                  = 0;

    // Incremental prediction support for early stopping: accumulate only the
    // newest tree's (shrinkage-scaled) contribution into `scores`, a buffer
    // the caller initialized to score_base() before the first tree.
    virtual float score_base() const                                        = 0;
    virtual void  accumulate_last_tree(features_view X, floats_out scores) const = 0;
    // Drop trees beyond the first n_trees (keep the best iteration's model).
    virtual void truncate(size_t n_trees) = 0;
};

namespace internal
{

// Accumulate a tree's (unscaled-by-lr) train contribution into `out` by
// routing rows in bin space. Thresholds map back to their bins exactly —
// the grower sets threshold = cuts[bin], and the binner is right-inclusive,
// so lower_bound(cuts, threshold) recovers the split bin. Used by DART to
// subtract dropped trees without caching per-tree train predictions.
inline void accumulate_train_contribution(DenseTree const &tree, Dataset const &ds,
                                          floats_out out)
{
    auto const           &nodes = tree.nodes();
    std::vector<bin_id_t> tbin(nodes.size(), 0);
    for (size_t i = 0; i < nodes.size(); ++i)
    {
        if (!DenseTree::is_leaf(nodes[i]))
        {
            auto const cuts = ds.mappers()[nodes[i].feature_id].cuts();
            tbin[i]         = static_cast<bin_id_t>(
                std::ranges::lower_bound(cuts, nodes[i].threshold_or_value) -
                cuts.begin());
        }
    }
    parallel::for_each_index(
        ds.n_rows(),
        [&](size_t r)
        {
            node_id_t idx = 0;
            while (!DenseTree::is_leaf(nodes[idx]))
            {
                auto const    &nd   = nodes[idx];
                auto const    &bins = ds.feature_bins(nd.feature_id);
                auto const     last =
                    static_cast<bin_id_t>(ds.n_bins(nd.feature_id) - 1);
                bin_id_t const b = bins[r];
                bool const     left =
                    (b == last) ? nd.default_left : b <= tbin[idx];
                idx = left ? nd.left : nd.right;
            }
            out[r] += nodes[idx].threshold_or_value;
        });
}

inline void accumulate_train_contribution(ObliviousTree const &tree,
                                          Dataset const &ds, floats_out out)
{
    auto const           &splits = tree.splits();
    std::vector<bin_id_t> tbin(splits.size(), 0);
    for (size_t lvl = 0; lvl < splits.size(); ++lvl)
    {
        auto const cuts = ds.mappers()[splits[lvl].feature_id].cuts();
        tbin[lvl]       = static_cast<bin_id_t>(
            std::ranges::lower_bound(cuts, splits[lvl].threshold) - cuts.begin());
    }
    parallel::for_each_index(
        ds.n_rows(),
        [&](size_t r)
        {
            size_t index = 0;
            for (size_t lvl = 0; lvl < splits.size(); ++lvl)
            {
                auto const    &s    = splits[lvl];
                auto const    &bins = ds.feature_bins(s.feature_id);
                auto const     last =
                    static_cast<bin_id_t>(ds.n_bins(s.feature_id) - 1);
                bin_id_t const b = bins[r];
                bool const     left =
                    (b == last) ? s.default_left : b <= tbin[lvl];
                index = (index << 1U) | (left ? 0U : 1U);
            }
            out[r] += tree.leaf_table()[index];
        });
}

} // namespace internal

template <Objective Obj, TreeGrower Gr, Sampler Sa>
class Booster final : public IBooster
{
  public:
    using objective_type = Obj;
    using grower_type    = Gr;
    using sampler_type   = Sa;
    using tree_type      = typename Gr::Tree;

    explicit Booster(Config const &config)
        : config_(config.booster_config), objective_(config), grower_(config.tree_config),
          sampler_(config), rng_(config.booster_config.random_seed)
    {
    }

    void update_one_iter(Dataset const &train) override
    {
        if (trees_.empty())
        {
            grad_.resize(train.n_rows());
            hess_.resize(train.n_rows());
            init_score_ = objective_.init_score(train.labels());
            scores_.assign(train.n_rows(), init_score_);
        }

        // DART: drop a random subset of existing trees; gradients are
        // computed against the model without them, and afterwards both the
        // dropped trees and the new tree are rescaled (k/(k+1), 1/(k+1)).
        std::vector<size_t> dropped;
        std::vector<float>  dropped_sum;
        if (config_.dart_drop_rate > 0.0F && !trees_.empty())
        {
            std::bernoulli_distribution drop(config_.dart_drop_rate);
            for (size_t i = 0; i < trees_.size(); ++i)
            {
                if (drop(rng_))
                {
                    dropped.push_back(i);
                }
            }
            if (!dropped.empty())
            {
                dropped_sum.assign(train.n_rows(), 0.0F);
                for (size_t const i : dropped)
                {
                    internal::accumulate_train_contribution(trees_[i], train,
                                                            dropped_sum);
                }
                parallel::for_each_index(
                    scores_.size(),
                    [&](size_t i)
                    { scores_[i] -= config_.learning_rate * dropped_sum[i]; });
            }
        }

        objective_.compute(scores_, train.labels(), grad_, hess_);

        if (!train.weights().empty())
        {
            for (size_t i = 0; i < grad_.size(); ++i)
            {
                grad_[i] *= train.weights()[i];
                hess_[i] *= train.weights()[i];
            }
        }

        if (row_indices_.size() != train.n_rows())
        {
            row_indices_.resize(train.n_rows());
        }
        size_t const n_selected = sampler_.sample(grad_, hess_, rng_, row_indices_);

        auto [tree, leaf_values] =
            grower_.grow(train, grad_, hess_, {row_indices_.data(), n_selected});

        if (!dropped.empty())
        {
            // xgboost's normalize_type="tree" factors: the new tree lands
            // with weight lr/(k+lr) — comparable to a plain shrinkage step —
            // and each dropped tree shrinks by k/(k+lr). (The original DART
            // paper's 1/(k+1) assumes unshrunk trees and starves the new
            // tree by ~1/lr when combined with a learning rate.)
            auto const  k         = static_cast<float>(dropped.size());
            float const new_scale = 1.0F / (k + config_.learning_rate);
            float const old_scale = k / (k + config_.learning_rate);
            tree.scale_leaves(new_scale);
            for (size_t const i : dropped)
            {
                trees_[i].scale_leaves(old_scale);
            }
            // scores_ currently exclude the dropped trees entirely; add back
            // their rescaled contribution plus the scaled new tree.
            parallel::for_each_index(
                scores_.size(),
                [&](size_t i)
                {
                    scores_[i] += config_.learning_rate *
                                  ((old_scale * dropped_sum[i]) +
                                   (new_scale * leaf_values[i]));
                });
        }
        else
        {
            parallel::for_each_index(scores_.size(),
                                     [&](size_t i) {
                                         scores_[i] +=
                                             config_.learning_rate * leaf_values[i];
                                     });
        }

        trees_.push_back(std::move(tree));
    }

    float eval(features_view X, floats_view labels) const override
    {
        std::vector<float> scores(X.extent(0));
        predict(X, scores);
        return objective_.eval(scores, labels);
    }

    void predict(features_view X, floats_out scores) const override
    {
        assert(X.extent(0) == scores.size());
        std::fill(scores.begin(), scores.end(), 0.0F);
        for (auto const &tree : trees_)
        {
            tree.predict(X, scores);
        }
        for (float &score : scores)
        {
            score = init_score_ + (score * config_.learning_rate);
        }
    }

    size_t n_iters() const override
    {
        return trees_.size();
    }

    float score_base() const override
    {
        return init_score_;
    }

    void accumulate_last_tree(features_view X, floats_out scores) const override
    {
        assert(!trees_.empty());
        assert(X.extent(0) == scores.size());
        std::vector<float> raw(scores.size(), 0.0F);
        trees_.back().predict(X, raw);
        parallel::for_each_index(scores.size(),
                                 [&](size_t i)
                                 { scores[i] += config_.learning_rate * raw[i]; });
    }

    void truncate(size_t n_trees) override
    {
        if (n_trees < trees_.size())
        {
            // erase, not resize: growth would require default-constructible
            // trees, and truncate only ever shrinks.
            trees_.erase(trees_.begin() + static_cast<std::ptrdiff_t>(n_trees),
                         trees_.end());
        }
    }

    // Save/load accessors. Public so io::save_booster / io::load_booster
    // can serialize state without befriending the I/O module.
    std::vector<tree_type> const &trees() const
    {
        return trees_;
    }
    float init_score() const
    {
        return init_score_;
    }
    void load_state(std::vector<tree_type> trees, float init_score)
    {
        trees_      = std::move(trees);
        init_score_ = init_score;
    }

  private:
    BoosterConfig          config_;
    objective_type         objective_;
    grower_type            grower_;
    sampler_type           sampler_;
    std::mt19937           rng_;
    std::vector<tree_type> trees_;
    std::vector<float>     scores_;
    std::vector<float>     grad_;
    std::vector<float>     hess_;
    std::vector<row_id_t>  row_indices_;
    float                  init_score_ = 0.0F;
};

} // namespace bonsai
