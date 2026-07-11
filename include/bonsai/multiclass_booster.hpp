#pragma once

#include "bonsai/booster.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/config/errors.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/parallel.hpp"
#include "bonsai/sampler.hpp"
#include "bonsai/types.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bonsai
{

// K-class softmax boosting: each round grows one tree per class on the
// softmax gradients (grad_k = p_k - 1[y == k], hess_k = 2 p_k (1 - p_k),
// xgboost's convention). trees_ is flat, round-major: tree for class k of
// round r sits at index r * K + k. predict() emits argmax class ids;
// eval() is the multiclass logloss. The 1-D Objective concept can't
// express the K-output shape, which is why this is its own IBooster
// implementation dispatched via BoosterFor<{softmax, G, Sa}>.
template <TreeGrower Gr, Sampler Sa> class MulticlassBooster final : public IBooster
{
  public:
    using grower_type  = Gr;
    using sampler_type = Sa;
    using tree_type    = typename Gr::Tree;

    explicit MulticlassBooster(Config const &config)
        : config_(config.booster_config), n_classes_(config.objective.n_classes),
          grower_(config.tree_config), sampler_(config),
          rng_(config.booster_config.random_seed)
    {
        if (n_classes_ < 2)
        {
            throw ConfigError("objective.n_classes must be >= 2 for softmax");
        }
    }

    void update_one_iter(Dataset const &train) override
    {
        size_t const n   = train.n_rows();
        size_t const n_k = n_classes_;
        if (scores_.empty())
        {
            grad_.resize(n);
            hess_.resize(n);
            init_scores_.assign(n_k, 0.0F);
            if (trees_.empty())
            {
                // Log class priors, like lightgbm's boost_from_average.
                std::vector<double> counts(n_k, 0.0);
                for (float const y : train.labels())
                {
                    counts[class_of(y, n_k)] += 1.0;
                }
                for (size_t k = 0; k < n_k; ++k)
                {
                    init_scores_[k] = static_cast<float>(
                        std::log(std::max(counts[k], 1.0) / static_cast<double>(n)));
                }
            }
            scores_.assign(n * n_k, 0.0F);
            for (size_t i = 0; i < n; ++i)
            {
                for (size_t k = 0; k < n_k; ++k)
                {
                    scores_[(i * n_k) + k] = init_scores_[k];
                }
            }
            if (!trees_.empty())
            {
                // Warm start: replay every tree's train contribution.
                for (size_t t = 0; t < trees_.size(); ++t)
                {
                    std::vector<float> raw(n, 0.0F);
                    internal::accumulate_train_contribution(trees_[t], train, raw);
                    size_t const k = t % n_k;
                    for (size_t i = 0; i < n; ++i)
                    {
                        scores_[(i * n_k) + k] += config_.learning_rate * raw[i];
                    }
                }
            }
        }

        // Per-row softmax probabilities feed every class's gradients.
        std::vector<float> probs(n * n_k);
        parallel::for_each_index(
            n,
            [&](size_t i)
            {
                double maxv = scores_[i * n_k];
                for (size_t k = 1; k < n_k; ++k)
                {
                    maxv = std::max(maxv, static_cast<double>(scores_[(i * n_k) + k]));
                }
                double sum = 0.0;
                for (size_t k = 0; k < n_k; ++k)
                {
                    double const e       = std::exp(scores_[(i * n_k) + k] - maxv);
                    probs[(i * n_k) + k] = static_cast<float>(e);
                    sum += e;
                }
                for (size_t k = 0; k < n_k; ++k)
                {
                    probs[(i * n_k) + k] /= static_cast<float>(sum);
                }
            });

        if (row_indices_.size() != n)
        {
            row_indices_.resize(n);
        }
        for (size_t k = 0; k < n_k; ++k)
        {
            parallel::for_each_index(
                n,
                [&](size_t i)
                {
                    float const p = probs[(i * n_k) + k];
                    float const y = class_of(train.labels()[i], n_k) == k ? 1.0F : 0.0F;
                    grad_[i]      = p - y;
                    hess_[i]      = std::max(2.0F * p * (1.0F - p), 1e-6F);
                });
            size_t const n_selected = sampler_.sample(grad_, hess_, rng_, row_indices_);
            auto [tree, leaf_values, leaf_ids] =
                grower_.grow(train, grad_, hess_, {row_indices_.data(), n_selected});
            parallel::for_each_index(
                n, [&](size_t i)
                { scores_[(i * n_k) + k] += config_.learning_rate * leaf_values[i]; });
            trees_.push_back(std::move(tree));
            grower_.recycle(std::move(leaf_values), std::move(leaf_ids));
        }
    }

    // Argmax class id per row.
    void predict(features_view X, floats_out y_hat) const override
    {
        predict_at(X, y_hat, 0);
    }

    void predict_at(features_view X, floats_out y_hat, size_t n_rounds) const override
    {
        size_t const n = X.extent(0);
        assert(y_hat.size() == n);
        auto const scores = raw_scores(X, n_rounds);
        parallel::for_each_index(n,
                                 [&](size_t i)
                                 {
                                     size_t best = 0;
                                     for (size_t k = 1; k < n_classes_; ++k)
                                     {
                                         if (scores[(i * n_classes_) + k] >
                                             scores[(i * n_classes_) + best])
                                         {
                                             best = k;
                                         }
                                     }
                                     y_hat[i] = static_cast<float>(best);
                                 });
    }

    // Multiclass logloss on raw scores.
    float eval(features_view X, floats_view labels) const override
    {
        size_t const n      = X.extent(0);
        auto const   scores = raw_scores(X, 0);
        double       total  = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            double maxv = scores[i * n_classes_];
            for (size_t k = 1; k < n_classes_; ++k)
            {
                maxv =
                    std::max(maxv, static_cast<double>(scores[(i * n_classes_) + k]));
            }
            double sum = 0.0;
            for (size_t k = 0; k < n_classes_; ++k)
            {
                sum += std::exp(scores[(i * n_classes_) + k] - maxv);
            }
            size_t const y = class_of(labels[i], n_classes_);
            total -= (scores[(i * n_classes_) + y] - maxv) - std::log(sum);
        }
        return static_cast<float>(total / static_cast<double>(n));
    }

    size_t n_iters() const override
    {
        return trees_.size() / n_classes_;
    }

    std::vector<double> feature_importance(ImportanceType type) const override
    {
        std::vector<double> out;
        for (auto const &tree : trees_)
        {
            internal::accumulate_importance(tree, type, out);
        }
        return out;
    }

    void predict_staged(features_view X, floats_out out) const override
    {
        size_t const n = X.extent(0);
        for (size_t r = 0; r < n_iters(); ++r)
        {
            predict_at(X, floats_out{out.data() + (r * n), n}, r + 1);
        }
    }

    void predict_leaf(features_view X, std::span<node_id_t> out) const override
    {
        size_t const n = X.extent(0);
        assert(out.size() == n * trees_.size());
        size_t const n_trees = trees_.size();
        parallel::for_each_index(n,
                                 [&](size_t i)
                                 {
                                     for (size_t t = 0; t < n_trees; ++t)
                                     {
                                         out[(i * n_trees) + t] = trees_[t].leaf_for(
                                             X, static_cast<row_id_t>(i));
                                     }
                                 });
    }

    std::string dump(std::span<std::string const> feature_names) const override
    {
        std::string out;
        for (size_t t = 0; t < trees_.size(); ++t)
        {
            out += "tree " + std::to_string(t / n_classes_) + " class " +
                   std::to_string(t % n_classes_) + ":\n";
            internal::dump_tree(trees_[t], feature_names, out);
        }
        return out;
    }

    void pred_contribs(features_view /*X*/, std::span<double> /*out*/,
                       size_t /*n_features*/) const override
    {
        throw std::runtime_error(
            "pred_contribs is not supported for multiclass models yet");
    }

    float score_base() const override
    {
        throw std::runtime_error(
            "early stopping is not supported for multiclass models yet");
    }
    void accumulate_last_tree(features_view /*X*/, floats_out /*scores*/) const override
    {
        throw std::runtime_error(
            "early stopping is not supported for multiclass models yet");
    }

    void truncate(size_t n_rounds) override
    {
        size_t const keep = n_rounds * n_classes_;
        if (keep < trees_.size())
        {
            trees_.erase(trees_.begin() + static_cast<std::ptrdiff_t>(keep),
                         trees_.end());
        }
    }

    // Save/load accessors, mirroring Booster's shape plus the extras the
    // multiclass envelope needs.
    std::vector<tree_type> const &trees() const
    {
        return trees_;
    }
    std::vector<float> const &init_scores() const
    {
        return init_scores_;
    }
    size_t n_classes() const
    {
        return n_classes_;
    }
    void load_state(std::vector<tree_type> trees, std::vector<float> init_scores)
    {
        trees_       = std::move(trees);
        init_scores_ = std::move(init_scores);
    }

  private:
    static size_t class_of(float label, size_t n_k)
    {
        auto const k = std::lround(label);
        return k >= 0 && static_cast<size_t>(k) < n_k ? static_cast<size_t>(k) : 0;
    }

    // Raw (init + lr * tree sums) scores, n_rows x K, using the first
    // n_rounds rounds (0 = all).
    std::vector<float> raw_scores(features_view X, size_t n_rounds) const
    {
        size_t const n      = X.extent(0);
        size_t const rounds = n_rounds == 0 ? n_iters() : std::min(n_rounds, n_iters());
        std::vector<float> scores(n * n_classes_, 0.0F);
        std::vector<float> raw(n);
        for (size_t k = 0; k < n_classes_; ++k)
        {
            std::ranges::fill(raw, 0.0F);
            for (size_t r = 0; r < rounds; ++r)
            {
                trees_[(r * n_classes_) + k].predict(X, raw);
            }
            for (size_t i = 0; i < n; ++i)
            {
                scores[(i * n_classes_) + k] =
                    init_scores_.empty()
                        ? config_.learning_rate * raw[i]
                        : init_scores_[k] + (config_.learning_rate * raw[i]);
            }
        }
        return scores;
    }

    BoosterConfig          config_;
    size_t                 n_classes_;
    grower_type            grower_;
    sampler_type           sampler_;
    std::mt19937           rng_;
    std::vector<tree_type> trees_;
    std::vector<float>     scores_;      // n_rows x K training accumulator
    std::vector<float>     init_scores_; // per-class log prior
    std::vector<float>     grad_;
    std::vector<float>     hess_;
    std::vector<row_id_t>  row_indices_;
};

} // namespace bonsai
