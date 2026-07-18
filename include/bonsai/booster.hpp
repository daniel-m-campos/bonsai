#pragma once

#include "bonsai/config/booster_config.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/detail/perf.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/objective.hpp"
#include "bonsai/objective_traits.hpp"
#include "bonsai/parallel.hpp"
#include "bonsai/sampler.hpp"
#include "bonsai/shap.hpp"
#include "bonsai/types.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <numeric>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace bonsai
{

// How to score a feature's contribution across the ensemble:
//   split — number of times the feature is chosen for a split
//   gain  — total loss reduction from those splits (usually what you want)
enum class ImportanceType : uint8_t
{
    split,
    gain,
};

class IBooster
{
  public:
    virtual ~IBooster() = default;

    // The one type-erased boundary (with IngestPlane) in the system: three
    // client groups share it deliberately rather than splitting into three
    // interfaces — grouped below as training / prediction / introspection /
    // the training-loop seam. If a third training-loop client ever appears,
    // split that group into its own view (design review 2026-07-12).

    // --- training
    virtual void  update_one_iter(Dataset const &train)           = 0;
    virtual float eval(features_view X, floats_view labels) const = 0;
    // --- prediction
    virtual void   predict(features_view X, floats_out y_hat) const = 0;
    virtual size_t n_iters() const                                  = 0;

    // Per-feature importance summed over all trees, sized max feature id + 1
    // (callers pad to the full feature count).
    // --- introspection
    virtual std::vector<double> feature_importance(ImportanceType type) const = 0;

    // Predict using only the first n_trees trees (0 = all). The plain
    // predict(X, out) is predict_at(X, out, 0).
    virtual void predict_at(features_view X, floats_out y_hat,
                            size_t n_trees) const = 0;

    // Per-iteration predictions in one pass: out is n_iters() * n_rows,
    // row-major by iteration (out[k*n_rows + i] = prediction of row i using
    // the first k+1 trees).
    virtual void predict_staged(features_view X, floats_out out) const = 0;

    // Per-class probabilities: out is n_rows * score_width(), row-major. Only
    // the multiclass (softmax) booster implements it — a row-wise softmax of
    // the class logits. Width-1 objectives expose P(class 1) via predict(),
    // so the default throws.
    virtual void predict_proba(features_view /*X*/, std::span<double> /*out*/) const
    {
        throw std::logic_error("predict_proba: per-class probabilities are only "
                               "available for the multiclass (softmax) objective; "
                               "width-1 objectives expose P via predict()");
    }

    // Per-row, per-tree leaf indices (DenseTree node ids / ObliviousTree
    // table indices): out is n_rows * n_iters(), row-major by row.
    virtual void predict_leaf(features_view X, std::span<node_id_t> out) const = 0;

    // Human-readable dump of every tree (feature names optional).
    virtual std::string dump(std::span<std::string const> feature_names) const = 0;

    // TreeSHAP contributions: out is n_rows * (n_features + 1), row-major,
    // last column = bias (init score + expected tree values). Rows sum to
    // the raw prediction exactly. Throws for models without covers (saved
    // before covers were recorded); multiclass fills one slice per class.
    virtual void pred_contribs(features_view X, std::span<double> out,
                               size_t n_features) const = 0;

    // --- the training-loop seam (CLI pipeline only)
    // Incremental prediction support for early stopping, shape-agnostic so
    // multiclass composes: the caller maintains a raw-score matrix of
    // n_rows x score_width() (row-major, width 1 except softmax).
    // seed_valid_scores fills it as of n_rounds boosting rounds (0 = base
    // scores only — the warm-start seam); accumulate_last_round adds the
    // newest round's tree(s); valid_loss scores it with the booster's own
    // configured objective.
    virtual size_t score_width() const
    {
        return 1;
    }
    virtual void  seed_valid_scores(features_view X, std::span<float> out,
                                    size_t n_rounds) const                        = 0;
    virtual void  accumulate_last_round(features_view X, floats_out scores) const = 0;
    virtual float valid_loss(std::span<float const> scores,
                             floats_view            labels) const                            = 0;
    // Drop trees beyond the first n_trees (keep the best iteration's model).
    virtual void truncate(size_t n_trees) = 0;
};

namespace internal
{

// Accumulate a tree's (unscaled-by-lr) train contribution into `out` by
// routing rows in bin space. ds.bin_of_threshold recovers each internal
// node's split bin from its stored threshold. Used by DART to subtract
// dropped trees without caching per-tree train predictions.
inline void accumulate_train_contribution(DenseTree const &tree, Dataset const &ds,
                                          floats_out out)
{
    auto const           &nodes = tree.nodes();
    std::vector<bin_id_t> tbin(nodes.size(), 0);
    for (size_t i = 0; i < nodes.size(); ++i)
    {
        if (!DenseTree::is_leaf(nodes[i]))
        {
            tbin[i] =
                ds.bin_of_threshold(nodes[i].feature_id, nodes[i].threshold_or_value);
        }
    }
    parallel::for_each_index(
        ds.n_rows(),
        [&](size_t r)
        {
            node_id_t idx = 0;
            while (!DenseTree::is_leaf(nodes[idx]))
            {
                auto const &nd   = nodes[idx];
                auto const  last = static_cast<bin_id_t>(ds.n_bins(nd.feature_id) - 1);
                bin_id_t const b = ds.bin_at(nd.feature_id, r);
                bool const     left = routes_left(b, last, tbin[idx], nd.default_left);
                idx                 = left ? nd.left : nd.right;
            }
            out[r] += nodes[idx].threshold_or_value;
        });
}

inline void accumulate_train_contribution(ObliviousTree const &tree, Dataset const &ds,
                                          floats_out out)
{
    auto const           &splits = tree.splits();
    std::vector<bin_id_t> tbin(splits.size(), 0);
    for (size_t lvl = 0; lvl < splits.size(); ++lvl)
    {
        tbin[lvl] = ds.bin_of_threshold(splits[lvl].feature_id, splits[lvl].threshold);
    }
    parallel::for_each_index(
        ds.n_rows(),
        [&](size_t r)
        {
            size_t index = 0;
            for (size_t lvl = 0; lvl < splits.size(); ++lvl)
            {
                auto const &s    = splits[lvl];
                auto const  last = static_cast<bin_id_t>(ds.n_bins(s.feature_id) - 1);
                bin_id_t const b = ds.bin_at(s.feature_id, r);
                bool const     left = routes_left(b, last, tbin[lvl], s.default_left);
                index               = (index << 1U) | (left ? 0U : 1U);
            }
            out[r] += tree.leaf_table()[index];
        });
}

inline std::string feature_label(std::span<std::string const> names, size_t f)
{
    return f < names.size() ? names[f] : "f" + std::to_string(f);
}

// Indented text dump, one line per node.
inline void dump_tree(DenseTree const &tree, std::span<std::string const> names,
                      std::string &out)
{
    auto const &nodes = tree.nodes();
    auto const &gains = tree.split_gains();
    // NOLINTNEXTLINE(misc-no-recursion)
    auto walk = [&](auto const &self, node_id_t id, int depth) -> void
    {
        out.append(static_cast<size_t>(depth) * 2, ' ');
        auto const &n = nodes[id];
        if (DenseTree::is_leaf(n))
        {
            out += "leaf=" + std::to_string(n.threshold_or_value);
            if (id < tree.covers().size())
            {
                out +=
                    " cover=" + std::to_string(static_cast<size_t>(tree.covers()[id]));
            }
            out += "\n";
            return;
        }
        out += feature_label(names, n.feature_id) +
               " <= " + std::to_string(n.threshold_or_value) +
               (n.default_left ? " [nan->left]" : " [nan->right]") +
               " gain=" + std::to_string(id < gains.size() ? gains[id] : 0.0F);
        if (id < tree.covers().size())
        {
            out += " cover=" + std::to_string(static_cast<size_t>(tree.covers()[id]));
        }
        out += "\n";
        self(self, n.left, depth + 1);
        self(self, n.right, depth + 1);
    };
    walk(walk, 0, 0);
}

inline void dump_tree(ObliviousTree const &tree, std::span<std::string const> names,
                      std::string &out)
{
    auto const &splits = tree.splits();
    auto const &gains  = tree.level_gains();
    for (size_t lvl = 0; lvl < splits.size(); ++lvl)
    {
        out += "level " + std::to_string(lvl) + ": " +
               feature_label(names, splits[lvl].feature_id) +
               " <= " + std::to_string(splits[lvl].threshold) +
               (splits[lvl].default_left ? " [nan->left]" : " [nan->right]") +
               " gain=" + std::to_string(lvl < gains.size() ? gains[lvl] : 0.0F) + "\n";
    }
    out += "leaves:";
    for (float const v : tree.leaf_table())
    {
        out += " " + std::to_string(v);
    }
    out += "\n";
    if (!tree.leaf_covers().empty())
    {
        out += "covers:";
        for (float const c : tree.leaf_covers())
        {
            out += " " + std::to_string(static_cast<size_t>(c));
        }
        out += "\n";
    }
}

// One tree's contribution to per-feature importance.
inline void accumulate_importance(DenseTree const &tree, ImportanceType type,
                                  std::vector<double> &out)
{
    auto const &nodes = tree.nodes();
    auto const &gains = tree.split_gains();
    for (size_t i = 0; i < nodes.size(); ++i)
    {
        if (DenseTree::is_leaf(nodes[i]))
        {
            continue;
        }
        size_t const f = nodes[i].feature_id;
        if (out.size() <= f)
        {
            out.resize(f + 1, 0.0);
        }
        out[f] +=
            type == ImportanceType::split ? 1.0 : (i < gains.size() ? gains[i] : 0.0F);
    }
}

inline void accumulate_importance(ObliviousTree const &tree, ImportanceType type,
                                  std::vector<double> &out)
{
    auto const &splits = tree.splits();
    auto const &gains  = tree.level_gains();
    for (size_t lvl = 0; lvl < splits.size(); ++lvl)
    {
        size_t const f = splits[lvl].feature_id;
        if (out.size() <= f)
        {
            out.resize(f + 1, 0.0);
        }
        out[f] += type == ImportanceType::split
                      ? 1.0
                      : (lvl < gains.size() ? gains[lvl] : 0.0F);
    }
}

inline void shap_one_row(DenseTree const &tree, features_view X, row_id_t row,
                         std::span<double> phi)
{
    tree_shap(tree, X, row, phi);
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
        : config_(config.booster_config), objective_(config),
          grower_(config.tree_config), sampler_(config),
          rng_(config.booster_config.random_seed)
    {
    }

    void update_one_iter(Dataset const &train) override
    {
        if (scores_.empty())
        {
            grad_.resize(train.n_rows());
            hess_.resize(train.n_rows());
            if (trees_.empty())
            {
                init_score_ = objective_.init_score(train.labels());
                scores_.assign(train.n_rows(), init_score_);
            }
            else
            {
                // Warm start: the booster was loaded with trees but no
                // training state. Rebuild every row's score by routing the
                // existing trees over the binned data.
                std::vector<float> raw(train.n_rows(), 0.0F);
                for (auto const &t : trees_)
                {
                    internal::accumulate_train_contribution(t, train, raw);
                }
                scores_.resize(train.n_rows());
                parallel::for_each_index(
                    train.n_rows(), [&](size_t i)
                    { scores_[i] = init_score_ + (config_.learning_rate * raw[i]); });
            }
        }

        if (try_resident_round(train))
        {
            return;
        }

        auto                    &prof = detail::FitProfiler::instance();
        detail::FitProfiler::Lap lap;

        // DART: drop a random subset of existing trees; gradients are
        // computed against the model without them, and after the grow both
        // the dropped trees and the new tree are rescaled (apply_dart_round).
        std::vector<size_t> dropped;
        std::vector<float>  dropped_sum;
        drop_dart_trees(train, dropped, dropped_sum);

        lap(prof.dart_s);
        objective_.compute(scores_, train.labels(), grad_, hess_);

        if (!train.weights().empty())
        {
            for (size_t i = 0; i < grad_.size(); ++i)
            {
                grad_[i] *= train.weights()[i];
                hess_[i] *= train.weights()[i];
            }
        }
        lap(prof.objective_s);

        size_t const n_selected = refill_row_indices(train);
        lap(prof.sample_s);

        auto [tree, leaf_values, leaf_ids] =
            grower_.grow(train, grad_, hess_, {row_indices_.data(), n_selected});
        lap(prof.grow_s);

        // Leaf renewal (constant-hessian objectives): replace each leaf's
        // Newton step with the objective's optimal value over the residuals
        // of the rows it covers. scores_ still exclude this tree (and, under
        // DART, the dropped trees) — exactly the state gradients used.
        if constexpr (requires(std::span<float> r) { objective_.renew_leaf(r); })
        {
            renew_leaves(tree, leaf_ids, leaf_values, train.labels());
        }
        lap(prof.renew_s);

        if (!dropped.empty())
        {
            apply_dart_round(tree, dropped, dropped_sum, leaf_values);
        }
        else
        {
            parallel::for_each_index(
                scores_.size(), [&](size_t i)
                { scores_[i] += config_.learning_rate * leaf_values[i]; });
        }
        lap(prof.score_s);

        trees_.push_back(std::move(tree));
        // Hand the output buffers back for the next tree (skips the
        // per-tree zero-init; grower.hpp documents the write-before-read
        // contract).
        grower_.recycle(std::move(leaf_values), std::move(leaf_ids));
    }

    // Fills row_indices_ for this tree. AllRowsSampler is deterministic
    // identity, so once row_indices_ holds the full iota only its size need be
    // checked: the per-tree refill (a measurable membw cost at scale) is
    // skipped. The content is byte-identical to calling sample() every tree, so
    // the model is unchanged. Other samplers draw fresh indices each tree.
    size_t refill_row_indices(Dataset const &train)
    {
        if constexpr (std::same_as<sampler_type, AllRowsSampler>)
        {
            if (row_indices_.size() != train.n_rows())
            {
                row_indices_.resize(train.n_rows());
                std::iota(row_indices_.begin(), row_indices_.end(), row_id_t{0});
            }
            return row_indices_.size();
        }
        else
        {
            if (row_indices_.size() != train.n_rows())
            {
                row_indices_.resize(train.n_rows());
            }
            return sampler_.sample(grad_, hess_, rng_, row_indices_);
        }
    }

    // Device-resident objective: when the grower keeps labels and scores on
    // the GPU and derives grad/hess there, the whole host objective / score
    // round-trip is skipped and this returns true. Gated at compile time on
    // the objective (must have a device gradient) and the sampler (must not
    // read gradients), and at run time on no DART and the escape hatch. Sample
    // weights are handled device-side (the gradient kernel scales grad/hess by
    // the resident weight), so a weighted fit stays eligible. The resident
    // state is armed for ONE Dataset: a different one (or a runtime gate
    // flipping) syncs scores home and disarms, so the host path always resumes
    // with the same state it would have had.
    bool try_resident_round(Dataset const &train)
    {
        if constexpr (device_objective_kind<objective_type> !=
                          DeviceObjectiveKind::none &&
                      (std::same_as<sampler_type, AllRowsSampler> ||
                       std::same_as<sampler_type, BernoulliSampler>) )
        {
            bool const host_forced = std::getenv("BONSAI_HOST_OBJECTIVE") != nullptr;
            bool const runtime_ok  = config_.dart_drop_rate <= 0.0F && !host_forced;
            if (resident_active_ && (!runtime_ok || resident_train_ != &train))
            {
                grower_.resident_end(std::span<float>{scores_});
                resident_active_ = false;
                resident_train_  = nullptr;
            }
            if (runtime_ok && !resident_active_)
            {
                resident_active_ = grower_.resident_begin(
                    train, device_objective_kind<objective_type>,
                    std::span<float const>{scores_}, config_.learning_rate);
                resident_train_ = resident_active_ ? &train : nullptr;
            }
            if (resident_active_)
            {
                resident_round(train);
                return true;
            }
        }
        return false;
    }

    // One boosting round with the resident objective armed: no host objective,
    // no weights loop, no leaf renewal (the eligible objectives have none), no
    // host score update. The sampler still runs (Bernoulli needs its indices;
    // AllRows is the freebie above) and grow returns an empty per-row output:
    // the device already derived the gradients and fused the score update.
    void resident_round(Dataset const &train)
    {
        auto                    &prof = detail::FitProfiler::instance();
        detail::FitProfiler::Lap lap;
        size_t const             n_selected = refill_row_indices(train);
        lap(prof.sample_s);
        auto res = grower_.grow(train, floats_view{}, floats_view{},
                                {row_indices_.data(), n_selected});
        lap(prof.grow_s);
        trees_.push_back(std::move(res.tree));
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

    // Group rows by leaf, hand each leaf's residuals to the objective, and
    // overwrite both the tree's leaf values and the per-row training values.
    // DART pre-grow half: pick this round's dropped trees and remove their
    // contribution from scores_, so gradients see the model without them.
    void drop_dart_trees(Dataset const &train, std::vector<size_t> &dropped,
                         std::vector<float> &dropped_sum)
    {
        if (config_.dart_drop_rate <= 0.0F || trees_.empty())
        {
            return;
        }
        std::bernoulli_distribution drop(config_.dart_drop_rate);
        for (size_t i = 0; i < trees_.size(); ++i)
        {
            if (drop(rng_))
            {
                dropped.push_back(i);
            }
        }
        if (dropped.empty())
        {
            return;
        }
        dropped_sum.assign(train.n_rows(), 0.0F);
        for (size_t const i : dropped)
        {
            internal::accumulate_train_contribution(trees_[i], train, dropped_sum);
        }
        parallel::for_each_index(
            scores_.size(),
            [&](size_t i) { scores_[i] -= config_.learning_rate * dropped_sum[i]; });
    }

    // DART post-grow half, xgboost's normalize_type="tree" factors: the new
    // tree lands with weight lr/(k+lr) — comparable to a plain shrinkage
    // step — and each dropped tree shrinks by k/(k+lr). (The original DART
    // paper's 1/(k+1) assumes unshrunk trees and starves the new tree by
    // ~1/lr when combined with a learning rate.) scores_ currently exclude
    // the dropped trees entirely; add back their rescaled contribution plus
    // the scaled new tree.
    void apply_dart_round(tree_type &tree, std::vector<size_t> const &dropped,
                          std::vector<float> const &dropped_sum,
                          train_leaf_values const  &leaf_values)
    {
        auto const  k         = static_cast<float>(dropped.size());
        float const new_scale = 1.0F / (k + config_.learning_rate);
        float const old_scale = k / (k + config_.learning_rate);
        tree.scale_leaves(new_scale);
        for (size_t const i : dropped)
        {
            trees_[i].scale_leaves(old_scale);
        }
        parallel::for_each_index(scores_.size(),
                                 [&](size_t i)
                                 {
                                     scores_[i] += config_.learning_rate *
                                                   ((old_scale * dropped_sum[i]) +
                                                    (new_scale * leaf_values[i]));
                                 });
    }

    void renew_leaves(tree_type &tree, std::vector<node_id_t> const &leaf_ids,
                      train_leaf_values &leaf_values, floats_view labels)
    {
        std::unordered_map<node_id_t, std::vector<float>> residuals;
        for (size_t r = 0; r < leaf_ids.size(); ++r)
        {
            residuals[leaf_ids[r]].push_back(labels[r] - scores_[r]);
        }
        std::unordered_map<node_id_t, float> renewed;
        renewed.reserve(residuals.size());
        for (auto &[leaf, res] : residuals)
        {
            float const v = objective_.renew_leaf(std::span<float>{res});
            tree.set_leaf_value(leaf, v);
            renewed.emplace(leaf, v);
        }
        for (size_t r = 0; r < leaf_ids.size(); ++r)
        {
            leaf_values[r] = renewed.at(leaf_ids[r]);
        }
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

    void predict_at(features_view X, floats_out scores, size_t n_trees) const override
    {
        assert(X.extent(0) == scores.size());
        size_t const k =
            n_trees == 0 ? trees_.size() : std::min(n_trees, trees_.size());
        std::fill(scores.begin(), scores.end(), 0.0F);
        for (size_t t = 0; t < k; ++t)
        {
            trees_[t].predict(X, scores);
        }
        for (float &score : scores)
        {
            score = init_score_ + (score * config_.learning_rate);
        }
    }

    void predict_staged(features_view X, floats_out out) const override
    {
        size_t const n = X.extent(0);
        assert(out.size() == n * trees_.size());
        std::vector<float> raw(n, 0.0F);
        for (size_t t = 0; t < trees_.size(); ++t)
        {
            trees_[t].predict(X, raw);
            parallel::for_each_index(
                n, [&](size_t i)
                { out[(t * n) + i] = init_score_ + (raw[i] * config_.learning_rate); });
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
            out += "tree " + std::to_string(t) + ":\n";
            internal::dump_tree(trees_[t], feature_names, out);
        }
        return out;
    }

    void pred_contribs(features_view X, std::span<double> out,
                       size_t n_features) const override
    {
        if constexpr (std::same_as<tree_type, ObliviousTree>)
        {
            // Expand once per tree (2^depth nodes), not per row: TreeSHAP's
            // cover-weighted walk then runs unchanged on the dense shape.
            std::vector<DenseTree> dense;
            dense.reserve(trees_.size());
            for (auto const &tree : trees_)
            {
                dense.push_back(dense_equivalent(tree));
            }
            contribs_over(dense, X, out, n_features);
        }
        else
        {
            contribs_over(trees_, X, out, n_features);
        }
    }

    template <typename Trees>
    void contribs_over(Trees const &trees, features_view X, std::span<double> out,
                       size_t n_features) const
    {
        size_t const n    = X.extent(0);
        size_t const cols = n_features + 1;
        assert(out.size() == n * cols);
        parallel::for_each_index(
            n,
            [&](size_t i)
            {
                std::span<double> const phi = out.subspan(i * cols, cols);
                std::ranges::fill(phi, 0.0);
                for (auto const &tree : trees)
                {
                    internal::shap_one_row(tree, X, static_cast<row_id_t>(i), phi);
                }
                for (double &v : phi)
                {
                    v *= config_.learning_rate;
                }
                phi[n_features] += init_score_;
            });
    }

    void seed_valid_scores(features_view X, std::span<float> out,
                           size_t n_rounds) const override
    {
        if (n_rounds > 0)
        {
            predict_at(X, floats_out{out.data(), out.size()}, n_rounds);
            return;
        }
        std::ranges::fill(out, init_score_);
    }

    float valid_loss(std::span<float const> scores, floats_view labels) const override
    {
        return objective_.eval(floats_view{scores.data(), scores.size()}, labels);
    }

    void accumulate_last_round(features_view X, floats_out scores) const override
    {
        assert(!trees_.empty());
        assert(X.extent(0) == scores.size());
        std::vector<float> raw(scores.size(), 0.0F);
        trees_.back().predict(X, raw);
        parallel::for_each_index(scores.size(), [&](size_t i)
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
    // Stale while the resident objective is armed (the device copy is
    // authoritative); resident_end syncs it before any host-path read.
    std::vector<float>    scores_;
    std::vector<float>    grad_;
    std::vector<float>    hess_;
    std::vector<row_id_t> row_indices_;
    float                 init_score_      = 0.0F;
    bool                  resident_active_ = false;
    // Identity cookie for the Dataset the resident state was armed on:
    // compared by address only, never dereferenced through.
    Dataset const *resident_train_ = nullptr;
};

} // namespace bonsai
