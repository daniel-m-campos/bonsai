#pragma once

#include "bonsai/config/booster_config.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/detail/bin_walk.hpp"
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
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
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

// What a device predict plan needs from a booster: the dense ensemble to pack
// in bin space, the scale and base every prediction applies to the packed
// sum, and the mutation epoch that says when a cached plan is stale.
struct PredictPlanInput
{
    std::span<DenseTree const> trees;
    float                      learning_rate = 0.0F;
    float                      init_score    = 0.0F;
    uint64_t                   epoch         = 0;
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

    // Trees in the ensemble, which is n_iters() only for width-1 objectives:
    // multiclass grows one tree per class per round. Per-tree outputs must be
    // sized from this, never from n_iters().
    virtual size_t n_trees() const = 0;

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
    // table indices): out is n_rows * n_trees(), row-major by row. Trees are
    // stored round-major, so multiclass column t holds round t / n_classes,
    // class t % n_classes.
    virtual void predict_leaf(features_view X, std::span<node_id_t> out) const = 0;

    // Human-readable dump of every tree (feature names optional).
    virtual std::string dump(std::span<std::string const> feature_names) const = 0;

    // TreeSHAP contributions: out is n_rows * (n_features + 1), row-major,
    // last column = bias (init score + expected tree values). Rows sum to
    // the raw prediction exactly. Throws for models without covers (saved
    // before covers were recorded); multiclass fills one slice per class.
    virtual void pred_contribs(features_view X, std::span<double> out,
                               size_t n_features) const = 0;

    // --- prediction over a Dataset the caller already binned
    // Bin-space routing is exact under the model's own cuts, so each of these
    // is bit-identical to its raw twin above (same outputs, same layouts) at a
    // quarter the bytes. The Dataset must carry the cuts the model was grown
    // on; a Dataset binned with other mappers routes to other leaves.
    virtual void predict_at_binned(Dataset const &bins, floats_out scores,
                                   size_t n_trees) const = 0;

    virtual void predict_staged_binned(Dataset const &bins, floats_out out) const = 0;

    virtual void predict_leaf_binned(Dataset const       &bins,
                                     std::span<node_id_t> out) const = 0;

    virtual void predict_proba_binned(Dataset const & /*bins*/,
                                      std::span<double> /*out*/) const
    {
        throw std::logic_error("predict_proba: per-class probabilities are only "
                               "available for the multiclass (softmax) objective; "
                               "width-1 objectives expose P via predict()");
    }

    virtual void pred_contribs_binned(Dataset const &bins, std::span<double> out,
                                      size_t n_features) const = 0;

    // --- the device predict seam
    // Everything a device predict plan packs, in one call so the ensemble is
    // read once. An empty `trees` declines: only a dense ensemble packs, so
    // oblivious and multiclass models ride the host bin walk. `trees` stays
    // valid until the booster mutates, which `epoch` reports.
    virtual PredictPlanInput predict_plan_input() const
    {
        return {};
    }

    // --- the training-loop seam (CLI pipeline only)
    // Incremental prediction support for early stopping, shape-agnostic so
    // multiclass composes: the caller maintains a raw-score matrix of
    // n_rows x score_width() (row-major, width 1 except softmax).
    // seed_validation_scores fills it as of n_rounds boosting rounds (0 =
    // base scores only, the warm-start seam); accumulate_last_round adds the
    // newest round's tree(s) by routing the raw float rows, and
    // accumulate_last_round_binned routes the same rows in bin space for a
    // caller that has binned them (identical routing, a quarter the bytes);
    // validation_loss scores the matrix with the booster's own configured
    // objective.
    // What `bins` owes accumulate_last_round_binned is the caller's to
    // supply, and each implementation's asserts spell it out.
    virtual size_t score_width() const
    {
        return 1;
    }

    virtual void seed_validation_scores(features_view X, std::span<float> out,
                                        size_t n_rounds) const = 0;

    virtual void accumulate_last_round(features_view X, floats_out scores) const = 0;

    virtual void accumulate_last_round_binned(Dataset const &bins,
                                              floats_out     scores) const = 0;

    virtual float validation_loss(std::span<float const> scores,
                                  floats_view            labels) const = 0;

    // Device validation seam. begin_resident_validation arms the grower's eval
    // plane with the binned validation rows and the seeded scores; false keeps
    // the host walk. accumulate_last_round_resident walks the newest tree on
    // that plane; false means the plane declined and the caller must run the
    // host walk for this round. When the objective has a device loss, the
    // round's loss comes back through `loss` and the scores stay on the
    // device; otherwise the updated scores come back through `scores` for the
    // host loss pass.
    virtual bool begin_resident_validation(Dataset const & /*bins*/,
                                           std::span<float const> /*seed*/)
    {
        return false;
    }

    virtual bool accumulate_last_round_resident(Dataset const & /*bins*/,
                                                floats_out /*scores*/,
                                                std::optional<float> & /*loss*/)
    {
        return false;
    }

    // Drop trees beyond the first n_trees (keep the best iteration's model).
    virtual void truncate(size_t n_trees) = 0;
};

namespace internal
{

// The bin-space walk lives in detail/bin_walk.hpp so src/shap.cpp can reach it
// too; these keep every existing call site spelled internal::.
using detail::leaf_binned;
using detail::split_bins;
using detail::SplitBins;
using detail::value_binned;

// Accumulate a tree's (unscaled-by-lr) contribution over a binned Dataset's
// rows, routing the columns the tree was grown on. Used by DART to subtract
// dropped trees without caching per-tree train predictions, by warm start, and
// by the binned prediction family.
template <Tree T>
void accumulate_train_contribution(T const &tree, Dataset const &ds, floats_out out)
{
    auto const sb = split_bins(tree, ds);
    parallel::for_each_index(ds.n_rows(),
                             [&](size_t r)
                             {
                                 out[r] += value_binned(tree, sb, [&](size_t f)
                                                        { return ds.bin_at(f, r); });
                             });
}

// accumulate_train_contribution's reader twin: one output per VIEW row, in the
// view's order, which is the shape every reader over a row view answers in.
// Identical to the twin above when the dataset is not a view.
template <Tree T>
void accumulate_view_contribution(T const &tree, Dataset const &ds, floats_out out)
{
    auto const     sb = split_bins(tree, ds);
    RowIndex const rows{ds.row_view()};
    parallel::for_each_index(rows.size(),
                             [&](size_t k)
                             {
                                 row_id_t const r = rows[k];
                                 out[k] += value_binned(tree, sb, [&](size_t f)
                                                        { return ds.bin_at(f, r); });
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

// The per-tree biases a contribs batch shares across its rows: the expected
// value is row-independent, so one walk per tree replaces one per (row, tree).
template <typename Trees> std::vector<double> shap_biases(Trees const &trees)
{
    std::vector<double> biases;
    biases.reserve(trees.size());
    for (auto const &tree : trees)
    {
        biases.push_back(tree_expected_value(tree));
    }
    return biases;
}

// Per-row, per-tree leaf indices; out is n_rows * trees.size(), row-major by
// row. Both boosters store trees flat (multiclass round-major), so the walk
// is the same one.
template <typename Trees>
void predict_leaf_over(Trees const &trees, features_view X, std::span<node_id_t> out)
{
    size_t const n       = X.extent(0);
    size_t const n_trees = trees.size();
    assert(out.size() == n * n_trees);
    parallel::for_each_index(n,
                             [&](size_t i)
                             {
                                 for (size_t t = 0; t < n_trees; ++t)
                                 {
                                     out[(i * n_trees) + t] =
                                         trees[t].leaf_for(X, static_cast<row_id_t>(i));
                                 }
                             });
}

// The per-tree SplitBins a binned batch shares across its rows: the threshold
// inversion is row-independent, so one walk per tree replaces one per (row,
// tree). The bias hoist's counterpart for routing.
template <typename Trees>
std::vector<SplitBins> tree_split_bins(Trees const &trees, Dataset const &bins)
{
    std::vector<SplitBins> sb;
    sb.reserve(trees.size());
    for (auto const &tree : trees)
    {
        sb.push_back(split_bins(tree, bins));
    }
    return sb;
}

// predict_leaf_over's twin over binned rows, same row-major-by-row output.
template <typename Trees>
void predict_leaf_over_binned(Trees const &trees, Dataset const &bins,
                              std::span<node_id_t> out)
{
    RowIndex const rows{bins.row_view()};
    size_t const   n_trees = trees.size();
    assert(out.size() == rows.size() * n_trees);
    auto const sb = tree_split_bins(trees, bins);
    parallel::for_each_index(
        rows.size(),
        [&](size_t k)
        {
            row_id_t const r      = rows[k];
            auto const     bin_of = [&](size_t f) { return bins.bin_at(f, r); };
            for (size_t t = 0; t < n_trees; ++t)
            {
                out[(k * n_trees) + t] = leaf_binned(trees[t], sb[t], bin_of);
            }
        });
}

// TreeSHAP's cover-weighted walk is written against the dense shape, so an
// oblivious ensemble is expanded once per tree (2^depth nodes) rather than
// once per row.
inline std::vector<DenseTree> densify(std::vector<ObliviousTree> const &trees)
{
    std::vector<DenseTree> dense;
    dense.reserve(trees.size());
    for (auto const &tree : trees)
    {
        dense.push_back(dense_equivalent(tree));
    }
    return dense;
}

// The dense equivalents cached per mutation epoch, so repeated explain calls
// between fits pay the conversion once. Readers are const and may be
// concurrent (the bindings release the GIL), hence the lock; a mutation
// never touches the cache, it just bumps the booster's epoch. Booster epochs
// start at 1, so epoch_ = 0 is the never-filled state and needs no flag.
class DensifyCache
{
  public:
    std::shared_ptr<std::vector<DenseTree> const>
    get(std::vector<ObliviousTree> const &trees, uint64_t epoch) const
    {
        std::scoped_lock const lock(mutex_);
        if (epoch_ != epoch)
        {
            cache_ = std::make_shared<std::vector<DenseTree> const>(densify(trees));
            epoch_ = epoch;
        }
        return cache_;
    }

  private:
    mutable std::mutex                                    mutex_;
    mutable std::shared_ptr<std::vector<DenseTree> const> cache_;
    mutable uint64_t                                      epoch_ = 0;
};

// A value paired with its mutation counter. Readers take read(); every writer
// goes through mutate(), which bumps the epoch before handing the value over,
// so a derived view (the dense SHAP cache, the device plans) can never see a
// changed value under an unchanged epoch. The counter is monotonic, so several
// mutations in one round are fine.
template <typename T> class Versioned
{
  public:
    T const &read() const
    {
        return value_;
    }
    uint64_t epoch() const
    {
        return epoch_;
    }
    T &mutate()
    {
        ++epoch_;
        return value_;
    }

  private:
    T        value_;
    uint64_t epoch_ = 1;
};

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
            if (trees_.read().empty())
            {
                // The init score is a statistic of the rows this fit
                // visits, which for a view is a subset of the plane's. On the
                // identity gather_rows copies, so the full-data path keeps
                // the labels it already has.
                RowView const           &view = train.row_view();
                std::vector<float> const gathered =
                    view.is_identity() ? std::vector<float>{}
                                       : gather_rows(view, train.labels());
                init_score_ = objective_.init_score(
                    view.is_identity() ? train.labels() : floats_view{gathered});
                scores_.assign(train.n_rows(), init_score_);
            }
            else
            {
                // Warm start: the booster was loaded with trees but no
                // training state. Rebuild every row's score by routing the
                // existing trees over the binned data.
                std::vector<float> raw(train.n_rows(), 0.0F);
                for (auto const &t : trees_.read())
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
            // Elementwise, no reduction: parallel order cannot change a bit.
            parallel::for_each_index(grad_.size(),
                                     [&](size_t i)
                                     {
                                         grad_[i] *= train.weights()[i];
                                         hess_[i] *= train.weights()[i];
                                     });
        }
        lap(prof.objective_s);

        size_t const n_selected = refill_row_indices(train);
        lap(prof.sample_s);

        auto [tree, leaf_values, leaf_ids] = grower_.grow(
            train, grad_, hess_, {row_indices_.data(), n_selected},
            selection_is_identity(train, n_selected), selection_runs(train));
        lap(prof.grow_s);

        // Leaf renewal (surrogate-hessian objectives): replace each leaf's
        // Newton step with the objective's optimal value over the residuals
        // of the rows it covers. scores_ still exclude this tree (and, under
        // DART, the dropped trees) — exactly the state gradients used.
        if constexpr (requires(std::span<float> r) { objective_.renew_leaf(r); })
        {
            renew_leaves(tree, leaf_ids, leaf_values, train.labels(), train.row_view());
        }
        lap(prof.renew_s);

        if (!dropped.empty())
        {
            apply_dart_round(tree, dropped, dropped_sum, leaf_values);
        }
        else
        {
            // Over the PLANE, not the view, and that is not an oversight. A
            // row id may repeat in a view (a with-replacement draw is the
            // point of allowing it), and scores_ holds one slot per row: the
            // repeat earns its extra weight by being read twice in the
            // histogram, not by having its prediction advanced twice. Walking
            // the view here would do exactly that and desynchronize the row's
            // score from the model. Rows the view omits are safe for a
            // different reason, stated on RecycledOutputs.
            parallel::for_each_index(
                scores_.size(), [&](size_t i)
                { scores_[i] += config_.learning_rate * leaf_values[i]; });
        }
        lap(prof.score_s);

        trees_.mutate().push_back(std::move(tree));
        // Hand the output buffers back for the next tree (skips the
        // per-tree zero-init; grower.hpp documents the write-before-read
        // contract).
        grower_.recycle(std::move(leaf_values), std::move(leaf_ids));
    }

    // Fills row_indices_ for this tree from the Dataset's row view, which is
    // every row of the plane unless the caller subset it. AllRowsSampler is
    // deterministic, so once row_indices_ holds the view's rows only its size
    // need be checked: the per-tree refill (a measurable membw cost at scale)
    // is skipped. The content is byte-identical to materializing every tree,
    // so the model is unchanged. Other samplers draw fresh indices each tree.
    size_t refill_row_indices(Dataset const &train)
    {
        RowView const &view = train.row_view();
        if constexpr (sampler_traits<sampler_type>::copies_view_verbatim)
        {
            if (row_indices_.size() != view.size())
            {
                view.materialize_into(row_indices_);
            }
            return row_indices_.size();
        }
        else
        {
            // The draw's universe is the view's rows, so the ids it emits are
            // the view's and at most that many land in row_indices_.
            if (candidates_.size() != view.size())
            {
                view.materialize_into(candidates_);
                row_indices_.resize(view.size());
            }
            return sampler_.sample(grad_, hess_, rng_, candidates_, row_indices_);
        }
    }

    // Whether this tree's row list is exactly [0, n_rows), which is what lets
    // the fills index bins and grad by position instead of gathering.
    // AllRowsSampler copies the view verbatim, so the descriptor answers in
    // constant time. A drawing sampler builds its own list and is asked the
    // old way, behind the view's constant-time veto: a view that is not the
    // whole plane can never yield the identity.
    bool selection_is_identity(Dataset const &train, size_t n_selected) const
    {
        RowView const &view = train.row_view();
        if constexpr (sampler_traits<sampler_type>::copies_view_verbatim)
        {
            return view.is_identity();
        }
        else
        {
            return view.is_identity() && n_selected == view.size() &&
                   rows_are_identity({row_indices_.data(), n_selected}, view.size());
        }
    }

    // This tree's rows as runs of consecutive plane rows, which is what lets
    // the column fill read a run's bins as a subspan instead of one indirect
    // load per element. AllRowsSampler copies the view verbatim, so the view's
    // own runs describe the list; a drawing sampler builds its own list and
    // the descriptor cannot speak for it, so those fits keep gathering.
    row_run_view selection_runs(Dataset const &train) const
    {
        if constexpr (sampler_traits<sampler_type>::copies_view_verbatim)
        {
            return train.row_view().runs();
        }
        else
        {
            return {};
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
                      !sampler_traits<sampler_type>::reads_gradients)
        {
            bool const host_forced = std::getenv("BONSAI_HOST_OBJECTIVE") != nullptr;
            // A view is eligible: the resident epilogue walks the view's rows
            // and leaves every other score untouched, which is the contract
            // the host path already keeps by routing only within the view.
            // Gradients stay full-length and globally indexed, so the rows
            // outside the view are derived and never read.
            bool const runtime_ok = config_.dart_drop_rate <= 0.0F && !host_forced;
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
        auto res = grower_.grow(
            train, floats_view{}, floats_view{}, {row_indices_.data(), n_selected},
            selection_is_identity(train, n_selected), selection_runs(train));
        lap(prof.grow_s);
        trees_.mutate().push_back(std::move(res.tree));
    }

    float eval(features_view X, floats_view labels) const override
    {
        std::vector<float> scores(X.extent(0));
        predict(X, scores);
        return objective_.eval(scores, labels);
    }

    void predict(features_view X, floats_out scores) const override
    {
        predict_at(X, scores, 0);
    }

    size_t n_iters() const override
    {
        return trees_.read().size();
    }

    size_t n_trees() const override
    {
        return trees_.read().size();
    }

    // Group rows by leaf, hand each leaf's residuals to the objective, and
    // overwrite both the tree's leaf values and the per-row training values.
    // DART pre-grow half: pick this round's dropped trees and remove their
    // contribution from scores_, so gradients see the model without them.
    void drop_dart_trees(Dataset const &train, std::vector<size_t> &dropped,
                         std::vector<float> &dropped_sum)
    {
        if (config_.dart_drop_rate <= 0.0F || trees_.read().empty())
        {
            return;
        }
        std::bernoulli_distribution drop(config_.dart_drop_rate);
        for (size_t i = 0; i < trees_.read().size(); ++i)
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
            internal::accumulate_train_contribution(trees_.read()[i], train,
                                                    dropped_sum);
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
        auto &trees = trees_.mutate();
        for (size_t const i : dropped)
        {
            trees[i].scale_leaves(old_scale);
        }
        parallel::for_each_index(scores_.size(),
                                 [&](size_t i)
                                 {
                                     scores_[i] += config_.learning_rate *
                                                   ((old_scale * dropped_sum[i]) +
                                                    (new_scale * leaf_values[i]));
                                 });
    }

    // Each leaf's value is replaced by the objective's optimum over the
    // residuals of the rows it covers. The rows are the fit's, which for a
    // view is the view's: rows outside it were never carried through this tree
    // and their leaf id names a leaf they do not belong to. The identity view
    // is every row, so a plain or sampled fit renews exactly as before.
    void renew_leaves(tree_type &tree, std::vector<node_id_t> const &leaf_ids,
                      train_leaf_values &leaf_values, floats_view labels,
                      RowView const &view)
    {
        RowIndex const                                    rows{view};
        std::unordered_map<node_id_t, std::vector<float>> residuals;
        for (size_t k = 0; k < rows.size(); ++k)
        {
            row_id_t const r = rows[k];
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
        for (size_t k = 0; k < rows.size(); ++k)
        {
            row_id_t const r = rows[k];
            leaf_values[r]   = renewed.at(leaf_ids[r]);
        }
    }

    std::vector<double> feature_importance(ImportanceType type) const override
    {
        std::vector<double> out;
        for (auto const &tree : trees_.read())
        {
            internal::accumulate_importance(tree, type, out);
        }
        return out;
    }

    void predict_at(features_view X, floats_out scores, size_t n_trees) const override
    {
        assert(X.extent(0) == scores.size());
        auto const  &trees = trees_.read();
        size_t const k = n_trees == 0 ? trees.size() : std::min(n_trees, trees.size());
        std::fill(scores.begin(), scores.end(), 0.0F);
        for (size_t t = 0; t < k; ++t)
        {
            trees[t].predict(X, scores);
        }
        for (float &score : scores)
        {
            score = init_score_ + (score * config_.learning_rate);
        }
    }

    void predict_staged(features_view X, floats_out out) const override
    {
        size_t const n     = X.extent(0);
        auto const  &trees = trees_.read();
        assert(out.size() == n * trees.size());
        std::vector<float> raw(n, 0.0F);
        for (size_t t = 0; t < trees.size(); ++t)
        {
            trees[t].predict(X, raw);
            parallel::for_each_index(
                n, [&](size_t i)
                { out[(t * n) + i] = init_score_ + (raw[i] * config_.learning_rate); });
        }
    }

    void predict_leaf(features_view X, std::span<node_id_t> out) const override
    {
        internal::predict_leaf_over(trees_.read(), X, out);
    }

    void predict_at_binned(Dataset const &bins, floats_out scores,
                           size_t n_trees) const override
    {
        assert(bins.view_n_rows() == scores.size());
        auto const  &trees = trees_.read();
        size_t const k = n_trees == 0 ? trees.size() : std::min(n_trees, trees.size());
        std::fill(scores.begin(), scores.end(), 0.0F);
        for (size_t t = 0; t < k; ++t)
        {
            internal::accumulate_view_contribution(trees[t], bins, scores);
        }
        for (float &score : scores)
        {
            score = init_score_ + (score * config_.learning_rate);
        }
    }

    // The device predict seam. Only a dense ensemble packs; an oblivious one
    // returns the declining default, whose host walk is predict_at_binned
    // above.
    PredictPlanInput predict_plan_input() const override
    {
        if constexpr (std::same_as<tree_type, DenseTree>)
        {
            return {.trees         = trees_.read(),
                    .learning_rate = config_.learning_rate,
                    .init_score    = init_score_,
                    .epoch         = trees_.epoch()};
        }
        else
        {
            return {};
        }
    }

    void predict_staged_binned(Dataset const &bins, floats_out out) const override
    {
        size_t const n     = bins.view_n_rows();
        auto const  &trees = trees_.read();
        assert(out.size() == n * trees.size());
        std::vector<float> raw(n, 0.0F);
        for (size_t t = 0; t < trees.size(); ++t)
        {
            internal::accumulate_view_contribution(trees[t], bins, raw);
            parallel::for_each_index(
                n, [&](size_t i)
                { out[(t * n) + i] = init_score_ + (raw[i] * config_.learning_rate); });
        }
    }

    void predict_leaf_binned(Dataset const       &bins,
                             std::span<node_id_t> out) const override
    {
        internal::predict_leaf_over_binned(trees_.read(), bins, out);
    }

    void pred_contribs_binned(Dataset const &bins, std::span<double> out,
                              size_t n_features) const override
    {
        if constexpr (std::same_as<tree_type, ObliviousTree>)
        {
            auto const dense = dense_.get(trees_.read(), trees_.epoch());
            contribs_over_binned(*dense, bins, out, n_features);
        }
        else
        {
            contribs_over_binned(trees_.read(), bins, out, n_features);
        }
    }

    // The shape both contribs paths share: per row, zero the slice, walk every
    // tree into it, scale by the learning rate, and add init_score to the bias
    // column. `into(t, row, phi)` is the only difference between them.
    template <typename Trees, typename Into>
    void contribs_impl(Trees const &trees, size_t n, std::span<double> out,
                       size_t n_features, Into const &into) const
    {
        size_t const cols = n_features + 1;
        assert(out.size() == n * cols);
        parallel::for_each_index(n,
                                 [&](size_t i)
                                 {
                                     std::span<double> const phi =
                                         out.subspan(i * cols, cols);
                                     std::ranges::fill(phi, 0.0);
                                     for (size_t t = 0; t < trees.size(); ++t)
                                     {
                                         into(t, static_cast<row_id_t>(i), phi);
                                     }
                                     for (double &v : phi)
                                     {
                                         v *= config_.learning_rate;
                                     }
                                     phi[n_features] += init_score_;
                                 });
    }

    // contribs_over's twin: routing reads the row's bins, the arithmetic that
    // follows is the same walk over the same covers.
    template <typename Trees>
    void contribs_over_binned(Trees const &trees, Dataset const &bins,
                              std::span<double> out, size_t n_features) const
    {
        auto const     biases = internal::shap_biases(trees);
        auto const     sb     = internal::tree_split_bins(trees, bins);
        RowIndex const rows{bins.row_view()};
        contribs_impl(trees, rows.size(), out, n_features,
                      [&](size_t t, row_id_t position, std::span<double> phi)
                      {
                          tree_shap_binned(trees[t], sb[t], bins, rows[position], phi,
                                           biases[t]);
                      });
    }

    std::string dump(std::span<std::string const> feature_names) const override
    {
        std::string out;
        auto const &trees = trees_.read();
        for (size_t t = 0; t < trees.size(); ++t)
        {
            out += "tree " + std::to_string(t) + ":\n";
            internal::dump_tree(trees[t], feature_names, out);
        }
        return out;
    }

    void pred_contribs(features_view X, std::span<double> out,
                       size_t n_features) const override
    {
        if constexpr (std::same_as<tree_type, ObliviousTree>)
        {
            auto const dense = dense_.get(trees_.read(), trees_.epoch());
            contribs_over(*dense, X, out, n_features);
        }
        else
        {
            contribs_over(trees_.read(), X, out, n_features);
        }
    }

    template <typename Trees>
    void contribs_over(Trees const &trees, features_view X, std::span<double> out,
                       size_t n_features) const
    {
        auto const biases = internal::shap_biases(trees);
        contribs_impl(trees, X.extent(0), out, n_features,
                      [&](size_t t, row_id_t row, std::span<double> phi)
                      { tree_shap(trees[t], X, row, phi, biases[t]); });
    }

    void seed_validation_scores(features_view X, std::span<float> out,
                                size_t n_rounds) const override
    {
        if (n_rounds > 0)
        {
            predict_at(X, floats_out{out.data(), out.size()}, n_rounds);
            return;
        }
        std::ranges::fill(out, init_score_);
    }

    float validation_loss(std::span<float const> scores,
                          floats_view            labels) const override
    {
        return objective_.eval(floats_view{scores.data(), scores.size()}, labels);
    }

    void accumulate_last_round(features_view X, floats_out scores) const override
    {
        assert(!trees_.read().empty());
        assert(X.extent(0) == scores.size());
        auto const &tree = trees_.read().back();
        float const lr   = config_.learning_rate;
        // One pass: predict's buffer starts at zero, so lr * value_for is the
        // same product the two-pass form formed.
        parallel::for_each_index(
            scores.size(), [&](size_t i)
            { scores[i] += lr * tree.value_for(X, static_cast<row_id_t>(i)); });
    }

    bool begin_resident_validation(Dataset const         &bins,
                                   std::span<float const> seed) override
    {
        return grower_.eval_begin(bins, device_objective_kind<objective_type>, seed);
    }

    bool accumulate_last_round_resident(Dataset const &bins, floats_out scores,
                                        std::optional<float> &loss) override
    {
        assert(!trees_.read().empty());
        return grower_.eval_accumulate(trees_.read().back(), bins,
                                       config_.learning_rate,
                                       {scores.data(), scores.size()}, loss);
    }

    void accumulate_last_round_binned(Dataset const &bins,
                                      floats_out     scores) const override
    {
        assert(!trees_.read().empty());
        assert(bins.view_n_rows() == scores.size());
        assert(!bins.mirror().bins().empty());
        auto const    &tree = trees_.read().back();
        float const    lr   = config_.learning_rate;
        auto const     sb   = internal::split_bins(tree, bins);
        auto const    &m    = bins.mirror();
        auto const     rm   = m.bins();
        RowIndex const rows{bins.row_view()};
        parallel::for_each_index(
            scores.size(),
            [&](size_t k)
            {
                row_id_t const r = rows[k];
                scores[k] += lr * internal::value_binned(tree, sb, [&](size_t f)
                                                         { return rm[m.index(r, f)]; });
            });
    }

    void truncate(size_t n_trees) override
    {
        if (n_trees < trees_.read().size())
        {
            // erase, not resize: growth would require default-constructible
            // trees, and truncate only ever shrinks.
            auto &trees = trees_.mutate();
            trees.erase(trees.begin() + static_cast<std::ptrdiff_t>(n_trees),
                        trees.end());
        }
    }

    // Save/load accessors. Public so io::save_booster / io::load_booster
    // can serialize state without befriending the I/O module.
    std::vector<tree_type> const &trees() const
    {
        return trees_.read();
    }
    float init_score() const
    {
        return init_score_;
    }
    void load_state(std::vector<tree_type> trees, float init_score)
    {
        trees_.mutate() = std::move(trees);
        init_score_     = init_score;
    }

  private:
    BoosterConfig  config_;
    objective_type objective_;
    grower_type    grower_;
    sampler_type   sampler_;
    std::mt19937   rng_;
    // The ensemble and its mutation epoch in one place: reads go through
    // read(), every write through mutate(), which is what bumps the epoch the
    // dense SHAP cache and the device plans rebuild on.
    internal::Versioned<std::vector<tree_type>> trees_;
    // Stale while the resident objective is armed (the device copy is
    // authoritative); resident_end syncs it before any host-path read.
    std::vector<float>    scores_;
    std::vector<float>    grad_;
    std::vector<float>    hess_;
    std::vector<row_id_t> row_indices_;
    // The rows a drawing sampler may pick from; empty when none draws.
    std::vector<row_id_t> candidates_;
    float                 init_score_      = 0.0F;
    bool                  resident_active_ = false;
    // Identity cookie for the Dataset the resident state was armed on:
    // compared by address only, never dereferenced through.
    Dataset const         *resident_train_ = nullptr;
    internal::DensifyCache dense_;
};

} // namespace bonsai
