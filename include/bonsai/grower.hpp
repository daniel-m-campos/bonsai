#pragma once

#include "bonsai/config/tree_config.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/objective_traits.hpp"
#include "bonsai/row_view.hpp"
#include "bonsai/split.hpp"
#include "bonsai/tree.hpp"
#include "bonsai/types.hpp"
#include <concepts>
#include <cstdint>
#include <functional>
#include <optional>
#include <random>
#include <span>
#include <vector>

namespace bonsai
{

// A tree level's worth of nodes handed to a HistogramEngine in one call.
using split_input_refs = std::span<std::reference_wrapper<SplitInput> const>;

using train_leaf_values = std::vector<float>;

// Output-buffer recycling: the booster hands the previous tree's per-row
// buffers back so the next grow reuses the allocation instead of the per-tree
// zero-init, 12.8GB of serial memset per 16M x 100 fit. Every element the
// fit's ROW LIST names is overwritten before any read: host stamping and
// route_unsampled cover that partition, and the device epilogue walks the
// same rows.
//
// A view names fewer rows than the plane has, and the booster's score update
// still walks the plane, because a repeated row id must not be advanced twice.
// So the slots a view omits are read. They are zero, and stay zero: the first
// grow of a fit resizes from empty and value-initializes all of them, and no
// writer ever names them afterwards, so `+= lr * 0` leaves the score at its
// init value. That is the contract, not an accident of recycling, and
// test_views_leave_out_of_view_scores_alone pins it. Anything that writes an
// out-of-view slot breaks it. One home for the two buffers so this is stated
// once for all three growers.
struct RecycledOutputs
{
    train_leaf_values      values;
    std::vector<node_id_t> leaf_ids;

    void set(train_leaf_values v, std::vector<node_id_t> ids)
    {
        values   = std::move(v);
        leaf_ids = std::move(ids);
    }
};

template <typename TreeT> struct GrowResult
{
    TreeT             tree;
    train_leaf_values values;
    // Per train row: the leaf that produced values[r] — DenseTree node id,
    // or ObliviousTree leaf-table index. Lets the booster regroup rows by
    // leaf (leaf renewal for surrogate-hessian objectives).
    std::vector<node_id_t> leaf_ids;
};

// The one growth contract all three strategies satisfy. grow() returns the
// finished tree WITH per-row leaf values and leaf ids, so the booster never
// re-walks rows it just routed; a grower that only returned the tree would
// silently reintroduce the per-tree re-predict this shape deleted. Engines
// plug in below (HistogramEngine, and the GPU concepts beside it); the
// grower owns every per-node decision, the engine every per-row loop.
template <typename T>
concept TreeGrower = requires(T g, Dataset const &ds, floats_view grad,
                              floats_view hess, row_index_view row_indices) {
    typename T::Tree;
    requires Tree<typename T::Tree>;
    {
        g.grow(ds, grad, hess, row_indices)
    } -> std::same_as<GrowResult<typename T::Tree>>;
};

// Builds a node's per-feature histograms. begin_tree runs once per grow()
// call so stateful backends can stage per-tree data (the CUDA engine
// uploads gradients there); populate fills the split input's hists for the
// selected features and leaves zero-binned placeholders the finders skip.
//
// The concept can only check the two signatures; the CONTRACT it stands
// for is wider and enforced by the parity suite:
// populate must accumulate exactly the node's rows' (grad, hess) into the
// bins the Dataset's mappers define, cell sums summed in an order that is
// a pure function of configuration,
// missing values in the last bin, and hists[f] sized n_bins(f) for every
// selected f. A type satisfying the syntax while bending any of these
// trains silently wrong models — see docs/guide/2 for what each clause is
// load-bearing for.
template <typename T>
concept HistogramEngine =
    requires(T b, Dataset const &ds, floats_view grad, floats_view hess,
             SplitInput &split_input, std::span<feature_id_t const> selected) {
        b.begin_tree(ds, grad, hess);
        b.populate(ds, grad, hess, split_input, selected);
    };

// The GPU data plane: histograms and rows stay device-resident, so only
// decisions and counts cross the bus.
// The LevelStep drives this whole cluster or none of it, so it is one concept
// and not seven. begin_root opens the tree on the device or throws
// (device-oversized-tree-refuses-not-falls-back), so the step has no per-tree
// mode to carry.
template <typename T>
concept GPULevelEngine =
    HistogramEngine<T> &&
    requires(T b, Dataset const &ds, TreeConfig const &config, floats_view grad,
             floats_view hess, SplitInput &root, std::span<feature_id_t const> selected,
             std::span<typename T::LeafStamp const>   stamps,
             std::span<typename T::PartitionOp const> pops,
             std::span<typename T::LevelOp const> lops, std::span<uint32_t> counts,
             std::span<SplitInput const> level, std::span<SplitOutput> out,
             std::span<HistCell> child_sums, std::span<float const> node_values,
             std::span<float> values, std::span<node_id_t> leaf_ids,
             std::span<float const>                    init_scores,
             std::span<typename T::ResidentNode const> res_nodes,
             std::span<float>                          scores_out) {
        typename T::LevelOp;
        typename T::PartitionOp;
        typename T::LeafStamp;
        typename T::ResidentNode;
        b.begin_root(ds, grad, hess, root, selected);
        b.stamp_leaves(stamps);
        b.partition_level(ds, pops, counts);
        b.advance_level(ds, lops);
        b.advance_layout_only();
        b.finalize_tree(node_values, values, leaf_ids);
        b.find_splits_many(ds, config, level, out, child_sums);
        b.find_level_split(ds, config, level, out, child_sums);
        {
            b.resident_begin(ds, DeviceObjectiveKind::mse, init_scores, 1.0F)
        } -> std::convertible_to<bool>;
        { b.resident_armed() } -> std::convertible_to<bool>;
        b.resident_finalize(res_nodes);
        b.resident_end(scores_out);
    };

// The GPU leaf plane: best-first growth expands one leaf at a time, so the
// histograms live in a per-tree slot pool instead of the level plane's
// ping-pong. A second concept beside
// GPULevelEngine, not a change to it: the depthwise and levelwise paths are
// untouched. Same rule as begin_root: leaf_begin_root opens the tree on the
// device or throws.
template <typename T>
concept GPULeafEngine =
    HistogramEngine<T> &&
    requires(T b, Dataset const &ds, TreeConfig const &config, floats_view grad,
             floats_view hess, SplitInput &root, std::span<feature_id_t const> selected,
             std::span<typename T::LeafStamp const> stamps,
             typename T::LeafPartOp const &part_op, uint32_t small_slot,
             uint32_t large_slot, std::span<SplitInput const> nodes,
             std::span<uint32_t const> slots, std::span<SplitOutput> out,
             std::span<HistCell> child_sums, std::span<float const> node_values,
             std::span<float> values, std::span<node_id_t> leaf_ids,
             std::span<float const>                    init_scores,
             std::span<typename T::ResidentNode const> res_nodes,
             std::span<float>                          scores_out) {
        typename T::LeafPartOp;
        typename T::LeafRound;
        typename T::LeafStamp;
        typename T::ResidentNode;
        b.leaf_begin_root(ds, config, grad, hess, root, selected);
        { b.leaf_split(ds, part_op) } -> std::convertible_to<typename T::LeafRound>;
        b.leaf_build(ds, small_slot, large_slot);
        b.leaf_find(ds, config, nodes, slots, out, child_sums);
        b.leaf_stamp(stamps);
        b.finalize_tree(node_values, values, leaf_ids);
        {
            b.resident_begin_leaf(ds, config, DeviceObjectiveKind::mse, init_scores,
                                  1.0F)
        } -> std::convertible_to<bool>;
        { b.resident_armed() } -> std::convertible_to<bool>;
        b.resident_finalize(res_nodes);
        b.resident_end(scores_out);
    };

struct CpuHistogramEngine
{
    // Drops the tree's cached fill plan: the next tree redraws its features.
    void begin_tree(Dataset const &ds, floats_view grad, floats_view hess);
    // Fills one node the level plane's way: a level of one.
    void populate(Dataset const &ds, floats_view grad, floats_view hess,
                  SplitInput &split_input, std::span<feature_id_t const> selected);
    // The leaf plane's fused fill of one split's smaller child. `sibling` is
    // the larger child, holding the parent's histograms: the worker that fills
    // a feature subtracts it from the sibling in the same region. Returns
    // whether it did, so a caller the fused path declines still subtracts for
    // itself.
    bool populate_lone(Dataset const &ds, floats_view grad, floats_view hess,
                       SplitInput &split_input, std::span<feature_id_t const> selected,
                       NodeHistograms &sibling);
    // Level-batched fill: all of a level's nodes in one call, so row-wise
    // work units from many small nodes share one parallel section
    // in one parallel section. populate() is the one-node case.
    void populate_many(Dataset const &ds, floats_view grad, floats_view hess,
                       split_input_refs nodes, std::span<feature_id_t const> selected);
};

static_assert(HistogramEngine<CpuHistogramEngine>);

// Device-resident objective seam, shared by every grower: forward to the
// engine when it offers one and report whether it armed. A compiled no-op
// returning false on engines without the seam (the CPU plane), keeping the
// booster generic.
template <typename EngineT>
bool engine_resident_begin(EngineT &engine, Dataset const &ds, DeviceObjectiveKind kind,
                           std::span<float const> scores, float learning_rate)
{
    if constexpr (requires { engine.resident_begin(ds, kind, scores, learning_rate); })
    {
        return engine.resident_begin(ds, kind, scores, learning_rate);
    }
    else
    {
        return false;
    }
}

// The leaf plane's arming call: same seam, one more argument. Best-first
// growth sizes its histogram pool from the tree config, so the once-per-fit
// capacity test the resident mode needs cannot be decided without it.
template <typename EngineT>
bool engine_resident_begin_leaf(EngineT &engine, Dataset const &ds,
                                TreeConfig const &config, DeviceObjectiveKind kind,
                                std::span<float const> scores, float learning_rate)
{
    if constexpr (requires {
                      engine.resident_begin_leaf(ds, config, kind, scores,
                                                 learning_rate);
                  })
    {
        return engine.resident_begin_leaf(ds, config, kind, scores, learning_rate);
    }
    else
    {
        return false;
    }
}

template <typename EngineT>
void engine_resident_end(EngineT &engine, std::span<float> scores)
{
    if constexpr (requires { engine.resident_end(scores); })
    {
        engine.resident_end(scores);
    }
}

// The validation-plane shims, same collapse: a host engine has no eval plane,
// so eval_begin answers false and the booster keeps the host walk.
template <typename EngineT>
bool engine_eval_begin(EngineT &engine, Dataset const &valid, DeviceObjectiveKind kind,
                       std::span<float const> scores)
{
    if constexpr (requires { engine.eval_begin(valid, kind, scores); })
    {
        return engine.eval_begin(valid, kind, scores);
    }
    else
    {
        return false;
    }
}

// The grower's device state, one home for all three growers: the engine and
// the two arming flags the resident-objective and validation seams set. The
// flags answer "did the device accept this fit", so grow() can skip the
// host-side per-row output the resident finalize replaces, and the booster
// can route the eval walk. begin() and begin_leaf() differ only in that
// best-first growth sizes its histogram pool from the tree config, a
// capacity decided once per fit; a CPU engine compiles both to `false`
// through the engine_* shims above. eval_armed() is read by the growers'
// eval_accumulate definitions, which stay per-grower because each tree shape
// flattens to a device node table its own way.
template <typename EngineT> class DeviceSeam
{
  public:
    EngineT &engine()
    {
        return engine_;
    }

    bool begin(Dataset const &ds, DeviceObjectiveKind kind,
               std::span<float const> scores, float learning_rate)
    {
        resident_ = engine_resident_begin(engine_, ds, kind, scores, learning_rate);
        return resident_;
    }

    bool begin_leaf(Dataset const &ds, TreeConfig const &config,
                    DeviceObjectiveKind kind, std::span<float const> scores,
                    float learning_rate)
    {
        resident_ = engine_resident_begin_leaf(engine_, ds, config, kind, scores,
                                               learning_rate);
        return resident_;
    }

    void end(std::span<float> scores)
    {
        engine_resident_end(engine_, scores);
        resident_ = false;
    }

    bool armed() const
    {
        return resident_;
    }

    bool eval_begin(Dataset const &valid, DeviceObjectiveKind kind,
                    std::span<float const> scores)
    {
        eval_ = engine_eval_begin(engine_, valid, kind, scores);
        return eval_;
    }

    bool eval_armed() const
    {
        return eval_;
    }

  private:
    EngineT engine_;
    bool    resident_ = false;
    bool    eval_     = false;
};

// What the three growers share: the tree config, the feature sampler's rng,
// the device seam, and the recycled outputs, with the seam forwarded under
// the names Booster calls. A grower adds its grow loop and its eval flatten
// (see DeviceSeam); the leafwise grower arms through begin_leaf instead.
template <HistogramEngine EngineT> class GrowerHost
{
  public:
    using Engine = EngineT;

    void recycle(train_leaf_values values, std::vector<node_id_t> leaf_ids)
    {
        recycled_.set(std::move(values), std::move(leaf_ids));
    }

    bool resident_begin(Dataset const &ds, DeviceObjectiveKind kind,
                        std::span<float const> scores, float learning_rate)
    {
        return seam_.begin(ds, kind, scores, learning_rate);
    }
    void resident_end(std::span<float> scores)
    {
        seam_.end(scores);
    }
    bool resident() const
    {
        return seam_.armed();
    }
    bool eval_begin(Dataset const &valid, DeviceObjectiveKind kind,
                    std::span<float const> scores)
    {
        return seam_.eval_begin(valid, kind, scores);
    }

  protected:
    explicit GrowerHost(TreeConfig const &cfg)
        : config_(cfg), feature_rng_(cfg.feature_seed)
    {
    }
    TreeConfig const &config() const
    {
        return config_;
    }
    std::mt19937 &feature_rng()
    {
        return feature_rng_;
    }
    DeviceSeam<EngineT> &seam()
    {
        return seam_;
    }
    RecycledOutputs &recycled()
    {
        return recycled_;
    }

  private:
    TreeConfig          config_;
    std::mt19937        feature_rng_;
    DeviceSeam<EngineT> seam_;
    RecycledOutputs     recycled_;
};

template <HistogramEngine EngineT   = CpuHistogramEngine,
          NodeSplitFinder SplitterT = HistogramNodeSplitFinder>
class DepthwiseGrower : public GrowerHost<EngineT>
{
  public:
    using Host = GrowerHost<EngineT>;
    using Tree = DenseTree;
    explicit DepthwiseGrower(TreeConfig const &cfg);
    GrowResult<Tree> grow(Dataset const &ds, floats_view grad, floats_view hess,
                          RowSelection selection = {});
    bool             eval_accumulate(Tree const &tree, Dataset const &valid, float lr,
                                     std::span<float> scores_out, std::optional<float> &loss);

  private:
    using Host::config;
    using Host::feature_rng;
    using Host::recycled;
    using Host::seam;
    std::vector<std::vector<feature_id_t>> interaction_groups_;
};

template <HistogramEngine  EngineT   = CpuHistogramEngine,
          LevelSplitFinder SplitterT = HistogramLevelSplitFinder>
class ObliviousGrower : public GrowerHost<EngineT>
{
  public:
    using Host = GrowerHost<EngineT>;
    using Tree = ObliviousTree;
    explicit ObliviousGrower(TreeConfig const &cfg);
    GrowResult<Tree> grow(Dataset const &ds, floats_view grad, floats_view hess,
                          RowSelection selection = {});
    bool             eval_accumulate(Tree const &tree, Dataset const &valid, float lr,
                                     std::span<float> scores_out, std::optional<float> &loss);

  private:
    using Host::config;
    using Host::feature_rng;
    using Host::recycled;
    using Host::seam;
};

template <HistogramEngine         EngineT   = CpuHistogramEngine,
          ParallelNodeSplitFinder SplitterT = HistogramNodeSplitFinder>
class LeafwiseGrower : public GrowerHost<EngineT>
{
  public:
    using Host = GrowerHost<EngineT>;
    using Tree = DenseTree;
    explicit LeafwiseGrower(TreeConfig const &cfg);
    GrowResult<Tree> grow(Dataset const &ds, floats_view grad, floats_view hess,
                          RowSelection selection = {});
    bool             resident_begin(Dataset const &ds, DeviceObjectiveKind kind,
                                    std::span<float const> scores, float learning_rate)
    {
        return seam().begin_leaf(ds, config(), kind, scores, learning_rate);
    }
    bool eval_accumulate(Tree const &tree, Dataset const &valid, float lr,
                         std::span<float> scores_out, std::optional<float> &loss);

  private:
    using Host::config;
    using Host::feature_rng;
    using Host::recycled;
    using Host::seam;
    std::vector<std::vector<feature_id_t>> interaction_groups_;
};

} // namespace bonsai
