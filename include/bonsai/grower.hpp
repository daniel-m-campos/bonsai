#pragma once

#include "bonsai/config/tree_config.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/split.hpp"
#include "bonsai/tree.hpp"
#include "bonsai/types.hpp"
#include <concepts>
#include <cstdint>
#include <functional>
#include <random>
#include <span>
#include <vector>

namespace bonsai
{

// A tree level's worth of nodes handed to a HistogramEngine in one call.
using split_input_refs = std::span<std::reference_wrapper<SplitInput> const>;

using train_leaf_values = std::vector<float>;

template <typename TreeT> struct GrowResult
{
    TreeT             tree;
    train_leaf_values values;
    // Per train row: the leaf that produced values[r] — DenseTree node id,
    // or ObliviousTree leaf-table index. Lets the booster regroup rows by
    // leaf (leaf renewal for constant-hessian objectives).
    std::vector<node_id_t> leaf_ids;
};

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
// for is wider and enforced by the parity suite (design review 2026-07-12):
// populate must accumulate exactly the node's rows' (grad, hess) into the
// bins the Dataset's mappers define, cell sums summed in an order that is
// a pure function of configuration (decision 49's determinism contract),
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
// decisions and counts cross the bus (docs/architecture/12-grower-backend.md).
// The LevelStep drives this whole cluster or none of it, so it is one concept
// and not seven; begin_root's bool return is the per-tree mode (it declines
// when the resident buffers cannot fit), captured once by the LevelStep.
template <typename T>
concept GPULevelEngine =
    HistogramEngine<T> &&
    requires(T b, Dataset const &ds, TreeConfig const &config, floats_view grad,
             floats_view hess, SplitInput &root, std::span<feature_id_t const> selected,
             std::span<typename T::LeafStamp const>   stamps,
             std::span<typename T::PartitionOp const> pops,
             std::span<typename T::LevelOp const> lops, std::span<uint32_t> counts,
             std::span<SplitInput const> level, std::span<SplitOutput> out,
             std::span<HistCell> child_sums, std::span<node_id_t> by_row,
             std::span<float const> node_values, std::span<float> row_values) {
        typename T::LevelOp;
        typename T::PartitionOp;
        typename T::LeafStamp;
        { b.begin_root(ds, grad, hess, root, selected) } -> std::convertible_to<bool>;
        b.stamp_leaves(stamps);
        b.partition_level(ds, pops, counts);
        b.advance_level(ds, lops);
        b.advance_layout_only();
        b.finalize_rows(by_row);
        b.finalize_tree(node_values, row_values, by_row);
        b.find_splits_many(ds, config, level, out, child_sums);
        b.find_level_split(ds, config, level, out, child_sums);
    };

struct CpuHistogramEngine
{
    void begin_tree(Dataset const & /*ds*/, floats_view /*grad*/, floats_view /*hess*/)
    {
    }
    void populate(Dataset const &ds, floats_view grad, floats_view hess,
                  SplitInput &split_input, std::span<feature_id_t const> selected);
    // Level-batched fill: all of a level's nodes in one call, so row-wise
    // work units from many small nodes share one parallel section
    // (docs/architecture/7-parallel.md). populate() is the one-node case.
    void populate_many(Dataset const &ds, floats_view grad, floats_view hess,
                       split_input_refs nodes, std::span<feature_id_t const> selected);
};

static_assert(HistogramEngine<CpuHistogramEngine>);

template <HistogramEngine EngineT   = CpuHistogramEngine,
          NodeSplitFinder SplitterT = HistogramNodeSplitFinder>
class DepthwiseGrower
{
  public:
    using Engine = EngineT;
    using Tree   = DenseTree;
    explicit DepthwiseGrower(TreeConfig const &cfg);
    GrowResult<Tree> grow(Dataset const &ds, floats_view grad, floats_view hess,
                          row_index_view row_indices);

    // Output-buffer recycling (PR #38's setup line): the booster hands the
    // previous tree's buffers back; every element is overwritten before any
    // read (host stamping + route_unsampled cover the row partition, the
    // device epilogue writes all rows), so reuse skips the per-tree
    // zero-init — 12.8GB of serial memset per 16M x 100 fit.
    void recycle(train_leaf_values values, std::vector<node_id_t> leaf_ids)
    {
        recycled_values_ = std::move(values);
        recycled_ids_    = std::move(leaf_ids);
    }

  private:
    TreeConfig                             config_;
    std::mt19937                           feature_rng_;
    std::vector<std::vector<feature_id_t>> interaction_groups_;
    EngineT                                engine_;
    train_leaf_values                      recycled_values_;
    std::vector<node_id_t>                 recycled_ids_;
};

template <HistogramEngine  EngineT   = CpuHistogramEngine,
          LevelSplitFinder SplitterT = HistogramLevelSplitFinder>
class ObliviousGrower
{
  public:
    using Engine = EngineT;
    using Tree   = ObliviousTree;
    explicit ObliviousGrower(TreeConfig const &cfg);
    GrowResult<Tree> grow(Dataset const &ds, floats_view grad, floats_view hess,
                          row_index_view row_indices);

    // Output-buffer recycling (PR #38's setup line): the booster hands the
    // previous tree's buffers back; every element is overwritten before any
    // read (host stamping + route_unsampled cover the row partition, the
    // device epilogue writes all rows), so reuse skips the per-tree
    // zero-init — 12.8GB of serial memset per 16M x 100 fit.
    void recycle(train_leaf_values values, std::vector<node_id_t> leaf_ids)
    {
        recycled_values_ = std::move(values);
        recycled_ids_    = std::move(leaf_ids);
    }

  private:
    TreeConfig             config_;
    std::mt19937           feature_rng_;
    EngineT                engine_;
    train_leaf_values      recycled_values_;
    std::vector<node_id_t> recycled_ids_;
};

template <HistogramEngine EngineT   = CpuHistogramEngine,
          NodeSplitFinder SplitterT = HistogramNodeSplitFinder>
class LeafwiseGrower
{
  public:
    using Engine = EngineT;
    using Tree   = DenseTree;
    explicit LeafwiseGrower(TreeConfig const &cfg);
    GrowResult<Tree> grow(Dataset const &ds, floats_view grad, floats_view hess,
                          row_index_view row_indices);

    // Output-buffer recycling (PR #38's setup line): the booster hands the
    // previous tree's buffers back; every element is overwritten before any
    // read (host stamping + route_unsampled cover the row partition, the
    // device epilogue writes all rows), so reuse skips the per-tree
    // zero-init — 12.8GB of serial memset per 16M x 100 fit.
    void recycle(train_leaf_values values, std::vector<node_id_t> leaf_ids)
    {
        recycled_values_ = std::move(values);
        recycled_ids_    = std::move(leaf_ids);
    }

  private:
    TreeConfig                             config_;
    std::mt19937                           feature_rng_;
    std::vector<std::vector<feature_id_t>> interaction_groups_;
    EngineT                                engine_;
    train_leaf_values                      recycled_values_;
    std::vector<node_id_t>                 recycled_ids_;
};

} // namespace bonsai
