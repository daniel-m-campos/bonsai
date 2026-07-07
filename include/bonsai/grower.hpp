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
// call so stateful backends can stage per-tree data (the CUDA builder
// uploads gradients there); populate fills the split input's hists for the
// selected features and leaves zero-binned placeholders the finders skip.
template <typename T>
concept HistogramEngine =
    requires(T b, Dataset const &ds, floats_view grad, floats_view hess,
             SplitInput &split_input, std::span<feature_id_t const> selected) {
        b.begin_tree(ds, grad, hess);
        b.populate(ds, grad, hess, split_input, selected);
    };

// Optional phase-2 hook: fill an entire tree level in one call (the CUDA
// builder collapses it to a single launch). populate_nodes uses it when
// present and otherwise loops populate per node.
template <typename T>
concept BatchHistogramEngine =
    HistogramEngine<T> &&
    requires(T b, Dataset const &ds, floats_view grad, floats_view hess,
             split_input_refs nodes, std::span<feature_id_t const> selected) {
        b.populate_many(ds, grad, hess, nodes, selected);
    };

// Optional phase-3 hooks: histograms and rows stay device-resident, so only
// decisions and counts cross the bus (docs/architecture/11-gpu-resident.md).
// The depthwise grower drives this whole cluster or none of it, so it is one
// concept and not seven; resident() is the runtime query of whether the device
// path is actually live for the current tree.
template <typename T>
concept GPULevelEngine =
    HistogramEngine<T> &&
    requires(T b, Dataset const &ds, TreeConfig const &config, floats_view grad,
             floats_view hess, SplitInput &root, std::span<feature_id_t const> selected,
             std::span<typename T::LeafStamp const>   stamps,
             std::span<typename T::PartitionOp const> pops,
             std::span<typename T::LevelOp const> lops, std::span<uint32_t> counts,
             std::span<SplitInput const> level, std::span<SplitOutput> out,
             std::span<HistCell> child_sums, std::span<node_id_t> by_row) {
        typename T::LevelOp;
        typename T::PartitionOp;
        typename T::LeafStamp;
        { b.begin_root(ds, grad, hess, root, selected) } -> std::convertible_to<bool>;
        { b.resident() } -> std::convertible_to<bool>;
        b.stamp_leaves(stamps);
        b.partition_level(ds, pops, counts);
        b.advance_level(ds, lops);
        b.finalize_rows(by_row);
        b.find_splits_many(ds, config, level, out, child_sums);
    };

struct CpuHistogramEngine
{
    void begin_tree(Dataset const & /*ds*/, floats_view /*grad*/, floats_view /*hess*/)
    {
    }
    void populate(Dataset const &ds, floats_view grad, floats_view hess,
                  SplitInput &split_input, std::span<feature_id_t const> selected);
};

static_assert(HistogramEngine<CpuHistogramEngine>);

template <HistogramEngine EngineT   = CpuHistogramEngine,
          NodeSplitFinder SplitterT = HistogramNodeSplitFinder>
class DepthwiseGrower
{
  public:
    using Tree = DenseTree;
    explicit DepthwiseGrower(TreeConfig const &cfg);
    GrowResult<Tree> grow(Dataset const &ds, floats_view grad, floats_view hess,
                          row_index_view row_indices);

  private:
    TreeConfig                             config_;
    std::mt19937                           feature_rng_;
    std::vector<std::vector<feature_id_t>> interaction_groups_;
    EngineT                                builder_;
};

template <HistogramEngine  EngineT   = CpuHistogramEngine,
          LevelSplitFinder SplitterT = HistogramLevelSplitFinder>
class ObliviousGrower
{
  public:
    using Tree = ObliviousTree;
    explicit ObliviousGrower(TreeConfig const &cfg);
    GrowResult<Tree> grow(Dataset const &ds, floats_view grad, floats_view hess,
                          row_index_view row_indices);

  private:
    TreeConfig   config_;
    std::mt19937 feature_rng_;
    EngineT      builder_;
};

template <HistogramEngine EngineT   = CpuHistogramEngine,
          NodeSplitFinder SplitterT = HistogramNodeSplitFinder>
class LeafwiseGrower
{
  public:
    using Tree = DenseTree;
    explicit LeafwiseGrower(TreeConfig const &cfg);
    GrowResult<Tree> grow(Dataset const &ds, floats_view grad, floats_view hess,
                          row_index_view row_indices);

  private:
    TreeConfig                             config_;
    std::mt19937                           feature_rng_;
    std::vector<std::vector<feature_id_t>> interaction_groups_;
    EngineT                                builder_;
};

} // namespace bonsai
