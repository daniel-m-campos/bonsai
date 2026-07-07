#pragma once

#include "bonsai/config/tree_config.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/split.hpp"
#include "bonsai/tree.hpp"
#include "bonsai/types.hpp"
#include <concepts>
#include <functional>
#include <random>
#include <span>
#include <vector>

namespace bonsai
{

// A tree level's worth of nodes handed to a HistogramBuilder in one call.
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
// uploads gradients there); populate fills node.hists for the selected
// features and leaves zero-binned placeholders the split finders skip.
template <typename T>
concept HistogramBuilder =
    requires(T b, Dataset const &ds, floats_view grad, floats_view hess,
             SplitInput &node, std::span<feature_id_t const> selected) {
        b.begin_tree(ds, grad, hess);
        b.populate(ds, grad, hess, node, selected);
    };

struct CpuHistogramBuilder
{
    void begin_tree(Dataset const & /*ds*/, floats_view /*grad*/, floats_view /*hess*/)
    {
    }
    void populate(Dataset const &ds, floats_view grad, floats_view hess,
                  SplitInput &node, std::span<feature_id_t const> selected);
};

static_assert(HistogramBuilder<CpuHistogramBuilder>);

template <NodeSplitFinder  SplitterT = HistogramNodeSplitFinder,
          HistogramBuilder BuilderT  = CpuHistogramBuilder>
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
    BuilderT                               builder_;
};

template <LevelSplitFinder SplitterT = HistogramLevelSplitFinder,
          HistogramBuilder BuilderT  = CpuHistogramBuilder>
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
    BuilderT     builder_;
};

template <NodeSplitFinder  SplitterT = HistogramNodeSplitFinder,
          HistogramBuilder BuilderT  = CpuHistogramBuilder>
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
    BuilderT                               builder_;
};

} // namespace bonsai
