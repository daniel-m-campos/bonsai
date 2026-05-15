#pragma once

#include "bonsai/config/tree_config.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/split.hpp"
#include "bonsai/tree.hpp"
#include "bonsai/types.hpp"
#include <concepts>
#include <vector>

namespace bonsai
{

using train_leaf_values = std::vector<float>;

template <typename TreeT> struct GrowResult
{
    TreeT tree;
    train_leaf_values values;
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

template <SplitFinder SplitterT = HistogramSplitFinder> class DepthwiseGrower
{
  public:
    using Tree = DenseTree;
    explicit DepthwiseGrower(TreeConfig const &cfg);
    GrowResult<Tree> grow(Dataset const &ds, floats_view grad, floats_view hess,
                          row_index_view row_indices);

  private:
    TreeConfig config_;
};

} // namespace bonsai
