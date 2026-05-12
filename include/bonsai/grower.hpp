#pragma once

#include "bonsai/config/tree_config.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/split.hpp"
#include "bonsai/tree.hpp"
#include "bonsai/types.hpp"
#include <concepts>

namespace bonsai
{

template <typename T>
concept TreeGrower = requires(T g, Dataset const &ds, floats_view grad,
                              floats_view hess, row_index_view row_indices) {
    typename T::Tree;
    requires Tree<typename T::Tree>;
    { g.grow(ds, grad, hess, row_indices) } -> std::same_as<typename T::Tree>;
};

template <SplitFinder SplitterT = HistogramSplitFinder> class DepthwiseGrower
{
  public:
    using Tree = DenseTree;
    explicit DepthwiseGrower(TreeConfig const &cfg);
    Tree grow(Dataset const &ds, floats_view grad, floats_view hess,
              row_index_view row_indices);

  private:
    TreeConfig config_;
};

} // namespace bonsai
