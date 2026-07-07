#pragma once

#include "bonsai/dataset.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/split.hpp"
#include "bonsai/types.hpp"
#include <memory>
#include <span>

namespace bonsai
{

// HistogramBuilder that offloads per-node histogram construction to the GPU
// (src/cuda/histogram_builder.cu, CUDA C++ compiled by clang). Split
// finding, row partitioning, and the sibling-subtraction trick stay on the
// CPU. Device state is created lazily: the binned matrix uploads once per
// dataset, gradients once per tree. Kernels accumulate float per row-chunk
// and merge chunks in double; atomics add in arbitrary order, so cells
// match the CPU builder to rounding, not bit-exactly. Nodes below a
// row-count cutoff build on the CPU instead — a kernel launch +
// synchronous copy-back round trip costs more than scanning a small node
// outright, and most nodes in a deep tree are small.
class CudaHistogramBuilder
{
  public:
    CudaHistogramBuilder();
    ~CudaHistogramBuilder();
    CudaHistogramBuilder(CudaHistogramBuilder &&) noexcept;
    CudaHistogramBuilder &operator=(CudaHistogramBuilder &&) noexcept;
    CudaHistogramBuilder(CudaHistogramBuilder const &)            = delete;
    CudaHistogramBuilder &operator=(CudaHistogramBuilder const &) = delete;

    void begin_tree(Dataset const &ds, floats_view grad, floats_view hess);
    void populate(Dataset const &ds, floats_view grad, floats_view hess,
                  SplitInput &node, std::span<feature_id_t const> selected);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

static_assert(HistogramBuilder<CudaHistogramBuilder>);

// The dispatchable grower this backend exists for (registered as
// "cuda_depthwise" when BONSAI_USE_CUDA is defined).
using CudaDepthwiseGrower =
    DepthwiseGrower<HistogramNodeSplitFinder, CudaHistogramBuilder>;

} // namespace bonsai
