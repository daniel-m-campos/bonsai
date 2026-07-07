#pragma once

#include "bonsai/dataset.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/split.hpp"
#include "bonsai/types.hpp"
#include <memory>
#include <span>

namespace bonsai
{

// True when this build carries the CUDA backend AND a usable device is
// present. cuda_depthwise is registered in every build; only training
// needs this to be true. See docs/architecture/10-cuda.md.
bool cuda_available();

// HistogramBuilder that offloads histogram construction to the GPU
// (src/cuda/histogram_builder.cu; a throwing stub backs it when built
// without BONSAI_CUDA). GPU cells match the CPU builder to tolerance, not
// bit-exactly: atomics accumulate in arbitrary order. Design and precision
// scheme: docs/architecture/10-cuda.md.
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
    // Batched variant: one kernel launch covers a whole tree level.
    void populate_many(Dataset const &ds, floats_view grad, floats_view hess,
                       split_input_refs nodes, std::span<feature_id_t const> selected);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

static_assert(HistogramBuilder<CudaHistogramBuilder>);

// Registered as "cuda_depthwise" (registry/typelists.hpp).
using CudaDepthwiseGrower =
    DepthwiseGrower<HistogramNodeSplitFinder, CudaHistogramBuilder>;

} // namespace bonsai
