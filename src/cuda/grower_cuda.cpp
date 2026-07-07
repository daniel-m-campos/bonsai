// Explicit instantiation of the CUDA-backed depthwise grower. Lives in its
// own TU so grower.cpp never depends on the CUDA backend; only compiled
// when BONSAI_CUDA is enabled.

#include "bonsai/cuda/histogram_builder.hpp"
#include "../grower_impl.hpp"

namespace bonsai
{

template class DepthwiseGrower<HistogramNodeSplitFinder, CudaHistogramBuilder>;

} // namespace bonsai
