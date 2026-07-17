// Explicit instantiation of the CUDA-backed depthwise grower
// (the CudaDepthwiseGrower alias). Lives in its own TU so grower.cpp never
// depends on the CUDA backend; only compiled when BONSAI_CUDA is enabled.
// The class-template form is spelled out here because explicit instantiation
// cannot name a type alias.

#include "bonsai/cuda/histogram_engine.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/split.hpp"
#include "grower_impl.hpp"

namespace bonsai
{

template class DepthwiseGrower<CudaHistogramEngine, HistogramNodeSplitFinder>;
template class ObliviousGrower<CudaHistogramEngine, HistogramLevelSplitFinder>;

} // namespace bonsai
