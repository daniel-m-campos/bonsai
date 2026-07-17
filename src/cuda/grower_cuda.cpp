// Explicit instantiation of the CUDA-backed depthwise grower
// (the CudaDepthwiseGrower alias). Lives in its own TU so grower.cpp never
// depends on the CUDA backend; only compiled when BONSAI_CUDA is enabled.
// The class-template form is spelled out here because explicit instantiation
// cannot name a type alias.

#include "bonsai/cuda/histogram_engine.hpp"
#include "bonsai/cuda/multi_engine.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/split.hpp"
#include "grower_impl.hpp"

namespace bonsai
{

template class DepthwiseGrower<CudaHistogramEngine, HistogramNodeSplitFinder>;
template class ObliviousGrower<CudaHistogramEngine, HistogramLevelSplitFinder>;

// The data-parallel engine over N device contexts (docs/architecture/19). The
// grower names/dispatch are a later stage; these instantiations let the parity
// tests drive the multi engine through the same grow loop as the single one.
template class DepthwiseGrower<MultiCudaHistogramEngine, HistogramNodeSplitFinder>;
template class ObliviousGrower<MultiCudaHistogramEngine, HistogramLevelSplitFinder>;

} // namespace bonsai
