
#include "bonsai/cuda/histogram_engine.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/split.hpp"
#include "grower_impl.hpp"

namespace bonsai
{

template class DepthwiseGrower<CudaHistogramEngine, HistogramNodeSplitFinder>;
template class ObliviousGrower<CudaHistogramEngine, HistogramLevelSplitFinder>;
template class LeafwiseGrower<CudaHistogramEngine, HistogramNodeSplitFinder>;

} // namespace bonsai
