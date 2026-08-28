#include "bonsai/grower.hpp"
#include "bonsai/metal/histogram_engine.hpp"
#include "bonsai/split.hpp"
#include "grower_impl.hpp"

namespace bonsai
{

template class DepthwiseGrower<MetalHistogramEngine, HistogramNodeSplitFinder>;

} // namespace bonsai
