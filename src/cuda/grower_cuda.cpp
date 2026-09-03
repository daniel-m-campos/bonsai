
#include "bonsai/cuda/histogram_engine.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/split.hpp"
#include "grower_impl.hpp"

namespace bonsai
{

template bool GrowerHost<CudaHistogramEngine>::eval_accumulate<DenseTree>(
    DenseTree const &, Dataset const &, float, std::span<float>,
    std::optional<float> &);
template bool GrowerHost<CudaHistogramEngine>::eval_accumulate<ObliviousTree>(
    ObliviousTree const &, Dataset const &, float, std::span<float>,
    std::optional<float> &);
template class DepthwiseGrower<CudaHistogramEngine, HistogramNodeSplitFinder>;
template class ObliviousGrower<CudaHistogramEngine, HistogramLevelSplitFinder>;
template class LeafwiseGrower<CudaHistogramEngine, HistogramNodeSplitFinder>;

} // namespace bonsai
