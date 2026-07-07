#pragma once

#include "bonsai/cuda/histogram_engine.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/split.hpp"

namespace bonsai
{

// The registered "cuda_depthwise" grower: the depthwise grow loop with the
// GPU histogram engine. A named alias so the registry, name trait, and CLI
// can refer to the type without spelling the full instantiation; the
// explicit instantiation lives in src/cuda/grower_cuda.cpp.
using CudaDepthwiseGrower = DepthwiseGrower<CudaHistogramEngine>;

// The "cuda_oblivious" grower: the oblivious (symmetric-tree) grow loop with
// the GPU engine. Uses the device level-find (one split per level across all
// frontier nodes); partition/advance are reused from the depthwise path.
using CudaObliviousGrower =
    ObliviousGrower<CudaHistogramEngine, HistogramLevelSplitFinder>;

} // namespace bonsai
