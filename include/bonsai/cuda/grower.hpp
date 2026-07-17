#pragma once

#include "bonsai/cuda/histogram_engine.hpp"
#include "bonsai/cuda/multi_engine.hpp"
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

// The "cuda_multi_depthwise" grower: the depthwise grow loop over the
// data-parallel engine (docs/architecture/19-multi-gpu.md). Selected by
// parallel.device_ids; the single-GPU cuda_depthwise path is untouched.
using CudaMultiDepthwiseGrower = DepthwiseGrower<MultiCudaHistogramEngine>;

// The "cuda_multi_oblivious" grower: the oblivious grow loop over the same
// engine, using the device level-find.
using CudaMultiObliviousGrower =
    ObliviousGrower<MultiCudaHistogramEngine, HistogramLevelSplitFinder>;

} // namespace bonsai
