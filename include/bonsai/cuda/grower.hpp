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

// The "cuda_levelwise" grower: the levelwise (symmetric-tree) grow loop with
// the GPU engine. Uses the device level-find (one split per level across all
// frontier nodes); partition/advance are reused from the depthwise path.
using CudaObliviousGrower =
    ObliviousGrower<CudaHistogramEngine, HistogramLevelSplitFinder>;

// The "cuda_leafwise" grower: best-first growth with the GPU engine's leaf
// plane. Histograms live in a per-tree
// slot pool, partition and split finding run on the device one node at a time,
// and the gain heap stays on the host.
using CudaLeafwiseGrower = LeafwiseGrower<CudaHistogramEngine>;

} // namespace bonsai
