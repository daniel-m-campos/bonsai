#pragma once

#include "bonsai/grower.hpp"
#include "bonsai/metal/histogram_engine.hpp"
#include "bonsai/split.hpp"

namespace bonsai
{

// The registered "metal_depthwise" grower: the host depthwise grow loop with
// the Metal histogram engine. Because the engine satisfies only the base
// HistogramEngine concept, this is the CPU grower's code path with the fill
// swapped, not a device grower; the explicit instantiation lives in
// src/metal/grower_metal.cpp.
using MetalDepthwiseGrower = DepthwiseGrower<MetalHistogramEngine>;

} // namespace bonsai
