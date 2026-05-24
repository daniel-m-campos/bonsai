#pragma once

#include "bonsai/grower.hpp"
#include "bonsai/objective.hpp"
#include "bonsai/sampler.hpp"
#include "bonsai/split.hpp"
#include "bonsai/typelist.hpp"

namespace bonsai
{

using Objectives = TypeList<MSEObjective, LogLossObjective>;
using Growers    = TypeList<DepthwiseGrower<HistogramNodeSplitFinder>>;
using Samplers   = TypeList<AllRowsSampler>;

} // namespace bonsai
