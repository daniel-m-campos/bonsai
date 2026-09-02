#pragma once

// One declarative row per TOML key. Adding a new key is one new
// `field<&SubConfig::name>()` line in the matching section header below.

#include <tuple>

#include "bonsai/config/sections/bin_mapper.hpp"
#include "bonsai/config/sections/booster.hpp"
#include "bonsai/config/sections/data.hpp"
#include "bonsai/config/sections/dispatch.hpp"
#include "bonsai/config/sections/metrics.hpp"
#include "bonsai/config/sections/objective.hpp"
#include "bonsai/config/sections/parallel.hpp"
#include "bonsai/config/sections/sampler.hpp"
#include "bonsai/config/sections/tree.hpp"

namespace bonsai::config::internal
{

inline constexpr auto all_sections = std::tuple{
    data_section,    bin_mapper_section, tree_section,
    sampler_section, booster_section,    dispatch_section,
    metrics_section, parallel_section,   objective_section,
};

} // namespace bonsai::config::internal
