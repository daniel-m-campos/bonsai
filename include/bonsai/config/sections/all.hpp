#pragma once

// One declarative row per TOML key. Adding a new key is one new
// `field<&SubConfig::name>()` line in the matching section header below.
// Both the TOML-loading path and the CLI-override path iterate this tuple.
//
// When C++26 reflection (P2996) lands, the per-section descriptors collapse
// from one row per field to a single fold over
// `std::meta::nonstatic_data_members_of(^SubConfig)`, and the
// `field<MemPtr>()` helper retires in favor of `meta::identifier_of(member)`.
// The Field/Section/FieldCodec/dispatch layers below this seam stay
// unchanged. See docs/architecture or `internal/field_name.hpp` for the
// stand-in mechanism.

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
    data_section,    bin_mapper_section, tree_section,     sampler_section,
    booster_section, dispatch_section,   metrics_section,  parallel_section,
    objective_section,
};

} // namespace bonsai::config::internal
