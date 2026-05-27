#pragma once

#include "bonsai/config/config.hpp"
#include "bonsai/config/dispatch_config.hpp"
#include "bonsai/config/internal/field.hpp"

namespace bonsai::config::internal
{

inline constexpr auto dispatch_section = make_section(
    "dispatch", &Config::dispatch, field<&DispatchConfig::objective_name>(),
    field<&DispatchConfig::grower_name>(), field<&DispatchConfig::sampler_name>());

} // namespace bonsai::config::internal
