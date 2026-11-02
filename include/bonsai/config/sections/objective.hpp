#pragma once

#include "bonsai/config/config.hpp"
#include "bonsai/config/internal/field.hpp"
#include "bonsai/config/objective_config.hpp"

namespace bonsai::config::internal
{

inline constexpr auto objective_section = make_section(
    "objective", &Config::objective, field<&ObjectiveConfig::huber_delta>(k_positive),
    field<&ObjectiveConfig::quantile_alpha>(k_open_unit),
    field<&ObjectiveConfig::n_classes>(), field<&ObjectiveConfig::n_targets>());

} // namespace bonsai::config::internal
