#pragma once

#include "bonsai/config/config.hpp"
#include "bonsai/config/internal/field.hpp"
#include "bonsai/config/metrics_config.hpp"

namespace bonsai::config::internal
{

inline constexpr auto metrics_section =
    make_section("metrics", &Config::metrics, field<&MetricsConfig::fit>(),
                 field<&MetricsConfig::eval>());

} // namespace bonsai::config::internal
