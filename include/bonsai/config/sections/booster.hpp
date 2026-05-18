#pragma once

#include "bonsai/config/booster_config.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/config/internal/field.hpp"

namespace bonsai::config::internal
{

inline auto constexpr booster_section = make_section(
    "booster", &Config::booster_config, field<&BoosterConfig::n_iters>(),
    field<&BoosterConfig::learning_rate>(), field<&BoosterConfig::random_seed>(),
    field<&BoosterConfig::log_intervals>());

} // namespace bonsai::config::internal
