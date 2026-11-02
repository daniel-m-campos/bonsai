#pragma once

#include "bonsai/config/booster_config.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/config/internal/field.hpp"

namespace bonsai::config::internal
{

inline constexpr auto booster_section = make_section(
    "booster", &Config::booster_config, field<&BoosterConfig::n_iters>(),
    field<&BoosterConfig::learning_rate>(k_positive),
    field<&BoosterConfig::random_seed>(), field<&BoosterConfig::log_intervals>(),
    field<&BoosterConfig::early_stopping_rounds>(),
    field<&BoosterConfig::dart_drop_rate>(k_probability));

} // namespace bonsai::config::internal
