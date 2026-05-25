#pragma once

#include "bonsai/config/config.hpp"
#include "bonsai/config/internal/field.hpp"
#include "bonsai/config/sampler_config.hpp"

namespace bonsai::config::internal
{

inline auto constexpr sampler_section =
    make_section("sampler", &Config::sampler, field<&SamplerConfig::subsample>());

} // namespace bonsai::config::internal
