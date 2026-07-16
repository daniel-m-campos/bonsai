#pragma once

#include "bonsai/config/config.hpp"
#include "bonsai/config/internal/field.hpp"
#include "bonsai/config/parallel_config.hpp"

namespace bonsai::config::internal
{

inline constexpr auto parallel_section =
    make_section("parallel", &Config::parallel, field<&ParallelConfig::n_threads>(),
                 field<&ParallelConfig::device_id>());

} // namespace bonsai::config::internal
