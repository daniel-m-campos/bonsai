#pragma once

#include "bonsai/config/bin_mapper_config.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/config/internal/field.hpp"

namespace bonsai::config::internal
{

inline constexpr auto bin_mapper_section =
    make_section("bin_mapper", &Config::bin_mapper, field<&BinMapperConfig::max_bin>(),
                 field<&BinMapperConfig::n_samples>(), field<&BinMapperConfig::seed>(),
                 field<&BinMapperConfig::min_data_in_bin>());

} // namespace bonsai::config::internal
