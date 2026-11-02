#pragma once

#include <limits>

#include "bonsai/config/bin_mapper_config.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/config/internal/field.hpp"
#include "bonsai/types.hpp"

namespace bonsai::config::internal
{

inline constexpr double k_fewest_splittable_bins = 3.0;

inline constexpr Bounds k_bin_count{k_fewest_splittable_bins,
                                    std::numeric_limits<bin_id_t>::max(), false, false};

inline constexpr auto bin_mapper_section = make_section(
    "bin_mapper", &Config::bin_mapper, field<&BinMapperConfig::max_bin>(k_bin_count),
    field<&BinMapperConfig::n_samples>(), field<&BinMapperConfig::seed>(),
    field<&BinMapperConfig::min_data_in_bin>());

} // namespace bonsai::config::internal
