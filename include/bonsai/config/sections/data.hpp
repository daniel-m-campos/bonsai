#pragma once

#include "bonsai/config/config.hpp"
#include "bonsai/config/data_config.hpp"
#include "bonsai/config/internal/field.hpp"

namespace bonsai::config::internal
{

inline constexpr auto data_section = make_section(
    "data", &Config::data, field<&DataConfig::train>(), field<&DataConfig::test>(),
    field<&DataConfig::format>(), field<&DataConfig::header>(),
    field<&DataConfig::label_column>(), field<&DataConfig::weight_column>(),
    field<&DataConfig::missing_nan>(), field<&DataConfig::missing_sentinel>(),
    field<&DataConfig::valid>(), field<&DataConfig::ignore_columns>());

} // namespace bonsai::config::internal
