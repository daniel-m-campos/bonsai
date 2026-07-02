#pragma once

#include "bonsai/config/config.hpp"
#include "bonsai/config/internal/field.hpp"
#include "bonsai/config/tree_config.hpp"

namespace bonsai::config::internal
{

inline constexpr auto tree_section = make_section(
    "tree", &Config::tree_config, field<&TreeConfig::min_child_hess>(),
    field<&TreeConfig::min_gain_to_split>(), field<&TreeConfig::lambda_l2>(),
    field<&TreeConfig::lambda_l1>(),
    field<&TreeConfig::max_depth>(), field<&TreeConfig::min_data_in_leaf>(),
    field<&TreeConfig::max_leaves>(), field<&TreeConfig::feature_fraction>(),
    field<&TreeConfig::feature_seed>(), field<&TreeConfig::monotone_constraints>());

} // namespace bonsai::config::internal
