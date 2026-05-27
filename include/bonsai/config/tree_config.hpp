#pragma once

#include <cstdint>
namespace bonsai
{

struct TreeConfig
{
    float   min_child_hess    = 1.0F;
    float   min_gain_to_split = 0.0F;
    float   lambda_l2         = 1.0F;
    uint8_t max_depth         = 6;
    uint8_t min_data_in_leaf  = 20;

    bool operator==(TreeConfig const &) const = default;
};

} // namespace bonsai
