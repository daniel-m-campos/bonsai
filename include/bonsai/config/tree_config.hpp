#pragma once

#include <cstdint>
namespace bonsai
{

struct TreeConfig
{
    float    min_child_hess    = 1.0F;
    float    min_gain_to_split = 0.0F;
    float    lambda_l2         = 1.0F;
    float    feature_fraction  = 1.0F; // per-tree feature subsample; 1 = all
    uint8_t  max_depth         = 6;
    uint8_t  min_data_in_leaf  = 20;
    uint32_t max_leaves        = 31; // leafwise only; 0 = unbounded (depth-capped)
    uint32_t feature_seed      = 2;  // rng seed for feature_fraction draws

    bool operator==(TreeConfig const &) const = default;
};

} // namespace bonsai
