#pragma once

#include <cstdint>
#include <vector>
namespace bonsai
{

struct TreeConfig
{
    float    min_child_hess    = 1.0F;
    float    min_gain_to_split = 0.0F;
    float    lambda_l2         = 1.0F;
    float    lambda_l1         = 0.0F; // L1 on leaf weights; 0 = off
    float    feature_fraction  = 1.0F; // per-tree feature subsample; 1 = all
    uint8_t  max_depth         = 6;
    uint8_t  min_data_in_leaf  = 20;
    uint32_t max_leaves        = 31; // leafwise only; 0 = unbounded (depth-capped)
    uint32_t feature_seed      = 2;  // rng seed for feature_fraction draws
    // Per-feature monotone direction: +1 increasing, -1 decreasing, 0 free.
    // Missing trailing entries are free. Node-splitting growers only.
    std::vector<int> monotone_constraints = {};

    bool operator==(TreeConfig const &) const = default;
};

} // namespace bonsai
