#pragma once

#include <cstdint>
#include <string>
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
    uint32_t max_leaves        = 31; // leafwise; 0 or >= 2^max_depth cannot bind
    uint32_t feature_seed      = 2;  // rng seed for feature_fraction draws
    // Per-feature monotone direction: +1 increasing, -1 decreasing, 0 free.
    // Missing trailing entries are free. Levelwise projects the finished leaf
    // table instead of vetoing splits (invariants: levelwise-monotone-holds).
    std::vector<int> monotone_constraints = {};
    // Feature groups allowed to interact on a tree path; one group per
    // string, ids separated by ',' (TOML) or '+' (CLI). Features outside
    // every group can only split alone. Empty = unconstrained.
    std::vector<std::string> interaction_constraints = {};

    // A depth-D tree has at most 2^D leaves, so a budget of 0 or of 2^D and
    // above never stops an expansion (invariants:
    // leaf-budget-cannot-bind-is-depthwise).
    bool leaves_bounded() const
    {
        if (max_leaves == 0)
        {
            return false;
        }
        return max_depth >= 32 || max_leaves < (uint32_t{1} << max_depth);
    }

    bool operator==(TreeConfig const &) const = default;
};

} // namespace bonsai
