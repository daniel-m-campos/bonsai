#pragma once

#include <cstdint>

namespace bonsai
{

struct BoosterConfig
{
    uint32_t n_iters       = 100;
    float    learning_rate = 0.05F;
    uint32_t random_seed   = 42;
    // 0 = silent; else log floor(n_iters / log_intervals) + 1 ticks during fit
    // (iter 0 baseline + each period boundary + final iter).
    uint32_t log_intervals = 0;
    // 0 = off; else stop when the first valid set's objective hasn't improved
    // for this many iterations, and keep only the best iteration's trees.
    uint32_t early_stopping_rounds = 0;

    bool operator==(BoosterConfig const &) const = default;
};

} // namespace bonsai
