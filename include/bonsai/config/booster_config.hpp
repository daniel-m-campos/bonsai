#pragma once

#include <cstdint>

namespace bonsai
{

struct BoosterConfig
{
    uint32_t n_iters     = 100;
    float learning_rate  = 0.05F;
    uint32_t random_seed = 42;
    // 0 = silent; else log floor(n_iters / log_intervals) + 1 ticks during fit
    // (iter 0 baseline + each period boundary + final iter).
    uint32_t log_intervals = 0;
};

} // namespace bonsai
