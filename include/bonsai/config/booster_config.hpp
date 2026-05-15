#pragma once

#include <cstdint>

namespace bonsai
{

struct BoosterConfig
{
    uint32_t n_iters     = 100;
    float learning_rate  = 0.05F;
    uint32_t random_seed = 42;
};

} // namespace bonsai
