#pragma once

#include <cstdint>

namespace bonsai
{

struct ParallelConfig
{
    uint32_t n_threads = 0; // 0 = all hardware threads

    bool operator==(ParallelConfig const &) const = default;
};

} // namespace bonsai
