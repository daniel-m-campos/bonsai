#pragma once

#include <cstdint>

namespace bonsai
{

struct ParallelConfig
{
    uint32_t n_threads = 0; // 0 = auto (hardware threads, capped at 16)

    bool operator==(ParallelConfig const &) const = default;
};

} // namespace bonsai
