#pragma once

#include <cstdint>

namespace bonsai
{

struct ParallelConfig
{
    // 0 = auto (invariants: auto-thread-cap)
    uint32_t n_threads = 0;
    // CUDA device for cuda_* growers. 0 = the default device; ignored by
    // CPU growers. Placement only: model bits are unaffected, and
    // the knob is deliberately NOT persisted in the model artifact.
    uint32_t device_id = 0;

    bool operator==(ParallelConfig const &) const = default;
};

} // namespace bonsai
