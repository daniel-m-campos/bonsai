#pragma once

#include <cstdint>

namespace bonsai
{

struct ParallelConfig
{
    uint32_t n_threads = 0; // 0 = auto (hardware threads, capped at 16)
    // CUDA device for cuda_* growers (issue #158). 0 = the default device;
    // ignored by CPU growers (the sampler.subsample precedent for
    // regime-scoped knobs). Placement only: model bits are unaffected, and
    // the knob is deliberately NOT persisted in the model artifact.
    uint32_t device_id = 0;

    bool operator==(ParallelConfig const &) const = default;
};

} // namespace bonsai
