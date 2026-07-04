#pragma once

#include <cstdint>

namespace bonsai
{

struct ObjectiveConfig
{
    float    huber_delta    = 1.0F; // huber: residual half-width of the L2 zone
    float    quantile_alpha = 0.5F; // quantile: target quantile in (0, 1)
    uint32_t n_classes      = 3;    // softmax: number of classes (labels 0..K-1)

    bool operator==(ObjectiveConfig const &) const = default;
};

} // namespace bonsai
