#pragma once

namespace bonsai
{

struct SamplerConfig
{
    float subsample = 1.0F;

    bool operator==(SamplerConfig const &) const = default;
};

} // namespace bonsai
