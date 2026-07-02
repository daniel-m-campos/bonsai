#pragma once

namespace bonsai
{

struct SamplerConfig
{
    float subsample  = 1.0F;
    float top_rate   = 0.2F; // goss: fraction kept by |grad|
    float other_rate = 0.1F; // goss: fraction of total sampled from the rest

    bool operator==(SamplerConfig const &) const = default;
};

} // namespace bonsai
