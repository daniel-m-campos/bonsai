#pragma once

#include <string>

namespace bonsai
{

struct DispatchConfig
{
    std::string objective_name = "mse";
    std::string grower_name    = "depthwise";
    std::string sampler_name   = "all_rows";

    bool operator==(DispatchConfig const &) const = default;
};

} // namespace bonsai
