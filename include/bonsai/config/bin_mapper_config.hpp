#pragma once

#include <cstddef>
#include <cstdint>

namespace bonsai
{

struct BinMapperConfig
{
    int      max_bin         = 255;
    size_t   n_samples       = 200000;
    uint64_t seed            = 0;
    int      min_data_in_bin = 1;

    bool operator==(BinMapperConfig const &) const = default;
};

} // namespace bonsai
