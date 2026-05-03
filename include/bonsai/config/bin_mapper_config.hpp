#pragma once

#include <cstdint>

namespace bonsai
{

struct BinMapperConfig
{
    int max_bin              = 255;
    int bin_construct_sample = 200000;
    uint64_t seed            = 0;
    int min_data_in_bin      = 1;
};

} // namespace bonsai
