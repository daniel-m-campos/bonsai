#pragma once

#include "bonsai/config/bin_mapper_config.hpp"
#include "bonsai/config/data_config.hpp"

namespace bonsai
{

struct Config
{
    DataConfig data;
    BinMapperConfig bin_mapper;
    // BoosterConfig, TreeConfig, SamplerConfig, SplitConfig,
    // ParallelConfig, IOConfig — added as components are designed.
};

} // namespace bonsai
