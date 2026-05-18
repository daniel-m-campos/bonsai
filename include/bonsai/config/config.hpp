#pragma once

#include "bonsai/config/bin_mapper_config.hpp"
#include "bonsai/config/booster_config.hpp"
#include "bonsai/config/data_config.hpp"
#include "bonsai/config/dispatch_config.hpp"
#include "bonsai/config/metrics_config.hpp"
#include "bonsai/config/tree_config.hpp"

namespace bonsai
{

struct Config
{
    DataConfig data;
    BinMapperConfig bin_mapper;
    TreeConfig tree_config;
    BoosterConfig booster_config;
    DispatchConfig dispatch;
    MetricsConfig metrics;
    // ParallelConfig, IOConfig — added as components are designed.
};

} // namespace bonsai
