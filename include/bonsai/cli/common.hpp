#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "bonsai/config/config.hpp"
#include "bonsai/config/toml.hpp"
#include "bonsai/detail/column_batch.hpp"
#include "bonsai/types.hpp"

namespace bonsai::cli
{

// Shared CLI options used by most subcommands.
struct CommonOpts
{
    std::string                   config_path;
    std::vector<config::Override> overrides;
    bool                          dump_config = false;
};

// Load config from TOML (if path is given) and apply CLI overrides.
inline Config resolve_config(CommonOpts const &opts)
{
    Config cfg;
    if (!opts.config_path.empty())
    {
        cfg = config::load_toml(opts.config_path);
    }
    config::apply_overrides(cfg, opts.overrides);
    return cfg;
}

// Row-major contiguous feature buffer matching features_view.
struct FeatureBuffer
{
    std::vector<float> data;
    size_t             n_rows{};
    size_t             n_features{};

    features_view view() const
    {
        return features_view{data.data(), n_rows, n_features};
    }
};

// Build a row-major feature matrix from a parsed CSV, using only the columns
// the bin mappers know about (preserves their order).
FeatureBuffer to_feature_buffer(detail::ColumnBatch const &batch);

} // namespace bonsai::cli
