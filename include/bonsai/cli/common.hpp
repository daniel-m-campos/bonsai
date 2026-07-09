#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "bonsai/config/config.hpp"
#include "bonsai/config/toml.hpp"
#include "bonsai/detail/column_batch.hpp"
#include "bonsai/parallel.hpp"
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
// Also applies process-wide settings the resolved config dictates
// (worker thread count) so every subcommand honors [parallel].
inline Config resolve_config(CommonOpts const &opts)
{
    Config cfg;
    if (!opts.config_path.empty())
    {
        cfg = config::load_toml(opts.config_path);
    }
    config::apply_overrides(cfg, opts.overrides);
    parallel::set_n_threads(cfg.parallel.n_threads);
    return cfg;
}

// Row-major contiguous feature buffer matching features_view. Either owns
// its storage (CLI: built from a parsed CSV) or borrows a caller-owned
// row-major matrix that must outlive it (Python module: the numpy array is
// alive for the duration of the train call).
struct FeatureBuffer
{
    std::vector<float>     data;
    std::span<float const> borrowed;
    size_t                 n_rows{};
    size_t                 n_features{};

    features_view view() const
    {
        return features_view{borrowed.empty() ? data.data() : borrowed.data(), n_rows,
                             n_features};
    }
};

// Build a row-major feature matrix from a parsed CSV, using only the columns
// the bin mappers know about (preserves their order).
FeatureBuffer to_feature_buffer(detail::ColumnBatch const &batch);

} // namespace bonsai::cli
