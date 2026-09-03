#pragma once

#include <cstddef>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "bonsai/config/config.hpp"
#include "bonsai/config/toml.hpp"
#include "bonsai/detail/column_batch.hpp"
#include "bonsai/parallel.hpp"
#include "bonsai/registry/objective_dispatch.hpp"
#include "bonsai/types.hpp"

namespace bonsai::cli
{

struct CommonOpts
{
    std::string                   config_path;
    std::vector<config::Override> overrides;
    bool                          dump_config = false;
};

// Load config from TOML (if path is given) and apply CLI overrides.
inline Config resolve_config(CommonOpts const &opts)
{
    Config cfg = config::resolve(opts.config_path, opts.overrides);
    parallel::set_n_threads(cfg.parallel.n_threads);
    return cfg;
}

// --dump-config prints the resolved config and the subcommand stops there,
// before it touches data or a model. Fit calls this after warm-start
// reconcile so the dump shows the config the fit would have used.
inline bool dumped_config(CommonOpts const &opts, Config const &cfg)
{
    if (!opts.dump_config)
    {
        return false;
    }
    std::println("{}", config::dump_toml(cfg));
    return true;
}

// The metric-name list a subcommand reports: the user's override if it is
// non-empty, else the objective's declared defaults. The returned views point
// into override_names (or into the objective's static name table), so it must
// outlive them.
inline std::vector<std::string_view>
choose_metric_names(std::vector<std::string> const &override_names,
                    std::string const              &objective_name)
{
    std::vector<std::string_view> out;
    if (!override_names.empty())
    {
        out.reserve(override_names.size());
        for (auto const &n : override_names)
        {
            out.emplace_back(n);
        }
        return out;
    }
    auto const defaults = default_metric_names_by_name(objective_name);
    out.assign(defaults.begin(), defaults.end());
    return out;
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

// Build a row-major feature matrix from a parsed CSV: every column of the
// batch, in batch order.
FeatureBuffer to_feature_buffer(detail::ColumnBatch const &batch);

} // namespace bonsai::cli
