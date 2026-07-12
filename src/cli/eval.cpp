#include "bonsai/cli/common.hpp"
#include "bonsai/cli/handlers.hpp"
#include "bonsai/cli/pipeline.hpp"

#include <cstdio>
#include <cstdlib>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "bonsai/config/data_config.hpp"
#include "bonsai/config/toml.hpp"
#include "bonsai/io/model.hpp"
#include "bonsai/metric.hpp"
#include "bonsai/registry/objective_dispatch.hpp"

namespace bonsai::cli
{

namespace
{

// Resolve the metric-name list to use for eval: user override if non-empty,
// else the objective's declared defaults.
std::vector<std::string_view>
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

} // namespace

int run_eval(EvalOpts const &opts)
{
    auto cfg = resolve_config(opts.common);
    if (opts.common.dump_config)
    {
        std::println("{}", config::dump_toml(cfg));
        return EXIT_SUCCESS;
    }
    auto loaded = io::load_booster(opts.model_path);

    DataConfig data_cfg = cfg.data;
    auto const path     = !opts.data_path.empty() ? opts.data_path : data_cfg.test;
    if (path.empty())
    {
        std::println(stderr, "eval: data path is required (--data or [data].test)");
        return 2;
    }

    auto sl    = score_and_label_csv(*loaded.booster, path, data_cfg);
    auto preds = sl.raw_scores;
    apply_link_inverse_by_name(loaded.cfg.dispatch.objective_name, preds);

    auto const task = task_kind_by_name(loaded.cfg.dispatch.objective_name);
    auto const names =
        choose_metric_names(cfg.metrics.eval, loaded.cfg.dispatch.objective_name);

    for (auto const name : names)
    {
        auto const  m = resolve_metric_for_task(name, task);
        float const v = m.compute(m.from_raw ? sl.raw_scores : preds, sl.labels);
        std::print("{}={} ", m.name, v);
    }
    std::println("n={}", sl.raw_scores.size());
    return EXIT_SUCCESS;
}

} // namespace bonsai::cli
