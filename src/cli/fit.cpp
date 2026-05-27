#include "bonsai/cli/common.hpp"
#include "bonsai/cli/handlers.hpp"
#include "bonsai/cli/pipeline.hpp"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <print>
#include <string>
#include <string_view>
#include <vector>

#include "bonsai/config/toml.hpp"
#include "bonsai/io/model.hpp"
#include "bonsai/metric.hpp"
#include "bonsai/registry/objective_dispatch.hpp"
#include "bonsai/types.hpp"

namespace bonsai::cli
{

namespace
{

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

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void print_metric_row(std::string_view label, floats_view preds, floats_view labels,
                      std::vector<Metric> const &metrics)
{
    std::print("{}:", label);
    for (auto const &m : metrics)
    {
        std::print(" {}={}", m.name, m.compute(preds, labels));
    }
}

} // namespace

int run_fit(FitOpts const &opts)
{
    auto cfg = resolve_config(opts.common);
    if (opts.common.dump_config)
    {
        std::println("{}", config::dump_toml(cfg));
        return EXIT_SUCCESS;
    }
    if (cfg.data.train.empty())
    {
        std::println(stderr, "fit: data.train is required");
        return 2;
    }

    std::println("fit: fitting bin mappers from {}", cfg.data.train);
    auto loaded = load_train_and_valid_from_csv(cfg);
    std::println("fit: {} rows x {} features", loaded.train.dataset.n_rows(),
                 loaded.train.dataset.n_features());

    auto const &obj_name = cfg.dispatch.objective_name;
    auto const  task     = task_kind_by_name(obj_name);
    auto const  names    = choose_metric_names(cfg.metrics.fit, obj_name);

    // Resolve names to Metric values up front so a bad name fails fast.
    std::vector<Metric> metrics;
    metrics.reserve(names.size());
    for (auto const name : names)
    {
        metrics.push_back(resolve_metric_for_task(name, task));
    }

    std::println("fit: training {} iterations ({} / {} / {})",
                 cfg.booster_config.n_iters, obj_name, cfg.dispatch.grower_name,
                 cfg.dispatch.sampler_name);

    auto const on_tick = [&](FitTick const &tick)
    {
        // Apply link inverse in place to train/valid scratch buffers, then
        // compute metrics. Buffers are owned by train_with_progress and
        // overwritten next tick.
        apply_link_inverse_by_name(obj_name, tick.train_preds);

        std::print("  [{}]", tick.iter);
        print_metric_row(" train", tick.train_preds, tick.train_labels, metrics);

        if (!tick.valid_preds.empty())
        {
            apply_link_inverse_by_name(obj_name, tick.valid_preds);
            print_metric_row(" | valid", tick.valid_preds, tick.valid_labels, metrics);
        }
        std::println("");
    };

    auto booster = train_with_progress(cfg, loaded, on_tick);

    if (!opts.model_path.empty())
    {
        std::println("fit: saving model to {}", opts.model_path);
        io::save_booster(*booster, opts.model_path, loaded.mappers, cfg);
    }
    return EXIT_SUCCESS;
}

} // namespace bonsai::cli
