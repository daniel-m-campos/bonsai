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

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void print_metric_row(std::string_view label, floats_view raw, floats_view preds,
                      floats_view labels, std::vector<Metric> const &metrics)
{
    std::print("{}:", label);
    for (auto const &m : metrics)
    {
        std::print(" {}={}", m.name, m.compute(m.from_raw ? raw : preds, labels));
    }
}

} // namespace

int run_fit(FitOpts const &opts)
{
    auto cfg = resolve_config(opts.common);

    io::LoadedBooster init;
    if (!opts.init_model_path.empty())
    {
        init = io::load_booster(opts.init_model_path);
        cfg  = reconcile_warm_start(
            std::move(cfg), init.cfg,
            config::stated_keys(opts.common.config_path, opts.common.overrides));
    }
    if (dump_config(opts.common, cfg))
    {
        return EXIT_SUCCESS;
    }
    if (cfg.data.train.empty())
    {
        std::println(stderr, "fit: data.train is required");
        return 2;
    }
    if (init.booster)
    {
        std::println("fit: continuing from {}", opts.init_model_path);
    }
    std::println("fit: fitting bin mappers from {}", cfg.data.train);
    auto loaded =
        init.booster
            ? load_train_and_validation_with_mappers(cfg, std::move(init.mappers))
            : load_train_and_validation_from_csv(cfg);
    std::println("fit: {} rows x {} features", loaded.train.dataset.plane_n_rows(),
                 loaded.train.dataset.n_features());

    auto const &obj_name = cfg.dispatch.objective_name;
    auto const  task     = task_kind_by_name(obj_name);
    auto const  names    = choose_metric_names(cfg.metrics.fit, obj_name);

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
        std::vector<float> const raw_train(tick.train_preds.begin(),
                                           tick.train_preds.end());
        apply_link_inverse_by_name(obj_name, tick.train_preds);

        std::print("  [{}]", tick.iter);
        print_metric_row(" train", raw_train, tick.train_preds, tick.train_labels,
                         metrics);

        if (!tick.validation_preds.empty())
        {
            std::vector<float> const raw_validation(tick.validation_preds.begin(),
                                                    tick.validation_preds.end());
            apply_link_inverse_by_name(obj_name, tick.validation_preds);
            print_metric_row(" | valid", raw_validation, tick.validation_preds,
                             tick.validation_labels, metrics);
        }
        std::println("");
    };

    auto booster = train_with_progress(cfg, loaded, on_tick, std::move(init.booster));

    if (!opts.model_path.empty())
    {
        std::println("fit: saving model to {}", opts.model_path);
        io::save_booster(*booster, opts.model_path, loaded.mappers, cfg);
    }
    return EXIT_SUCCESS;
}

} // namespace bonsai::cli
