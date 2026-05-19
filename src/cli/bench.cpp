#include "bonsai/cli/common.hpp"
#include "bonsai/cli/handlers.hpp"
#include "bonsai/cli/pipeline.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <print>

#include "bonsai/config/toml.hpp"
#include "bonsai/io/model.hpp"

namespace bonsai::cli
{

namespace
{

double seconds_since(std::chrono::steady_clock::time_point t)
{
    auto const now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - t).count();
}

} // namespace

int run_bench(BenchOpts const &opts)
{
    auto cfg = resolve_config(opts.common);
    if (opts.common.dump_config)
    {
        std::println("{}", config::dump_toml(cfg));
        return EXIT_SUCCESS;
    }
    if (cfg.data.train.empty())
    {
        std::println(stderr, "bench: data.train is required");
        return 2;
    }

    using clk = std::chrono::steady_clock;

    auto const t0     = clk::now();
    auto loaded       = load_train_from_csv(cfg, cfg.data.train);
    auto const t_load = seconds_since(t0);

    auto const t1    = clk::now();
    auto booster     = train_in_memory(cfg, loaded.train);
    auto const t_fit = seconds_since(t1);

    auto const t2        = clk::now();
    auto const eval_path = cfg.data.test.empty() ? cfg.data.train : cfg.data.test;
    auto scored          = score_csv(*booster, eval_path, cfg.data);
    auto const t_predict = seconds_since(t2);

    auto const rows_per_sec =
        static_cast<double>(loaded.train.n_rows() * cfg.booster_config.n_iters) / t_fit;

    std::println("bench:");
    std::println("  load_seconds={}", t_load);
    std::println("  fit_seconds={}", t_fit);
    std::println("  predict_seconds={}", t_predict);
    std::println("  rows_per_sec={}", rows_per_sec);
    std::println("  n_train_rows={}", loaded.train.n_rows());
    std::println("  n_predict_rows={}", scored.raw_scores.size());
    std::println("  n_iters={}", cfg.booster_config.n_iters);

    if (!opts.model_path.empty())
    {
        io::save_booster(*booster, opts.model_path, loaded.mappers, cfg.dispatch,
                         cfg.booster_config.learning_rate);
    }
    return EXIT_SUCCESS;
}

} // namespace bonsai::cli
