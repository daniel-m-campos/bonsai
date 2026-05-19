#include "bonsai/cli/common.hpp"
#include "bonsai/cli/handlers.hpp"
#include "bonsai/cli/pipeline.hpp"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <print>

#include "bonsai/config/toml.hpp"
#include "bonsai/io/model.hpp"

namespace bonsai::cli
{

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
    auto loaded = load_train_from_csv(cfg, cfg.data.train);
    std::println("fit: {} rows x {} features", loaded.train.n_rows(),
                 loaded.train.n_features());

    auto const n_iters = cfg.booster_config.n_iters;
    std::println("fit: training {} iterations ({} / {} / {})", n_iters,
                 cfg.dispatch.objective_name, cfg.dispatch.grower_name,
                 cfg.dispatch.sampler_name);

    std::size_t constexpr k_log_every = 10;
    auto booster                      = train_in_memory(cfg, loaded.train,
                                                        [&](std::size_t iter, std::size_t total)
                                                        {
                                       if (iter % k_log_every == 0 || iter == total)
                                       {
                                           std::println("  iter {}/{}", iter, total);
                                       }
                                   });

    if (!opts.model_path.empty())
    {
        std::println("fit: saving model to {}", opts.model_path);
        io::save_booster(*booster, opts.model_path, loaded.mappers, cfg.dispatch,
                         cfg.booster_config.learning_rate);
    }
    return EXIT_SUCCESS;
}

} // namespace bonsai::cli
