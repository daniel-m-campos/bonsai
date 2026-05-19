#include "bonsai/cli/common.hpp"
#include "bonsai/cli/handlers.hpp"
#include "bonsai/cli/pipeline.hpp"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <print>

#include "bonsai/config/data_config.hpp"
#include "bonsai/io/model.hpp"
#include "bonsai/registry/objective_dispatch.hpp"

namespace bonsai::cli
{

int run_predict(PredictOpts const &opts)
{
    auto cfg    = resolve_config(opts.common);
    auto loaded = io::load_booster(opts.model_path);

    DataConfig data_cfg = cfg.data;
    if (!opts.data_path.empty())
    {
        data_cfg.train = opts.data_path;
    }
    auto const path = !opts.data_path.empty() ? opts.data_path : data_cfg.test;
    if (path.empty())
    {
        std::println(stderr, "predict: data path is required (--data or [data].test)");
        return 2;
    }

    auto scored = score_csv(*loaded.booster, path, data_cfg);

    if (opts.apply_link)
    {
        apply_link_inverse_by_name(loaded.dispatch.objective_name, scored.raw_scores);
    }

    if (opts.out_path.empty())
    {
        write_predictions(stdout, scored.raw_scores);
        return EXIT_SUCCESS;
    }
    std::unique_ptr<std::FILE, decltype(&std::fclose)> out{
        std::fopen(opts.out_path.c_str(), "w"), &std::fclose};
    if (!out)
    {
        std::println(stderr, "predict: cannot write '{}'", opts.out_path);
        return EXIT_FAILURE;
    }
    write_predictions(out.get(), scored.raw_scores);
    return EXIT_SUCCESS;
}

} // namespace bonsai::cli
