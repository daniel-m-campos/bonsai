#include "bonsai/cli/common.hpp"
#include "bonsai/cli/handlers.hpp"
#include "bonsai/cli/pipeline.hpp"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <print>

#include "bonsai/registry/objective_dispatch.hpp"

namespace bonsai::cli
{

int run_predict(PredictOpts const &opts)
{
    auto const inputs =
        scoring_inputs(opts.common, opts.model_path, opts.data_path, "predict");
    if (!inputs)
    {
        return inputs.error();
    }
    auto scored = score_csv(*inputs->loaded.booster, inputs->path, inputs->cfg.data,
                            inputs->loaded.mappers.size(), opts.num_iteration);

    if (opts.apply_link)
    {
        apply_link_inverse_by_name(inputs->loaded.cfg.dispatch.objective_name,
                                   scored.raw_scores);
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
