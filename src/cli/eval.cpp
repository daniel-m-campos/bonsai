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

#include "bonsai/metric.hpp"
#include "bonsai/registry/objective_dispatch.hpp"

namespace bonsai::cli
{

int run_eval(EvalOpts const &opts)
{
    auto const inputs =
        scoring_inputs(opts.common, opts.model_path, opts.data_path, "eval");
    if (!inputs)
    {
        return inputs.error();
    }
    auto const &objective = inputs->loaded.cfg.dispatch.objective_name;
    auto        sl        = score_and_label_csv(*inputs->loaded.booster, inputs->path,
                                                inputs->cfg.data, inputs->loaded.mappers.size());
    auto        preds     = sl.raw_scores;
    apply_link_inverse_by_name(objective, preds);

    auto const task  = task_kind_by_name(objective);
    auto const names = choose_metric_names(inputs->cfg.metrics.eval, objective);

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
