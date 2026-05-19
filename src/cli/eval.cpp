#include "bonsai/cli/common.hpp"
#include "bonsai/cli/handlers.hpp"
#include "bonsai/cli/pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <print>
#include <string>
#include <vector>

#include "bonsai/config/data_config.hpp"
#include "bonsai/config/toml.hpp"
#include "bonsai/io/model.hpp"
#include "bonsai/objective.hpp"
#include "bonsai/registry/objective_dispatch.hpp"

namespace bonsai::cli
{

namespace
{

// Phase 1 transitional: emits the same metrics bit-for-bit as before the
// metrics-table deletion. Phase 2 replaces this with the new TaskKind+Metric
// registry. Routes rmse through MSEObjective::eval to preserve exact float
// rounding behavior.
void print_default_metrics(std::string const &objective_name,
                           std::vector<float> &raw_scores,
                           std::vector<float> const &labels)
{
    if (objective_name == "logloss")
    {
        apply_link_inverse_by_name(objective_name, raw_scores);
        auto const &p       = raw_scores;
        auto const &y       = labels;
        auto const n        = p.size();
        double constexpr eps = 1e-7;
        double ll            = 0.0;
        std::size_t correct  = 0;
        for (std::size_t i = 0; i < n; ++i)
        {
            double const pi = p[i];
            double const ti = y[i] > 0.5F ? 1.0 : 0.0;
            ll -= (ti * std::log(std::max(pi, eps))) +
                  ((1.0 - ti) * std::log(std::max(1.0 - pi, eps)));
            if ((pi >= 0.5) == (ti >= 0.5))
            {
                ++correct;
            }
        }
        auto const nd = static_cast<double>(n);
        std::print("logloss={} accuracy={} ",
                   static_cast<float>(ll / nd),
                   static_cast<float>(static_cast<double>(correct) / nd));
    }
    else
    {
        // MSE link is identity; raw_scores == predictions. Use the
        // objective's eval directly so float rounding matches pre-refactor.
        float const mse = MSEObjective::eval(raw_scores, labels);
        std::print("rmse={} ", std::sqrt(mse));
    }
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

    auto sl = score_and_label_csv(*loaded.booster, path, data_cfg);
    print_default_metrics(loaded.dispatch.objective_name, sl.raw_scores, sl.labels);
    std::println("n={}", sl.raw_scores.size());
    return EXIT_SUCCESS;
}

} // namespace bonsai::cli
