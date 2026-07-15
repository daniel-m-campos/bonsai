// Phase 1 parity-guarantee net. Pins the exact MSE eval output we want
// preserved through the Phase 2 metric refactor: California Housing,
// 20 iterations, default seed -> rmse=0.7157657.
//
// Same recipe as the CLI smoke documented in the plan:
//   bonsai fit -c configs/california_housing.toml --set booster.n_iters=20
//   bonsai eval --data tests/data/california_housing_test.csv
//
// If this test fails after a Phase 2 change, you've drifted the eval path.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <string>

#include "bonsai/cli/pipeline.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/objective.hpp"

using namespace bonsai;      // NOLINT
using namespace bonsai::cli; // NOLINT

namespace
{

std::string const k_train =
    std::string{BONSAI_TESTS_DATA_DIR} + "/california_housing_train.csv";
std::string const k_test =
    std::string{BONSAI_TESTS_DATA_DIR} + "/california_housing_test.csv";

Config make_california_housing_config()
{
    // Mirrors configs/california_housing.toml, with n_iters=20 to match the
    // smoke test in the plan.
    Config cfg;
    cfg.data.train                    = k_train;
    cfg.data.test                     = k_test;
    cfg.data.label_column             = 0;
    cfg.bin_mapper.max_bin            = 255;
    cfg.bin_mapper.n_samples          = 200000;
    cfg.bin_mapper.seed               = 0;
    cfg.bin_mapper.min_data_in_bin    = 1;
    cfg.tree_config.max_depth         = 6;
    cfg.tree_config.min_data_in_leaf  = 20;
    cfg.tree_config.min_child_hess    = 0.001F;
    cfg.tree_config.min_gain_to_split = 0.0F;
    cfg.tree_config.lambda_l2         = 1.0F;
    cfg.booster_config.n_iters        = 20;
    cfg.booster_config.learning_rate  = 0.05F;
    cfg.booster_config.random_seed    = 42;
    // dispatch defaults to mse / depthwise / all_rows.
    return cfg;
}

} // namespace

TEST_CASE("Eval baseline: California Housing, MSE, 20 iters -> rmse=0.7157605",
          "[eval_baseline][mse]")
{
    auto const cfg     = make_california_housing_config();
    auto const loaded  = load_train_from_csv(cfg, cfg.data.train);
    auto const booster = train_in_memory(cfg, loaded.train);

    auto const sl = score_and_label_csv(*booster, cfg.data.test, cfg.data);
    REQUIRE(sl.raw_scores.size() == 4128);

    float const mse  = MSEObjective::eval(sl.raw_scores, sl.labels);
    float const rmse = std::sqrt(mse);

    // Bit-exact pin. If this needs to change, the eval path's float-rounding
    // semantics drifted -- investigate before relaxing. Pin lineage:
    // decision 50 (float histogram cells, 0.7175214 -> 0.7175160), decision
    // 51 (ceiling stride, -> 0.7157605), issue #61 (exact cuts for
    // duplicate-heavy columns, -> 0.7162500: +0.07% here, traded for -9%
    // on house_sales where the collapse was severe; campaign table in the
    // fix PR), issue #63 (count-weighted cuts on heavy-value columns,
    // -> 0.71719 on arm64 / 0.71725 on x86-64: the new thresholds put one
    // split on an fma-contraction knife edge, exposing that clang fused
    // a*b+c only on targets with the instruction), decision 59
    // (-ffp-contract=off on the host plane, -> 0.71725 on EVERY
    // architecture — this pin is now platform-independent by construction),
    // decision 74 (FLT_MAX top-band closer: the sentinel bin is NaN-only on
    // every fitting path, -> 0.7153: -0.27% here, inside the chance band;
    // the leak synthetics in scripts/probe_missing_bin.py are the real
    // motivation).
    CHECK(rmse == Catch::Approx(0.7153F).margin(1e-6));
}
