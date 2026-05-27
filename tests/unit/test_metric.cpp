#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <vector>

#include "bonsai/metric.hpp"
#include "bonsai/task.hpp"

using namespace bonsai; // NOLINT

TEST_CASE("compute_rmse: perfect predictions -> 0", "[metric][rmse]")
{
    std::vector<float> const p{1.0F, 2.0F, 3.0F, 4.0F};
    CHECK(compute_rmse(p, p) == Catch::Approx(0.0F));
}

TEST_CASE("compute_rmse: constant error of 2 -> 2", "[metric][rmse]")
{
    std::vector<float> const p{2.0F, 3.0F, 4.0F};
    std::vector<float> const y{0.0F, 1.0F, 2.0F};
    CHECK(compute_rmse(p, y) == Catch::Approx(2.0F));
}

TEST_CASE("compute_mae: constant abs error of 1.5 -> 1.5", "[metric][mae]")
{
    std::vector<float> const p{1.5F, 2.5F, -1.5F};
    std::vector<float> const y{0.0F, 1.0F, 0.0F};
    CHECK(compute_mae(p, y) == Catch::Approx(1.5F));
}

TEST_CASE("compute_r2: perfect predictions -> 1", "[metric][r2]")
{
    std::vector<float> const y{1.0F, 2.0F, 3.0F, 4.0F};
    CHECK(compute_r2(y, y) == Catch::Approx(1.0F));
}

TEST_CASE("compute_r2: predict the mean -> 0", "[metric][r2]")
{
    std::vector<float> const y{1.0F, 2.0F, 3.0F, 4.0F};
    std::vector<float> const p{2.5F, 2.5F, 2.5F, 2.5F}; // mean(y) = 2.5
    CHECK(compute_r2(p, y) == Catch::Approx(0.0F));
}

TEST_CASE("compute_r2: constant labels yields NaN", "[metric][r2][degenerate]")
{
    std::vector<float> const y{2.0F, 2.0F, 2.0F};
    std::vector<float> const p{1.0F, 2.0F, 3.0F};
    CHECK(std::isnan(compute_r2(p, y)));
}

TEST_CASE("compute_logloss: confident correct -> ~0", "[metric][logloss]")
{
    // probs ~ 1 for label 1, probs ~ 0 for label 0
    std::vector<float> const p{0.999F, 0.001F};
    std::vector<float> const y{1.0F, 0.0F};
    CHECK(compute_logloss(p, y) == Catch::Approx(0.0F).margin(1e-2));
}

TEST_CASE("compute_logloss: 0.5 predictions -> log(2)", "[metric][logloss]")
{
    std::vector<float> const p{0.5F, 0.5F};
    std::vector<float> const y{1.0F, 0.0F};
    CHECK(compute_logloss(p, y) == Catch::Approx(std::log(2.0F)).epsilon(1e-5));
}

TEST_CASE("compute_accuracy: confident correct -> 1.0", "[metric][accuracy]")
{
    std::vector<float> const p{0.9F, 0.1F};
    std::vector<float> const y{1.0F, 0.0F};
    CHECK(compute_accuracy(p, y) == Catch::Approx(1.0F));
}

TEST_CASE("compute_accuracy: 0.5 threshold is inclusive", "[metric][accuracy]")
{
    // p=0.5 counts as predicting 1.
    std::vector<float> const p{0.5F, 0.5F};
    std::vector<float> const y{1.0F, 0.0F};
    // both predicted positive: one match (y=1), one miss (y=0) -> 0.5
    CHECK(compute_accuracy(p, y) == Catch::Approx(0.5F));
}

TEST_CASE("find_metric: known name returns the right metric", "[metric][find]")
{
    auto const m = find_metric("rmse");
    CHECK(m.name == "rmse");
    CHECK(m.task == TaskKind::regression);
    CHECK(m.compute == &compute_rmse);
}

TEST_CASE("find_metric: unknown name throws MetricNotFoundError",
          "[metric][find][error]")
{
    CHECK_THROWS_AS(find_metric("nope"), MetricNotFoundError);
}

TEST_CASE("resolve_metric_for_task: matching task returns metric", "[metric][resolve]")
{
    auto const m = resolve_metric_for_task("rmse", TaskKind::regression);
    CHECK(m.name == "rmse");
}

TEST_CASE("resolve_metric_for_task: mismatched task throws", "[metric][resolve][error]")
{
    CHECK_THROWS_AS(resolve_metric_for_task("accuracy", TaskKind::regression),
                    MetricTaskMismatchError);
    CHECK_THROWS_AS(resolve_metric_for_task("rmse", TaskKind::binary_classification),
                    MetricTaskMismatchError);
}

TEST_CASE("resolve_metric_for_task: unknown name still throws MetricNotFoundError",
          "[metric][resolve][error]")
{
    CHECK_THROWS_AS(resolve_metric_for_task("nope", TaskKind::regression),
                    MetricNotFoundError);
}
