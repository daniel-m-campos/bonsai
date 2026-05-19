#include <catch2/catch_test_macros.hpp>

#include "bonsai/objective.hpp"
#include "bonsai/objective_traits.hpp"
#include "bonsai/registry/make_booster.hpp" // UnknownImplError
#include "bonsai/registry/objective_dispatch.hpp"
#include "bonsai/task.hpp"

using namespace bonsai; // NOLINT

TEST_CASE("task_of<MSEObjective>::value is regression", "[task]")
{
    static_assert(task_of<MSEObjective>::value == TaskKind::regression);
    CHECK(task_of<MSEObjective>::value == TaskKind::regression);
}

TEST_CASE("task_of<LogLossObjective>::value is binary_classification", "[task]")
{
    static_assert(task_of<LogLossObjective>::value == TaskKind::binary_classification);
    CHECK(task_of<LogLossObjective>::value == TaskKind::binary_classification);
}

TEST_CASE("task_kind_by_name dispatches by string name", "[task][dispatch]")
{
    CHECK(task_kind_by_name("mse") == TaskKind::regression);
    CHECK(task_kind_by_name("logloss") == TaskKind::binary_classification);
}

TEST_CASE("task_kind_by_name: unknown name throws UnknownImplError",
          "[task][dispatch][error]")
{
    CHECK_THROWS_AS(task_kind_by_name("nope"), UnknownImplError);
}

TEST_CASE("task_kind_name returns the canonical string", "[task]")
{
    CHECK(task_kind_name(TaskKind::regression) == "regression");
    CHECK(task_kind_name(TaskKind::binary_classification) == "binary_classification");
}

TEST_CASE("default_metric_names_by_name returns objective's declared defaults",
          "[task][dispatch]")
{
    auto const mse_defaults = default_metric_names_by_name("mse");
    REQUIRE(mse_defaults.size() == 1);
    CHECK(mse_defaults[0] == "rmse");

    auto const ll_defaults = default_metric_names_by_name("logloss");
    REQUIRE(ll_defaults.size() == 2);
    CHECK(ll_defaults[0] == "logloss");
    CHECK(ll_defaults[1] == "accuracy");
}

TEST_CASE("default_metric_names_by_name: unknown name throws",
          "[task][dispatch][error]")
{
    CHECK_THROWS_AS(default_metric_names_by_name("nope"), UnknownImplError);
}
