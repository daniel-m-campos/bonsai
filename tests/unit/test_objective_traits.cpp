#include <algorithm>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <vector>

#include "bonsai/objective.hpp"
#include "bonsai/objective_traits.hpp"

using namespace bonsai; // NOLINT

TEST_CASE("link_inverse_of<MSEObjective>: apply is identity", "[link_inverse][mse]")
{
    std::vector<float>       scores   = {-3.5F, 0.0F, 1.25F, 17.0F, -0.125F};
    std::vector<float> const original = scores;

    link_inverse_of<MSEObjective>::apply(scores);

    CHECK(std::ranges::equal(scores, original));
}

TEST_CASE("link_inverse_of<LogLossObjective>: apply matches sigmoid",
          "[link_inverse][logloss]")
{
    SECTION("zero maps to 0.5 exactly")
    {
        std::vector<float> scores = {0.0F};
        link_inverse_of<LogLossObjective>::apply(scores);
        CHECK(scores[0] == 0.5F);
    }
    SECTION("log(3) maps to 0.75")
    {
        std::vector<float> scores = {std::log(3.0F)};
        link_inverse_of<LogLossObjective>::apply(scores);
        CHECK(scores[0] == Catch::Approx(0.75F).epsilon(1e-6));
    }
    SECTION("large positive saturates to finite 1")
    {
        std::vector<float> scores = {100.0F};
        link_inverse_of<LogLossObjective>::apply(scores);
        CHECK(scores[0] == Catch::Approx(1.0F));
        CHECK(std::isfinite(scores[0]));
    }
    SECTION("large negative saturates to finite 0")
    {
        std::vector<float> scores = {-100.0F};
        link_inverse_of<LogLossObjective>::apply(scores);
        CHECK(scores[0] == Catch::Approx(0.0F));
        CHECK(std::isfinite(scores[0]));
    }
}
