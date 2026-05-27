#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <vector>

#include "bonsai/registry/make_booster.hpp" // UnknownImplError
#include "bonsai/registry/objective_dispatch.hpp"

using namespace bonsai; // NOLINT

TEST_CASE("apply_link_inverse_by_name: logloss matches sigmoid",
          "[objective_dispatch][link][logloss]")
{
    std::vector<float> scores = {0.0F, std::log(3.0F)};
    apply_link_inverse_by_name("logloss", scores);
    CHECK(scores[0] == Catch::Approx(0.5F));
    CHECK(scores[1] == Catch::Approx(0.75F).epsilon(1e-6));
}

TEST_CASE("apply_link_inverse_by_name: mse is identity",
          "[objective_dispatch][link][mse]")
{
    std::vector<float>       scores   = {-2.0F, 0.0F, 3.5F};
    std::vector<float> const original = scores;
    apply_link_inverse_by_name("mse", scores);
    CHECK(scores == original);
}

TEST_CASE("apply_link_inverse_by_name: unknown name throws UnknownImplError",
          "[objective_dispatch][link][error]")
{
    std::vector<float> scores = {0.0F};
    CHECK_THROWS_AS(apply_link_inverse_by_name("does_not_exist", scores),
                    UnknownImplError);
}
