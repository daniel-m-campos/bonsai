#include <catch2/catch_test_macros.hpp>

#include <bit>
#include <cmath>
#include <limits>
#include <vector>

#include "cuda/detail/cut_order.hpp"

using namespace bonsai::cuda_detail;

namespace
{

FeatBest cut(double gain, int32_t bin, int32_t dl, int32_t valid = 1)
{
    return {.gain  = gain,
            .gL    = 0.0,
            .hL    = 0.0,
            .gR    = 0.0,
            .hR    = 0.0,
            .bin   = bin,
            .dl    = dl,
            .valid = valid,
            .sel   = 0};
}

std::vector<double> positive_gains()
{
    return {std::numeric_limits<double>::denorm_min(),
            std::numeric_limits<double>::min(),
            1e-300,
            0.5,
            1.0,
            std::nextafter(1.0, 2.0),
            2.0,
            1e10,
            std::numeric_limits<double>::max(),
            std::numeric_limits<double>::infinity()};
}

} // namespace

// INVARIANT: cut-order-key
// A valid cut carries a strictly positive gain: score_cut_exact admits a
// candidate only when gain > 0.0, and the level finder only when it beats a
// best seeded at 0.0. On that domain the bit pattern of a double orders like
// its value, so split_better, which compares the pattern to keep the warp
// scan on the integer pipe, ranks every pair exactly as feat_better does.
// If either finder ever marked a non-positive or NaN gain valid, the two
// finders would disagree and this test would fail on the -0.0 and NaN rows.
TEST_CASE("CutOrder: split_better agrees with feat_better on every valid pair",
          "[cuda][finder][edge]")
{
    auto const gains = positive_gains();
    for (double const ga : gains)
    {
        for (double const gb : gains)
        {
            for (int32_t const ba : {0, 7, 254})
            {
                for (int32_t const bb : {0, 7, 254})
                {
                    for (int32_t const da : {0, 1})
                    {
                        for (int32_t const db : {0, 1})
                        {
                            FeatBest const a = cut(ga, ba, da);
                            FeatBest const b = cut(gb, bb, db);
                            REQUIRE(split_better(a, b) == feat_better(a, b));
                            REQUIRE(split_better(b, a) == feat_better(b, a));
                        }
                    }
                }
            }
        }
    }
}

TEST_CASE(
    "CutOrder: an invalid cut loses to any valid cut and ties another invalid one",
    "[cuda][finder]")
{
    FeatBest const none = cut(0.0, 0, 0, 0);
    FeatBest const some = cut(std::numeric_limits<double>::denorm_min(), 255, 0);
    REQUIRE(split_better(some, none));
    REQUIRE_FALSE(split_better(none, some));
    REQUIRE_FALSE(split_better(none, cut(1e300, 3, 1, 0)));
    REQUIRE(feat_better(some, none));
    REQUIRE_FALSE(feat_better(none, some));
}

TEST_CASE("CutOrder: the key order breaks outside the positive domain",
          "[cuda][finder][edge][nan]")
{
    double const nan = std::numeric_limits<double>::quiet_NaN();
    REQUIRE(key_of_positive_gain(-0.0) < key_of_positive_gain(0.0));
    REQUIRE(key_of_positive_gain(-1.0) < key_of_positive_gain(-2.0));
    REQUIRE(key_of_positive_gain(nan) > key_of_positive_gain(1e300));
    FeatBest const plus  = cut(0.0, 1, 0);
    FeatBest const minus = cut(-0.0, 0, 0);
    REQUIRE(split_better(plus, minus));
    REQUIRE_FALSE(feat_better(plus, minus));
}
