#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <span>
#include <vector>

#include "bonsai/monotone.hpp"

using namespace bonsai; // NOLINT

namespace
{

std::vector<float> projected(std::vector<int> const   &levels,
                             std::vector<float> const &weights,
                             std::vector<float>        table)
{
    project_monotone(std::span<int const>{levels}, std::span<float const>{weights},
                     std::span<float>{table});
    return table;
}

} // namespace

TEST_CASE("MonotoneProjection: an unconstrained table is left alone",
          "[monotone][edge]")
{
    std::vector<float> const table{5.0F, 1.0F, 9.0F, 2.0F};
    CHECK(projected({0, 0}, {1.0F, 1.0F, 1.0F, 1.0F}, table) == table);
}

TEST_CASE("MonotoneProjection: an already-ordered table is left alone", "[monotone]")
{
    std::vector<float> const table{1.0F, 2.0F, 3.0F, 4.0F};
    CHECK(projected({+1, +1}, {1.0F, 1.0F, 1.0F, 1.0F}, table) == table);
}

TEST_CASE("MonotoneProjection: an increasing level averages an inverted pair",
          "[monotone]")
{
    // Leaf 0 is the left child (feature below the cut), leaf 1 the right. A +1
    // constraint requires leaf 0 <= leaf 1, so the inverted pair merges at the
    // weighted mean and both leaves take it.
    auto const out = projected({+1}, {1.0F, 1.0F}, {5.0F, 1.0F});
    CHECK(out[0] == Catch::Approx(3.0F));
    CHECK(out[1] == Catch::Approx(3.0F));
}

TEST_CASE("MonotoneProjection: a decreasing level orders the pair the other way",
          "[monotone]")
{
    CHECK(projected({-1}, {1.0F, 1.0F}, {5.0F, 1.0F}) ==
          std::vector<float>{5.0F, 1.0F});

    auto const out = projected({-1}, {1.0F, 1.0F}, {1.0F, 5.0F});
    CHECK(out[0] == Catch::Approx(3.0F));
    CHECK(out[1] == Catch::Approx(3.0F));
}

TEST_CASE("MonotoneProjection: weights pull the merged value toward the heavier leaf",
          "[monotone]")
{
    // Same inverted pair as above, but leaf 1 carries three times the hessian:
    // the merged value is (1*4 + 3*0) / 4 = 1, not the unweighted 2.
    auto const out = projected({+1}, {1.0F, 3.0F}, {4.0F, 0.0F});
    CHECK(out[0] == Catch::Approx(1.0F));
    CHECK(out[1] == Catch::Approx(1.0F));
}

TEST_CASE("MonotoneProjection: a free level splits the leaves into independent groups",
          "[monotone]")
{
    // Level 0 free, level 1 constrained +1. Level 0 is the high bit, so leaves
    // {0,1} form one group and {2,3} another; an inversion in one must not move
    // the other.
    auto const out =
        projected({0, +1}, {1.0F, 1.0F, 1.0F, 1.0F}, {5.0F, 1.0F, 10.0F, 20.0F});
    CHECK(out[0] == Catch::Approx(3.0F));
    CHECK(out[1] == Catch::Approx(3.0F));
    CHECK(out[2] == Catch::Approx(10.0F));
    CHECK(out[3] == Catch::Approx(20.0F));
}

TEST_CASE("MonotoneProjection: two constrained levels satisfy the product order",
          "[monotone]")
{
    // Both levels +1. Leaf index bits are (level0, level1), so leaf 0 is the
    // lattice bottom and leaf 3 the top, with 1 and 2 incomparable to each
    // other. A fully inverted table must come out non-decreasing along every
    // covering pair.
    auto const out =
        projected({+1, +1}, {1.0F, 1.0F, 1.0F, 1.0F}, {4.0F, 3.0F, 2.0F, 1.0F});
    CHECK(out[0] <= out[1]);
    CHECK(out[0] <= out[2]);
    CHECK(out[1] <= out[3]);
    CHECK(out[2] <= out[3]);
}

TEST_CASE("MonotoneProjection: mixed directions order each level by its own sign",
          "[monotone]")
{
    // Level 0 increasing, level 1 decreasing. Within a fixed level-0 bit the
    // right child must not exceed the left; across level 0 the low half must
    // not exceed the high half at the same level-1 outcome.
    auto const out =
        projected({+1, -1}, {1.0F, 1.0F, 1.0F, 1.0F}, {1.0F, 2.0F, 3.0F, 4.0F});
    CHECK(out[0] >= out[1]);
    CHECK(out[2] >= out[3]);
    CHECK(out[0] <= out[2]);
    CHECK(out[1] <= out[3]);
}

TEST_CASE("MonotoneProjection: a zero-weight block falls back to the plain mean",
          "[monotone][edge]")
{
    // Empty leaves carry zero hessian and a zero value. A merged block of only
    // those has no weighted mean to take, and dividing by its total weight
    // would be a division by zero rather than a number.
    auto const out = projected({+1}, {0.0F, 0.0F}, {5.0F, 1.0F});
    CHECK(out[0] == Catch::Approx(3.0F));
    CHECK(out[1] == Catch::Approx(3.0F));
}

TEST_CASE("MonotoneProjection: a longer chain merges only what it must", "[monotone]")
{
    // One constrained level over three levels of tree: leaves 0..7 with level 2
    // constrained means four independent adjacent pairs.
    auto const out = projected({0, 0, +1}, std::vector<float>(8, 1.0F),
                               {2.0F, 1.0F, 0.0F, 5.0F, 9.0F, 9.0F, 4.0F, 3.0F});
    CHECK(out[0] == Catch::Approx(1.5F));
    CHECK(out[1] == Catch::Approx(1.5F));
    CHECK(out[2] == Catch::Approx(0.0F));
    CHECK(out[3] == Catch::Approx(5.0F));
    CHECK(out[4] == Catch::Approx(9.0F));
    CHECK(out[5] == Catch::Approx(9.0F));
    CHECK(out[6] == Catch::Approx(3.5F));
    CHECK(out[7] == Catch::Approx(3.5F));
}
