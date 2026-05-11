#include <catch2/catch_test_macros.hpp>
#include <cstddef>

#include "bonsai/histogram.hpp"

using namespace bonsai; // NOLINT

TEST_CASE("Histogram: ctor sizes cells to n_bins and zero-initializes",
          "[histogram][ctor]")
{
    Histogram hist{4};

    REQUIRE(hist.size() == 4);
    for (size_t b = 0; b < hist.size(); ++b)
    {
        CHECK(hist[b].sum_grad == 0.0);
        CHECK(hist[b].sum_hess == 0.0);
    }
}

TEST_CASE("Histogram: add accumulates grad and hess into the indexed cell",
          "[histogram][add]")
{
    Histogram hist{3};

    // Values chosen to be exact in IEEE 754 (powers of two, dyadic fractions).
    hist.add(1, 0.5, 1.0);
    hist.add(1, -0.25, 2.0);
    hist.add(0, 0.125, 4.0);

    CHECK(hist[0].sum_grad == 0.125);
    CHECK(hist[0].sum_hess == 4.0);
    CHECK(hist[1].sum_grad == 0.25); // 0.5 + (-0.25)
    CHECK(hist[1].sum_hess == 3.0);
    CHECK(hist[2].sum_grad == 0.0);
    CHECK(hist[2].sum_hess == 0.0);
}

TEST_CASE("Histogram: clear zeroes cells and preserves size", "[histogram][clear]")
{
    Histogram hist{3};
    hist.add(0, 1.0, 1.0);
    hist.add(2, -1.0, 1.0);

    hist.clear();

    REQUIRE(hist.size() == 3);
    for (size_t b = 0; b < hist.size(); ++b)
    {
        CHECK(hist[b].sum_grad == 0.0);
        CHECK(hist[b].sum_hess == 0.0);
    }
}

TEST_CASE("Histogram: operator-= computes parent minus sibling elementwise",
          "[histogram][subtraction]")
{
    // Dyadic fractions so subtraction is bit-exact.
    Histogram parent{4};
    parent.add(0, -1.0, 2.0);
    parent.add(1, +1.0, 2.0);
    parent.add(2, +0.25, 1.0);
    parent.add(3, +0.5, 1.0); // missing bin

    Histogram sibling{4};
    sibling.add(0, -0.5, 1.0);
    sibling.add(1, +0.25, 1.0);
    sibling.add(3, +0.5, 1.0);

    parent -= sibling;

    CHECK(parent[0].sum_grad == -0.5);
    CHECK(parent[0].sum_hess == 1.0);
    CHECK(parent[1].sum_grad == +0.75);
    CHECK(parent[1].sum_hess == 1.0);
    CHECK(parent[2].sum_grad == +0.25);
    CHECK(parent[2].sum_hess == 1.0);
    CHECK(parent[3].sum_grad == 0.0);
    CHECK(parent[3].sum_hess == 0.0);
}

TEST_CASE("Histogram: missing-bin cell accumulates like any other cell",
          "[histogram][add][nan]")
{
    Histogram hist{4}; // bins 0..2 real, bin 3 missing per BinMapper convention

    hist.add(3, 0.5, 1.0);
    hist.add(3, -0.125, 1.0);

    CHECK(hist[3].sum_grad == 0.375);
    CHECK(hist[3].sum_hess == 2.0);
    // Real bins untouched.
    CHECK(hist[0].sum_grad == 0.0);
    CHECK(hist[1].sum_grad == 0.0);
    CHECK(hist[2].sum_grad == 0.0);
}

TEST_CASE("Histogram: missing returns the last cell", "[histogram][nan]")
{
    Histogram hist{3};
    hist.add(0, 0.25, 1.0);
    hist.add(1, -0.5, 1.0);
    hist.add(2, 0.125, 1.0); // bin 2 = missing per BinMapper convention

    CHECK(hist.missing().sum_grad == 0.125);
    CHECK(hist.missing().sum_hess == 1.0);
}

TEST_CASE("Histogram: reals returns a span over real bins only, excluding missing",
          "[histogram]")
{
    Histogram hist{4};
    hist.add(0, 0.5, 1.0);
    hist.add(1, -0.25, 1.0);
    hist.add(2, 0.125, 1.0);
    hist.add(3, 1.0, 1.0); // bin 3 = missing

    auto reals = hist.sweep_cells();

    REQUIRE(reals.size() == 3);
    CHECK(reals[0].sum_grad == 0.5);
    CHECK(reals[1].sum_grad == -0.25);
    CHECK(reals[2].sum_grad == 0.125);
}

TEST_CASE("Histogram: reals iterates the real bins in order", "[histogram]")
{
    Histogram hist{3};
    hist.add(0, 0.25, 1.0);
    hist.add(1, -0.5, 2.0);
    // bin 2 = missing, intentionally non-zero to confirm exclusion
    hist.add(2, 99.0, 99.0);

    double total_grad = 0.0;
    double total_hess = 0.0;
    for (auto const &cell : hist.sweep_cells())
    {
        total_grad += cell.sum_grad;
        total_hess += cell.sum_hess;
    }

    CHECK(total_grad == -0.25); // 0.25 + (-0.5)
    CHECK(total_hess == 3.0);   // 1 + 2; missing-bin's 99 excluded
}
