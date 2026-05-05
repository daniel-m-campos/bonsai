#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <vector>

#include "bonsai/bin_mapper.hpp"
#include "bonsai/config/bin_mapper_config.hpp"

using namespace bonsai; // NOLINT

auto constexpr f_nan = std::numeric_limits<float>::quiet_NaN();
auto constexpr f_inf = std::numeric_limits<float>::infinity();

TEST_CASE("BinMapper: small column produces one cut per distinct value plus sentinel",
          "[bin_mapper][fit]")
{
    std::vector<float> column = {5.0F, 2.0F, 4.0F, 1.0F, 3.0F};
    BinMapperConfig cfg{.n_samples = column.size()};
    auto mapper = BinMapper::fit(std::span(column), cfg);

    CHECK_FALSE(mapper.has_missing_bin());
    CHECK(std::ranges::is_sorted(mapper.cuts()));
    CHECK(mapper.cuts().back() == f_inf);
    // Bin 0 is (-inf, cuts[0]] so the smallest value gets no cut of its own.
    std::vector<float> expected = {2.0F, 3.0F, 4.0F, 5.0F, f_inf};
    CHECK(std::ranges::equal(mapper.cuts(), expected));
    CHECK(mapper.n_buckets() == mapper.cuts().size());
}

TEST_CASE("BinMapper: subsamples cuts when n_samples < column", "[bin_mapper][fit]")
{
    std::vector<float> column = {5.0F, 2.0F, 4.0F, 1.0F, 3.0F};
    BinMapperConfig cfg{.n_samples = column.size() - 2};
    auto mapper = BinMapper::fit(std::span(column), cfg);

    CHECK_FALSE(mapper.has_missing_bin());
    CHECK(std::ranges::is_sorted(mapper.cuts()));
    CHECK(mapper.cuts().back() == f_inf);
    CHECK(mapper.cuts().size() <= cfg.n_samples);
}

TEST_CASE("BinMapper: reserves a missing bin when column contains NaN",
          "[bin_mapper][fit][nan]")
{
    std::vector<float> column = {5.0F, f_nan, 4.0F, 1.0F, 3.0F};
    BinMapperConfig cfg{.n_samples = column.size()};
    auto mapper = BinMapper::fit(std::span(column), cfg);

    CHECK(mapper.has_missing_bin());
    CHECK(std::ranges::is_sorted(mapper.cuts()));
    CHECK(mapper.cuts().back() == f_inf);
    CHECK(std::ranges::none_of(mapper.cuts(), [](float x) { return std::isnan(x); }));
}

TEST_CASE("BinMapper: duplicates are skipped", "[bin_mapper][fit][dup]")
{
    std::vector<float> column = {5.0F, 4.0F, 4.0F, 1.0F, 3.0F};
    BinMapperConfig cfg{.n_samples = column.size()};
    auto mapper = BinMapper::fit(std::span(column), cfg);

    CHECK_FALSE(mapper.has_missing_bin());
    CHECK(std::ranges::is_sorted(mapper.cuts()));
    CHECK(mapper.cuts().back() == f_inf);
    CHECK(std::ranges::adjacent_find(mapper.cuts()) == mapper.cuts().end());
    // 5 values, one duplicate, so at most 4 distinct cuts + sentinel.
    CHECK(mapper.cuts().size() <= 5);
}

TEST_CASE("BinMapper: subsamples deterministically on a large seeded column",
          "[bin_mapper][fit][n_samples]")
{
    size_t constexpr column_size = 100;
    size_t constexpr n_samples   = 40;
    uint64_t constexpr data_seed = 0xB047A1;

    std::mt19937 rng(data_seed);
    std::uniform_real_distribution<float> dist(-50.0F, 50.0F);
    std::vector<float> column(column_size);
    std::ranges::generate(column, [&] { return dist(rng); });

    BinMapperConfig cfg{.n_samples = n_samples, .seed = 42};
    auto mapper       = BinMapper::fit(std::span(column), cfg);
    auto mapper_again = BinMapper::fit(std::span(column), cfg);

    CHECK_FALSE(mapper.has_missing_bin());
    CHECK(std::ranges::is_sorted(mapper.cuts()));
    CHECK(mapper.cuts().back() == f_inf);
    CHECK(mapper.cuts().size() <= n_samples);

    CHECK(std::ranges::equal(mapper.cuts(), mapper_again.cuts()));

    BinMapperConfig cfg_other{.n_samples = n_samples, .seed = 7};
    auto mapper_other = BinMapper::fit(std::span(column), cfg_other);
    CHECK(!std::ranges::equal(mapper.cuts(), mapper_other.cuts()));
}

TEST_CASE("BinMapper: max_bin caps the cut count including the sentinel",
          "[bin_mapper][fit][max_bin]")
{
    std::mt19937 rng(0xDEADBEEF);
    std::uniform_real_distribution<float> dist(-50.0F, 50.0F);
    std::vector<float> column(100);
    std::ranges::generate(column, [&] { return dist(rng); });

    BinMapperConfig cfg{.max_bin = 8, .n_samples = column.size()};
    auto mapper = BinMapper::fit(std::span(column), cfg);

    CHECK_FALSE(mapper.has_missing_bin());
    CHECK(mapper.cuts().size() <= static_cast<size_t>(cfg.max_bin));
    CHECK(std::ranges::is_sorted(mapper.cuts()));
    CHECK(std::ranges::adjacent_find(mapper.cuts()) == mapper.cuts().end());
    CHECK(mapper.cuts().back() == f_inf);
}

TEST_CASE("BinMapper: max_bin reserves a slot for the missing bin",
          "[bin_mapper][fit][max_bin][nan]")
{
    std::mt19937 rng(0xFEEDFACE);
    std::uniform_real_distribution<float> dist(-50.0F, 50.0F);
    std::vector<float> column(100);
    std::ranges::generate(column, [&] { return dist(rng); });
    for (size_t i = 0; i < column.size(); i += 11)
    {
        column[i] = f_nan;
    }

    BinMapperConfig cfg{.max_bin = 8, .n_samples = column.size()};
    auto mapper = BinMapper::fit(std::span(column), cfg);

    CHECK(mapper.has_missing_bin());
    // Total bin budget includes the missing bin, so cuts <= max_bin - 1.
    CHECK(mapper.cuts().size() <= static_cast<size_t>(cfg.max_bin) - 1);
}

TEST_CASE("BinMapper: NaNs are filtered out of the subsample branch",
          "[bin_mapper][fit][nan][n_samples]")
{
    size_t constexpr column_size = 100;
    size_t constexpr n_samples   = 40;

    std::mt19937 rng(0xCAFEF00D);
    std::uniform_real_distribution<float> dist(-50.0F, 50.0F);
    std::vector<float> column(column_size);
    std::ranges::generate(column, [&] { return dist(rng); });
    for (size_t i = 0; i < column.size(); i += 7)
    {
        column[i] = f_nan;
    }

    BinMapperConfig cfg{.n_samples = n_samples, .seed = 1};
    auto mapper = BinMapper::fit(std::span(column), cfg);

    CHECK(mapper.has_missing_bin());
    CHECK(std::ranges::none_of(mapper.cuts(), [](float x) { return std::isnan(x); }));
    CHECK(std::ranges::is_sorted(mapper.cuts()));
    CHECK(mapper.cuts().back() == f_inf);
    CHECK(mapper.cuts().size() <= n_samples);
}

TEST_CASE("BinMapper: single-value column collapses to one cut plus sentinel",
          "[bin_mapper][fit][dup]")
{
    std::vector<float> column(5, 3.0F);
    BinMapperConfig cfg{.n_samples = column.size()};
    auto mapper = BinMapper::fit(std::span(column), cfg);

    CHECK_FALSE(mapper.has_missing_bin());
    CHECK(mapper.cuts().size() == 2);
    CHECK(mapper.cuts().front() == 3.0F);
    CHECK(mapper.cuts().back() == f_inf);
    CHECK(mapper.n_buckets() == 2);
}

TEST_CASE("BinMapper: all-NaN column emits only the sentinel", "[bin_mapper][fit][nan]")
{
    std::vector<float> column(5, f_nan);
    BinMapperConfig cfg{.n_samples = column.size()};
    auto mapper = BinMapper::fit(std::span(column), cfg);

    CHECK(mapper.has_missing_bin());
    CHECK(mapper.cuts().size() == 1);
    CHECK(mapper.cuts().front() == f_inf);
}
