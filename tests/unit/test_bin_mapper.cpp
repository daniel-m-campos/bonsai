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
#include "bonsai/config/errors.hpp"

using namespace bonsai; // NOLINT

constexpr auto f_nan = std::numeric_limits<float>::quiet_NaN();
constexpr auto f_inf = std::numeric_limits<float>::infinity();

TEST_CASE("BinMapper: small column produces one cut per distinct value plus sentinel",
          "[bin_mapper][fit]")
{
    std::vector<float> column = {5.0F, 2.0F, 4.0F, 1.0F, 3.0F};
    BinMapperConfig    cfg{.n_samples = column.size()};
    auto               mapper = BinMapper::fit(std::span(column), cfg);

    CHECK(std::ranges::is_sorted(mapper.cuts()));
    CHECK(mapper.cuts().back() == f_inf);
    // Distinct values fit the budget, so every one gets a right-inclusive
    // cut (issue #61); the FLT_MAX closer (decision 74) keeps the sentinel
    // missing-only even for out-of-sample values above the observed max.
    std::vector<float> expected = {
        1.0F, 2.0F, 3.0F, 4.0F, 5.0F, std::numeric_limits<float>::max(), f_inf};
    CHECK(std::ranges::equal(mapper.cuts(), expected));
    CHECK(mapper.n_bins() == mapper.cuts().size());
}

TEST_CASE("BinMapper: subsamples cuts when n_samples < column", "[bin_mapper][fit]")
{
    std::vector<float> column = {5.0F, 2.0F, 4.0F, 1.0F, 3.0F};
    BinMapperConfig    cfg{.n_samples = column.size() - 2};
    auto               mapper = BinMapper::fit(std::span(column), cfg);

    CHECK(std::ranges::is_sorted(mapper.cuts()));
    CHECK(mapper.cuts().back() == f_inf);
    CHECK(mapper.cuts().size() <= cfg.n_samples + 2); // + closer + sentinel
}

TEST_CASE("BinMapper: reserves a missing bin when column contains NaN",
          "[bin_mapper][fit][nan]")
{
    std::vector<float> column = {5.0F, f_nan, 4.0F, 1.0F, 3.0F};
    BinMapperConfig    cfg{.n_samples = column.size()};
    auto               mapper = BinMapper::fit(std::span(column), cfg);

    CHECK(std::ranges::is_sorted(mapper.cuts()));
    CHECK(mapper.cuts().back() == f_inf);
    CHECK(std::ranges::none_of(mapper.cuts(), [](float x) { return std::isnan(x); }));
}

TEST_CASE("BinMapper: duplicates are skipped", "[bin_mapper][fit][dup]")
{
    std::vector<float> column = {5.0F, 4.0F, 4.0F, 1.0F, 3.0F};
    BinMapperConfig    cfg{.n_samples = column.size()};
    auto               mapper = BinMapper::fit(std::span(column), cfg);

    CHECK(std::ranges::is_sorted(mapper.cuts()));
    CHECK(mapper.cuts().back() == f_inf);
    CHECK(std::ranges::adjacent_find(mapper.cuts()) == mapper.cuts().end());
    // 5 values, one duplicate: at most 4 distinct cuts + closer + sentinel.
    CHECK(mapper.cuts().size() <= 6);
}

TEST_CASE("BinMapper: a heavy value gets its own bin and the budget survives",
          "[bin_mapper][fit][dup]")
{
    // 50 zeros + 1..44: distinct (45) exceeds the budget (6) and the zero
    // run outweighs a mean bin. The stride walked the run and dedup burned
    // the budget (issue #63); the greedy walk isolates the heavy value and
    // still spends every cut.
    std::vector<float> column(50, 0.0F);
    for (int v = 1; v <= 44; ++v)
    {
        column.push_back(static_cast<float>(v));
    }
    BinMapperConfig cfg{.max_bin = 8, .n_samples = column.size()};
    auto            mapper = BinMapper::fit(std::span(column), cfg);

    CHECK(mapper.cuts().size() == 8); // full budget: 6 cuts + closer + sentinel
    CHECK(std::ranges::is_sorted(mapper.cuts()));
    CHECK(std::ranges::adjacent_find(mapper.cuts()) == mapper.cuts().end());
    CHECK(mapper.transform(0.0F) == 0); // the heavy value sits alone
    CHECK(mapper.transform(1.0F) == 1);
}

TEST_CASE("BinMapper: subsamples deterministically on a large seeded column",
          "[bin_mapper][fit][n_samples]")
{
    constexpr size_t   column_size = 100;
    constexpr size_t   n_samples   = 40;
    constexpr uint64_t data_seed   = 0xB047A1;

    std::mt19937                          rng(data_seed);
    std::uniform_real_distribution<float> dist(-50.0F, 50.0F);
    std::vector<float>                    column(column_size);
    std::ranges::generate(column, [&] { return dist(rng); });

    BinMapperConfig cfg{.n_samples = n_samples, .seed = 42};
    auto            mapper       = BinMapper::fit(std::span(column), cfg);
    auto            mapper_again = BinMapper::fit(std::span(column), cfg);

    CHECK(std::ranges::is_sorted(mapper.cuts()));
    CHECK(mapper.cuts().back() == f_inf);
    CHECK(mapper.cuts().size() <= n_samples + 2); // + closer + sentinel

    CHECK(std::ranges::equal(mapper.cuts(), mapper_again.cuts()));

    BinMapperConfig cfg_other{.n_samples = n_samples, .seed = 7};
    auto            mapper_other = BinMapper::fit(std::span(column), cfg_other);
    CHECK(!std::ranges::equal(mapper.cuts(), mapper_other.cuts()));
}

TEST_CASE("BinMapper: max_bin caps the cut count including the sentinel",
          "[bin_mapper][fit][max_bin]")
{
    std::mt19937                          rng(0xDEADBEEF);
    std::uniform_real_distribution<float> dist(-50.0F, 50.0F);
    std::vector<float>                    column(100);
    std::ranges::generate(column, [&] { return dist(rng); });

    BinMapperConfig cfg{.max_bin = 8, .n_samples = column.size()};
    auto            mapper = BinMapper::fit(std::span(column), cfg);

    CHECK(mapper.cuts().size() <= static_cast<size_t>(cfg.max_bin));
    CHECK(std::ranges::is_sorted(mapper.cuts()));
    CHECK(std::ranges::adjacent_find(mapper.cuts()) == mapper.cuts().end());
    CHECK(mapper.cuts().back() == f_inf);
}

TEST_CASE("BinMapper: max_bin reserves a slot for the missing bin",
          "[bin_mapper][fit][max_bin][nan]")
{
    std::mt19937                          rng(0xFEEDFACE);
    std::uniform_real_distribution<float> dist(-50.0F, 50.0F);
    std::vector<float>                    column(100);
    std::ranges::generate(column, [&] { return dist(rng); });
    for (size_t i = 0; i < column.size(); i += 11)
    {
        column[i] = f_nan;
    }

    BinMapperConfig cfg{.max_bin = 8, .n_samples = column.size()};
    auto            mapper = BinMapper::fit(std::span(column), cfg);

    // The budget's two reserved slots are now spent explicitly: the
    // FLT_MAX closer and the +inf missing sentinel (decision 74).
    CHECK(mapper.cuts().size() <= static_cast<size_t>(cfg.max_bin));
}

TEST_CASE("BinMapper: NaNs are filtered out of the subsample branch",
          "[bin_mapper][fit][nan][n_samples]")
{
    constexpr size_t column_size = 100;
    constexpr size_t n_samples   = 40;

    std::mt19937                          rng(0xCAFEF00D);
    std::uniform_real_distribution<float> dist(-50.0F, 50.0F);
    std::vector<float>                    column(column_size);
    std::ranges::generate(column, [&] { return dist(rng); });
    for (size_t i = 0; i < column.size(); i += 7)
    {
        column[i] = f_nan;
    }

    BinMapperConfig cfg{.n_samples = n_samples, .seed = 1};
    auto            mapper = BinMapper::fit(std::span(column), cfg);

    CHECK(std::ranges::none_of(mapper.cuts(), [](float x) { return std::isnan(x); }));
    CHECK(std::ranges::is_sorted(mapper.cuts()));
    CHECK(mapper.cuts().back() == f_inf);
    CHECK(mapper.cuts().size() <= n_samples + 2); // + closer + sentinel
}

TEST_CASE("BinMapper: single-value column collapses to one cut plus sentinel",
          "[bin_mapper][fit][dup]")
{
    std::vector<float> column(5, 3.0F);
    BinMapperConfig    cfg{.n_samples = column.size()};
    auto               mapper = BinMapper::fit(std::span(column), cfg);

    CHECK(mapper.cuts().size() == 3); // value + closer + sentinel
    CHECK(mapper.cuts().front() == 3.0F);
    CHECK(mapper.cuts().back() == f_inf);
    CHECK(mapper.n_bins() == 3);
}

TEST_CASE("BinMapper: all-NaN column emits only the sentinel", "[bin_mapper][fit][nan]")
{
    std::vector<float> column(5, f_nan);
    BinMapperConfig    cfg{.n_samples = column.size()};
    auto               mapper = BinMapper::fit(std::span(column), cfg);

    CHECK(mapper.cuts().size() == 2); // closer + sentinel
    CHECK(mapper.cuts().front() == std::numeric_limits<float>::max());
    CHECK(mapper.cuts().back() == f_inf);
}

TEST_CASE("BinMapper: the missing bin is NaN-only on every fitting path (issue #155)",
          "[bin_mapper][fit][closer]")
{
    // Pre-decision-74, the stride and greedy paths leaked their top values
    // into the sentinel (trained as missing, predicted by raw threshold: a
    // train/predict routing skew). One case per path: the observed maximum
    // and an out-of-sample larger value must both bin BELOW the sentinel.
    auto check_nan_only = [](BinMapper const &m, float observed_max)
    {
        auto const sentinel = static_cast<bin_id_t>(m.n_bins() - 1);
        CHECK(m.transform(observed_max) < sentinel);
        CHECK(m.transform(observed_max * 2.0F) < sentinel); // out-of-sample
        CHECK(m.transform(f_nan) == sentinel);
    };

    SECTION("stride path (continuous, over budget)")
    {
        std::mt19937                          rng(0x155);
        std::uniform_real_distribution<float> dist(0.0F, 50.0F);
        std::vector<float>                    column(100);
        std::ranges::generate(column, [&] { return dist(rng); });
        BinMapperConfig cfg{.max_bin = 8, .n_samples = column.size()};
        check_nan_only(BinMapper::fit(std::span(column), cfg),
                       std::ranges::max(column));
    }
    SECTION("greedy path (heavy value at the max, the capped-sensor shape)")
    {
        std::vector<float> column(50, 9.0F); // capped run at the maximum
        for (int v = 1; v <= 44; ++v)
        {
            column.push_back(static_cast<float>(v) / 10.0F);
        }
        BinMapperConfig cfg{.max_bin = 8, .n_samples = column.size()};
        check_nan_only(BinMapper::fit(std::span(column), cfg), 9.0F);
    }
    SECTION("distinct path (out-of-sample values above the sampled max)")
    {
        std::vector<float> column = {1.0F, 2.0F, 3.0F};
        BinMapperConfig    cfg{.n_samples = column.size()};
        check_nan_only(BinMapper::fit(std::span(column), cfg), 3.0F);
    }
}

TEST_CASE("BinMapper: transform routes values to half-open right-inclusive bins",
          "[bin_mapper][transform]")
{
    std::vector<float> column = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
    BinMapperConfig    cfg{.n_samples = column.size()};
    auto               mapper = BinMapper::fit(std::span(column), cfg);
    // cuts = [1, 2, 3, 4, 5, FLT_MAX, +inf]: one exact bin per distinct
    // value, the closer's top band above 5, and a NaN-only sentinel.

    CHECK(mapper.transform(0.0F) == 0);   // below first cut
    CHECK(mapper.transform(2.0F) == 1);   // exactly a cut, right-inclusive
    CHECK(mapper.transform(2.5F) == 2);   // between cuts
    CHECK(mapper.transform(5.0F) == 4);   // exactly last observed cut
    CHECK(mapper.transform(100.0F) == 5); // above the observed max: the closer band
    CHECK(mapper.transform(f_inf) == 6);  // non-finite joins NaN in the sentinel
}

TEST_CASE("BinMapper: transform routes NaN to the missing bin",
          "[bin_mapper][transform][nan]")
{
    std::vector<float> column = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
    BinMapperConfig    cfg{.n_samples = column.size()};
    auto               mapper = BinMapper::fit(std::span(column), cfg);

    CHECK(mapper.transform(f_nan) == mapper.n_bins() - 1);
}

TEST_CASE("BinMapper: transform is monotonic over the fitted column",
          "[bin_mapper][transform]")
{
    std::mt19937                          rng(0xABCDEF);
    std::uniform_real_distribution<float> dist(-50.0F, 50.0F);
    std::vector<float>                    column(100);
    std::ranges::generate(column, [&] { return dist(rng); });

    BinMapperConfig cfg{.n_samples = column.size()};
    auto            mapper = BinMapper::fit(std::span(column), cfg);

    std::vector<float> sorted = column;
    std::ranges::sort(sorted);

    auto last = mapper.transform(sorted.front());
    for (float x : sorted)
    {
        auto bin = mapper.transform(x);
        CHECK(bin >= last);
        last = bin;
    }
}

TEST_CASE("BinMapper: from_edges appends the sentinel and bins right-inclusive",
          "[bin_mapper][edges]")
{
    auto const mapper = BinMapper::from_edges({0.0F, 18.0F, 65.0F});

    // Two appended cuts: FLT_MAX closes the top band as a REAL bin (the split
    // scan never offers the last real bin as a candidate, so without it the
    // last user edge would be dead), and +inf keeps the missing bin NaN-only.
    std::vector<float> const expected = {0.0F, 18.0F, 65.0F,
                                         std::numeric_limits<float>::max(), f_inf};
    CHECK(std::ranges::equal(mapper.cuts(), expected));
    CHECK(mapper.n_bins() == 5); // k edges -> k+1 usable bands + missing
    // Right-inclusive, matching the fitted-cut convention (guide ch.0).
    CHECK(mapper.transform(-3.0F) == 0);
    CHECK(mapper.transform(0.0F) == 0);
    CHECK(mapper.transform(17.9F) == 1);
    CHECK(mapper.transform(18.0F) == 1);
    CHECK(mapper.transform(64.0F) == 2);
    // Above the last edge is its own real band; missing stays NaN-only.
    CHECK(mapper.transform(66.0F) == 3);
    CHECK(mapper.transform(f_nan) == 4);
}

TEST_CASE("BinMapper: from_edges rejects malformed edge lists", "[bin_mapper][edges]")
{
    CHECK_THROWS_AS(BinMapper::from_edges({}), ConfigError);
    CHECK_THROWS_AS(BinMapper::from_edges({1.0F, 1.0F}), ConfigError); // not strict
    CHECK_THROWS_AS(BinMapper::from_edges({2.0F, 1.0F}), ConfigError); // decreasing
    CHECK_THROWS_AS(BinMapper::from_edges({0.0F, f_inf}),
                    ConfigError);                                 // sentinel is ours
    CHECK_THROWS_AS(BinMapper::from_edges({f_nan}), ConfigError); // not finite
    CHECK_THROWS_AS(BinMapper::from_edges({std::numeric_limits<float>::max()}),
                    ConfigError); // FLT_MAX is the reserved top-band cut
}
