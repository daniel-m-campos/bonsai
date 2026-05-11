#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <string>
#include <vector>

#include "bonsai/bin_mappers.hpp"
#include "bonsai/config/bin_mapper_config.hpp"
#include "bonsai/detail/column_batch.hpp"

using namespace bonsai; // NOLINT

namespace
{

std::vector<float> linspace(float lo, float hi, size_t n)
{
    std::vector<float> v;
    v.reserve(n);
    float const step = (hi - lo) / static_cast<float>(n - 1);
    for (size_t i = 0; i < n; ++i)
    {
        v.push_back(lo + (step * static_cast<float>(i)));
    }
    return v;
}

} // namespace

TEST_CASE("BinMappers: fit produces one mapper per feature, preserves names",
          "[bin_mappers][basic]")
{
    detail::ColumnBatch batch{
        .features      = {linspace(0.0F, 1.0F, 10), linspace(-5.0F, 5.0F, 10),
                          linspace(100.0F, 200.0F, 10)},
        .labels        = std::vector<float>(10, 0.0F),
        .weights       = {},
        .feature_names = {"a", "b", "c"},
    };

    BinMappers const m = BinMappers::fit(batch, BinMapperConfig{});

    REQUIRE(m.size() == 3);
    auto names = m.feature_names();
    REQUIRE(names.size() == 3);
    CHECK(std::string(names[0]) == "a");
    CHECK(std::string(names[1]) == "b");
    CHECK(std::string(names[2]) == "c");

    // Spot-check: transforming an in-range value returns a real bin index.
    auto const b = m[0].transform(0.5F);
    CHECK(b < m[0].n_bins());
}

TEST_CASE("BinMappers: empty ColumnBatch produces empty mapper set",
          "[bin_mappers][edge]")
{
    detail::ColumnBatch batch{};
    BinMappers const m = BinMappers::fit(batch, BinMapperConfig{});
    CHECK(m.size() == 0);
    CHECK(m.feature_names().empty());
}

TEST_CASE("BinMappers: fit is independent across features",
          "[bin_mappers][independence]")
{
    // Feature 0: only two distinct values → quantile sweep yields few cuts.
    // Feature 1: many distinct values → many cuts.
    std::vector<float> sparse(64, 0.0F);
    for (size_t i = 32; i < 64; ++i)
    {
        sparse[i] = 1.0F;
    }
    detail::ColumnBatch batch{
        .features      = {sparse, linspace(0.0F, 100.0F, 64)},
        .labels        = std::vector<float>(64, 0.0F),
        .weights       = {},
        .feature_names = {"sparse", "dense"},
    };

    BinMappers const m = BinMappers::fit(batch, BinMapperConfig{});

    REQUIRE(m.size() == 2);
    CHECK(m[0].n_bins() < m[1].n_bins());
}
