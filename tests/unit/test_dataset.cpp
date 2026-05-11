#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <limits>
#include <vector>

#include "bonsai/bin_mappers.hpp"
#include "bonsai/config/bin_mapper_config.hpp"
#include "bonsai/dataset.hpp"
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

TEST_CASE(
    "Dataset: bin produces column-major bin_id storage matching mapper transforms",
    "[dataset][basic]")
{
    detail::ColumnBatch batch{
        .features      = {{0.1F, 0.4F, 0.7F, 0.9F}, {-2.0F, -1.0F, 0.0F, 1.0F}},
        .labels        = {0.0F, 1.0F, 0.0F, 1.0F},
        .weights       = {},
        .feature_names = {"a", "b"},
    };
    BinMappers const mappers = BinMappers::fit(batch, BinMapperConfig{});
    Dataset const ds         = Dataset::bin(batch, mappers, {});

    REQUIRE(ds.n_features() == 2);
    REQUIRE(ds.n_rows() == 4);
    for (size_t f = 0; f < ds.n_features(); ++f)
    {
        auto bins = ds.feature_bins(f);
        REQUIRE(bins.size() == ds.n_rows());
        for (size_t r = 0; r < ds.n_rows(); ++r)
        {
            CHECK(bins[r] == ds.mappers()[f].transform(batch.features[f][r]));
        }
    }
}

TEST_CASE("Dataset: n_rows / n_features / n_bins accessors", "[dataset][accessors]")
{
    detail::ColumnBatch batch{
        .features      = {linspace(0.0F, 1.0F, 8), linspace(-3.0F, 3.0F, 8)},
        .labels        = std::vector<float>(8, 0.0F),
        .weights       = {},
        .feature_names = {"a", "b"},
    };
    BinMappers const mappers = BinMappers::fit(batch, BinMapperConfig{});
    Dataset const ds         = Dataset::bin(batch, mappers, {});

    CHECK(ds.n_rows() == 8);
    CHECK(ds.n_features() == 2);
    for (size_t f = 0; f < ds.n_features(); ++f)
    {
        CHECK(ds.n_bins(f) == ds.mappers()[f].n_bins());
    }
}

TEST_CASE("Dataset: labels pass through, weights empty when uniform",
          "[dataset][labels]")
{
    detail::ColumnBatch batch{
        .features      = {linspace(0.0F, 1.0F, 4)},
        .labels        = {0.0F, 1.0F, 0.0F, 1.0F},
        .weights       = {},
        .feature_names = {"a"},
    };
    BinMappers const mappers = BinMappers::fit(batch, BinMapperConfig{});
    Dataset const ds         = Dataset::bin(batch, mappers, {});

    auto lab = ds.labels();
    REQUIRE(lab.size() == 4);
    CHECK(lab[0] == 0.0F);
    CHECK(lab[1] == 1.0F);
    CHECK(ds.weights().empty());
}

TEST_CASE("Dataset: weights pass through when provided", "[dataset][weights]")
{
    detail::ColumnBatch batch{
        .features      = {linspace(0.0F, 1.0F, 4)},
        .labels        = {0.0F, 1.0F, 0.0F, 1.0F},
        .weights       = {0.5F, 1.0F, 2.0F, 1.5F},
        .feature_names = {"a"},
    };
    BinMappers const mappers = BinMappers::fit(batch, BinMapperConfig{});
    Dataset const ds         = Dataset::bin(batch, mappers, {});

    auto w = ds.weights();
    REQUIRE(w.size() == 4);
    CHECK(w[0] == 0.5F);
    CHECK(w[3] == 1.5F);
}

TEST_CASE("Dataset: NaN row routes to missing bin (last bin)", "[dataset][missing]")
{
    auto const nan = std::numeric_limits<float>::quiet_NaN();
    detail::ColumnBatch batch{
        .features      = {{0.1F, 0.5F, nan, 0.9F}},
        .labels        = {0.0F, 0.0F, 0.0F, 0.0F},
        .weights       = {},
        .feature_names = {"a"},
    };
    BinMappers const mappers = BinMappers::fit(batch, BinMapperConfig{});
    Dataset const ds         = Dataset::bin(batch, mappers, {});

    auto bins = ds.feature_bins(0);
    REQUIRE(bins.size() == 4);
    CHECK(bins[2] == ds.n_bins(0) - 1);
}
