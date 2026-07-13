#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include "bonsai/bin_mappers.hpp"
#include "bonsai/config/bin_mapper_config.hpp"
#include "bonsai/detail/column_batch.hpp"
#include "bonsai/types.hpp"

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

// A feature-major batch of seeded random columns, larger than the sample.
detail::ColumnBatch random_batch(size_t n_rows, size_t n_feats)
{
    std::mt19937                          rng(123);
    std::uniform_real_distribution<float> u(0.0F, 1.0F);
    detail::ColumnBatch                   batch;
    batch.labels.assign(n_rows, 0.0F);
    for (size_t f = 0; f < n_feats; ++f)
    {
        std::vector<float> col(n_rows);
        for (auto &v : col)
        {
            v = u(rng);
        }
        batch.features.push_back(std::move(col));
        batch.feature_names.push_back("f" + std::to_string(f));
    }
    return batch;
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
    BinMappers const    m = BinMappers::fit(batch, BinMapperConfig{});
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

TEST_CASE("BinMappers: shared-sample fit is deterministic above n_samples",
          "[bin_mappers][n_samples]")
{
    // n_rows > n_samples so the shared row sample kicks in (decision 64); the
    // seeded selection must be reproducible run to run, cut for cut.
    auto const            batch = random_batch(4000, 4);
    BinMapperConfig const cfg{.n_samples = 500, .seed = 42};

    BinMappers const a = BinMappers::fit(batch, cfg);
    BinMappers const b = BinMappers::fit(batch, cfg);

    REQUIRE(a.size() == b.size());
    for (size_t f = 0; f < a.size(); ++f)
    {
        auto const ca = a[f].cuts();
        auto const cb = b[f].cuts();
        REQUIRE(ca.size() == cb.size());
        for (size_t i = 0; i < ca.size(); ++i)
        {
            CHECK(ca[i] == cb[i]);
        }
    }
}

TEST_CASE("BinMappers: ColumnBatch and features_view overloads agree above n_samples",
          "[bin_mappers][n_samples][overload]")
{
    // Both overloads draw the same shared row sample and gather the same rows,
    // so cuts must be bit-identical even when sampling is active.
    size_t const          n_rows = 4000;
    size_t const          n_feat = 4;
    auto const            batch  = random_batch(n_rows, n_feat);
    BinMapperConfig const cfg{.n_samples = 500, .seed = 42};

    // Row-major mirror of the feature-major batch for the features_view overload.
    std::vector<float> row_major(n_rows * n_feat);
    for (size_t r = 0; r < n_rows; ++r)
    {
        for (size_t f = 0; f < n_feat; ++f)
        {
            row_major[(r * n_feat) + f] = batch.features[f][r];
        }
    }
    features_view const view{row_major.data(), n_rows, n_feat};

    BinMappers const from_batch = BinMappers::fit(batch, cfg);
    BinMappers const from_view =
        BinMappers::fit(view, std::vector<std::string>(n_feat, "x"), cfg);

    REQUIRE(from_batch.size() == from_view.size());
    for (size_t f = 0; f < from_batch.size(); ++f)
    {
        auto const cb = from_batch[f].cuts();
        auto const cv = from_view[f].cuts();
        REQUIRE(cb.size() == cv.size());
        for (size_t i = 0; i < cb.size(); ++i)
        {
            CHECK(cb[i] == cv[i]);
        }
    }
}
