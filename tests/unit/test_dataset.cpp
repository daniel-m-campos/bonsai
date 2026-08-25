#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <vector>

#include "bonsai/bin_mapper.hpp"
#include "bonsai/bin_mappers.hpp"
#include "bonsai/booster.hpp"
#include "bonsai/config/bin_mapper_config.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/config/data_config.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/detail/column_batch.hpp"
#include "bonsai/io/model.hpp"
#include "bonsai/registry/make_booster.hpp"
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

// A dataset's binned columns, copied out; exactly one width is populated,
// as Dataset::from_bins expects them back.
struct Bins
{
    std::vector<std::vector<uint8_t>>  u8;
    std::vector<std::vector<uint16_t>> u16;
};

Bins bins_of(Dataset const &ds)
{
    Bins out;
    out.u8.resize(ds.bins_are_u8() ? ds.n_features() : 0);
    out.u16.resize(ds.bins_are_u8() ? 0 : ds.n_features());
    for (size_t f = 0; f < ds.n_features(); ++f)
    {
        ds.visit_bins(f,
                      [&](auto col)
                      {
                          using T =
                              std::remove_const_t<typename decltype(col)::element_type>;
                          if constexpr (std::is_same_v<T, uint8_t>)
                          {
                              out.u8[f].assign(col.begin(), col.end());
                          }
                          else
                          {
                              out.u16[f].assign(col.begin(), col.end());
                          }
                      });
    }
    return out;
}

detail::ColumnBatch two_feature_batch(size_t n)
{
    std::vector<float> labels;
    labels.reserve(n);
    for (size_t i = 0; i < n; ++i)
    {
        labels.push_back(static_cast<float>(i % 3));
    }
    return detail::ColumnBatch{
        .features      = {linspace(0.0F, 1.0F, n), linspace(-3.0F, 3.0F, n)},
        .labels        = std::move(labels),
        .weights       = {},
        .feature_names = {"a", "b"},
    };
}

Config from_bins_cfg()
{
    Config cfg{};
    cfg.tree_config.min_data_in_leaf = 0;
    cfg.tree_config.min_child_hess   = 0.0F;
    cfg.tree_config.max_depth        = 3;
    cfg.booster_config.learning_rate = 0.3F;
    return cfg;
}

// Model bytes, through the on-disk format, so "same model" is literal.
std::vector<uint8_t> saved_bytes(IBooster const &booster, BinMappers const &mappers,
                                 Config const &cfg, std::string const &name)
{
    auto const path = std::filesystem::temp_directory_path() / name;
    io::save_booster(booster, path.string(), mappers, cfg);
    std::ifstream        in(path, std::ios::binary);
    std::vector<uint8_t> bytes{std::istreambuf_iterator<char>(in),
                               std::istreambuf_iterator<char>()};
    in.close();
    std::error_code ec;
    std::filesystem::remove(path, ec);
    return bytes;
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
    Dataset const    ds      = Dataset::bin(batch, mappers, {});

    REQUIRE(ds.n_features() == 2);
    REQUIRE(ds.n_rows() == 4);
    for (size_t f = 0; f < ds.n_features(); ++f)
    {
        REQUIRE(ds.visit_bins(f, [](auto bins) { return bins.size(); }) == ds.n_rows());
        for (size_t r = 0; r < ds.n_rows(); ++r)
        {
            CHECK(ds.bin_at(f, r) == ds.mappers()[f].transform(batch.features[f][r]));
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
    Dataset const    ds      = Dataset::bin(batch, mappers, {});

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
    Dataset const    ds      = Dataset::bin(batch, mappers, {});

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
    Dataset const    ds      = Dataset::bin(batch, mappers, {});

    auto w = ds.weights();
    REQUIRE(w.size() == 4);
    CHECK(w[0] == 0.5F);
    CHECK(w[3] == 1.5F);
}

TEST_CASE("Dataset: NaN row routes to missing bin (last bin)", "[dataset][missing]")
{
    auto const          nan = std::numeric_limits<float>::quiet_NaN();
    detail::ColumnBatch batch{
        .features      = {{0.1F, 0.5F, nan, 0.9F}},
        .labels        = {0.0F, 0.0F, 0.0F, 0.0F},
        .weights       = {},
        .feature_names = {"a"},
    };
    BinMappers const mappers = BinMappers::fit(batch, BinMapperConfig{});
    Dataset const    ds      = Dataset::bin(batch, mappers, {});

    REQUIRE(ds.visit_bins(0, [](auto bins) { return bins.size(); }) == 4);
    CHECK(ds.bin_at(0, 2) == ds.n_bins(0) - 1);
}

TEST_CASE("row-major fit and bin match the ColumnBatch overloads exactly", "[dataset]")
{
    size_t const n = 999;
    size_t const f = 7;
    // Deterministic values with ties and a NaN sprinkled in.
    std::vector<float> row_major(n * f);
    for (size_t i = 0; i < row_major.size(); ++i)
    {
        row_major[i] = static_cast<float>((i * 2654435761U) % 10007) / 997.0F;
    }
    row_major[123] = std::numeric_limits<float>::quiet_NaN();

    detail::ColumnBatch batch;
    batch.features.assign(f, std::vector<float>(n));
    std::vector<float> labels(n);
    for (size_t r = 0; r < n; ++r)
    {
        labels[r] = static_cast<float>(r % 13);
        for (size_t c = 0; c < f; ++c)
        {
            batch.features[c][r] = row_major[(r * f) + c];
        }
    }
    batch.labels = labels;
    for (size_t c = 0; c < f; ++c)
    {
        batch.feature_names.push_back("f" + std::to_string(c));
    }

    BinMapperConfig cfg;
    auto const      view = features_view{row_major.data(), n, f};
    auto const      a    = BinMappers::fit(batch, cfg);
    auto const      b    = BinMappers::fit(
        view, {batch.feature_names.begin(), batch.feature_names.end()}, cfg);
    REQUIRE(a.size() == b.size());
    for (size_t c = 0; c < f; ++c)
    {
        auto const ca = a[c].cuts();
        auto const cb = b[c].cuts();
        REQUIRE(std::vector<float>(ca.begin(), ca.end()) ==
                std::vector<float>(cb.begin(), cb.end()));
    }

    DataConfig dc;
    auto const da = Dataset::bin(batch, a, dc);
    auto const db = Dataset::bin(view, labels, a, dc);
    REQUIRE(da.n_rows() == db.n_rows());
    for (size_t c = 0; c < f; ++c)
    {
        REQUIRE(da.bins_are_u8() == db.bins_are_u8());
        for (size_t r = 0; r < da.n_rows(); ++r)
        {
            REQUIRE(da.bin_at(c, r) == db.bin_at(c, r));
        }
    }
}

TEST_CASE("Dataset: from_bins mints its own lazy caches", "[dataset][from_bins]")
{
    auto const       batch   = two_feature_batch(16);
    BinMappers const mappers = BinMappers::fit(batch, BinMapperConfig{});
    Dataset const    parent  = Dataset::bin(batch, mappers, {});
    REQUIRE(parent.bins_are_u8());

    // Mint the parent's mirror first: a shared row_major_ would already hold
    // the parent's bins by the time the child asks for its own.
    auto const parent_mirror = parent.row_major_bins();
    REQUIRE(parent_mirror.size() == parent.n_rows() * parent.n_features());

    Bins shifted = bins_of(parent);
    for (size_t f = 0; f < parent.n_features(); ++f)
    {
        for (auto &bin : shifted.u8[f])
        {
            bin = static_cast<uint8_t>((bin + 1) % parent.n_bins(f));
        }
    }
    Dataset const child =
        Dataset::from_bins(shifted.u8, {}, true, mappers, batch.labels);
    auto const child_mirror = child.row_major_bins();
    REQUIRE(child_mirror.size() == parent_mirror.size());

    Bins const    same = bins_of(parent);
    Dataset const twin = Dataset::from_bins(same.u8, {}, true, mappers, batch.labels);
    auto const    twin_mirror = twin.row_major_bins();
    REQUIRE(twin_mirror.size() == parent_mirror.size());

    for (size_t f = 0; f < parent.n_features(); ++f)
    {
        for (size_t r = 0; r < parent.n_rows(); ++r)
        {
            auto const want =
                static_cast<bin_id_t>((parent.bin_at(f, r) + 1) % parent.n_bins(f));
            CHECK(child.bin_at(f, r) == want);
            CHECK(child_mirror[child.mirror_index(r, f)] == want);
            // The parent keeps its own, and a second child gets its own.
            CHECK(parent_mirror[parent.mirror_index(r, f)] == parent.bin_at(f, r));
            CHECK(twin.bin_at(f, r) == parent.bin_at(f, r));
            CHECK(twin_mirror[twin.mirror_index(r, f)] == parent.bin_at(f, r));
        }
    }
}

TEST_CASE("Dataset: from_bins returns the u8 bins it was handed",
          "[dataset][from_bins]")
{
    auto const       batch   = two_feature_batch(64);
    BinMappers const mappers = BinMappers::fit(batch, BinMapperConfig{.max_bin = 8});
    Dataset const    parent  = Dataset::bin(batch, mappers, {});
    REQUIRE(parent.bins_are_u8());

    Bins const    bins = bins_of(parent);
    Dataset const ds   = Dataset::from_bins(bins.u8, {}, true, mappers, batch.labels);

    CHECK(ds.bins_are_u8());
    CHECK(ds.n_rows() == parent.n_rows());
    CHECK(ds.n_features() == parent.n_features());
    CHECK(ds.view_n_rows() == parent.n_rows());
    for (size_t f = 0; f < ds.n_features(); ++f)
    {
        REQUIRE(ds.visit_bins(f, [](auto col) { return col.size(); }) == ds.n_rows());
        for (size_t r = 0; r < ds.n_rows(); ++r)
        {
            REQUIRE(ds.bin_at(f, r) == parent.bin_at(f, r));
            REQUIRE(ds.visit_bins(f, [r](auto col) { return bin_id_t{col[r]}; }) ==
                    parent.bin_at(f, r));
        }
    }
}

TEST_CASE("Dataset: from_bins returns the u16 bins it was handed",
          "[dataset][from_bins]")
{
    auto const       batch   = two_feature_batch(600);
    BinMappers const mappers = BinMappers::fit(batch, BinMapperConfig{.max_bin = 2048});
    Dataset const    parent  = Dataset::bin(batch, mappers, {});
    REQUIRE_FALSE(parent.bins_are_u8());

    Bins const    bins = bins_of(parent);
    Dataset const ds   = Dataset::from_bins({}, bins.u16, false, mappers, batch.labels);

    CHECK_FALSE(ds.bins_are_u8());
    CHECK(ds.n_rows() == parent.n_rows());
    CHECK(ds.n_features() == parent.n_features());
    // u16 bins have no row-major mirror, on either dataset.
    CHECK(ds.row_major_bins().empty());
    for (size_t f = 0; f < ds.n_features(); ++f)
    {
        REQUIRE(ds.visit_bins(f, [](auto col) { return col.size(); }) == ds.n_rows());
        for (size_t r = 0; r < ds.n_rows(); ++r)
        {
            REQUIRE(ds.bin_at(f, r) == parent.bin_at(f, r));
            REQUIRE(ds.visit_bins(f, [r](auto col) { return bin_id_t{col[r]}; }) ==
                    parent.bin_at(f, r));
        }
    }
}

TEST_CASE("Dataset: from_bins reads its cuts off the mappers it was given",
          "[dataset][from_bins]")
{
    auto const       batch   = two_feature_batch(32);
    BinMappers const mappers = BinMappers::fit(batch, BinMapperConfig{});
    Dataset const    parent  = Dataset::bin(batch, mappers, {});

    Bins const       bins = bins_of(parent);
    BinMappers const kept =
        BinMappers::from_mappers(std::vector<BinMapper>{mappers[1]}, {"b"});
    Dataset const ds = Dataset::from_bins(std::vector<std::vector<uint8_t>>{bins.u8[1]},
                                          {}, true, kept, batch.labels);

    CHECK(ds.n_features() == 1);
    CHECK(ds.n_bins(0) == parent.n_bins(1));
    auto const names = ds.mappers().feature_names();
    REQUIRE(names.size() == 1);
    CHECK(names[0] == "b");
    CHECK(std::vector<float>(ds.cuts(0).begin(), ds.cuts(0).end()) ==
          std::vector<float>(parent.cuts(1).begin(), parent.cuts(1).end()));
    CHECK(ds.mappers().same_cuts(kept));
    CHECK_FALSE(ds.mappers().same_cuts(mappers));
}

TEST_CASE("Dataset: from_bins carries labels and weights", "[dataset][from_bins]")
{
    detail::ColumnBatch batch{
        .features      = {linspace(0.0F, 1.0F, 4)},
        .labels        = {0.0F, 1.0F, 0.0F, 1.0F},
        .weights       = {0.5F, 1.0F, 2.0F, 1.5F},
        .feature_names = {"a"},
    };
    BinMappers const mappers = BinMappers::fit(batch, BinMapperConfig{});
    Bins const       bins    = bins_of(Dataset::bin(batch, mappers, {}));
    Dataset const    ds =
        Dataset::from_bins(bins.u8, {}, true, mappers, batch.labels, batch.weights);

    REQUIRE(ds.labels().size() == 4);
    CHECK(ds.labels()[1] == 1.0F);
    REQUIRE(ds.weights().size() == 4);
    CHECK(ds.weights()[0] == 0.5F);
    CHECK(ds.weights()[3] == 1.5F);
    CHECK(ds.n_rows() == 4);
    CHECK(ds.n_features() == 1);

    // No weights is the uniform case, as every other factory reports it.
    Dataset const unweighted =
        Dataset::from_bins(bins.u8, {}, true, mappers, batch.labels);
    CHECK(unweighted.weights().empty());
}

TEST_CASE("Dataset: from_bins refuses columns that do not describe its rows",
          "[dataset][from_bins]")
{
    auto const       batch   = two_feature_batch(16);
    BinMappers const mappers = BinMappers::fit(batch, BinMapperConfig{});
    Bins const       bins    = bins_of(Dataset::bin(batch, mappers, {}));

    SECTION("a column shorter than the label vector")
    {
        auto short_bins = bins.u8;
        short_bins[1].pop_back();
        CHECK_THROWS_AS(Dataset::from_bins(short_bins, {}, true, mappers, batch.labels),
                        std::invalid_argument);
        CHECK_THROWS_WITH(
            Dataset::from_bins(short_bins, {}, true, mappers, batch.labels),
            Catch::Matchers::ContainsSubstring("column 1") &&
                Catch::Matchers::ContainsSubstring("15") &&
                Catch::Matchers::ContainsSubstring("16 rows"));
    }

    SECTION("fewer mappers than columns")
    {
        BinMappers const one =
            BinMappers::from_mappers(std::vector<BinMapper>{mappers[0]}, {"a"});
        CHECK_THROWS_AS(Dataset::from_bins(bins.u8, {}, true, one, batch.labels),
                        std::invalid_argument);
        CHECK_THROWS_WITH(Dataset::from_bins(bins.u8, {}, true, one, batch.labels),
                          Catch::Matchers::ContainsSubstring("1 bin mapper") &&
                              Catch::Matchers::ContainsSubstring("2 binned columns"));
    }

    SECTION("no columns of the declared width")
    {
        CHECK_THROWS_AS(Dataset::from_bins({}, {}, true, mappers, batch.labels),
                        std::invalid_argument);
        CHECK_THROWS_WITH(Dataset::from_bins({}, {}, true, mappers, batch.labels),
                          Catch::Matchers::ContainsSubstring("no binned columns"));
        // u16 bins under a u8 declaration are no columns at all.
        CHECK_THROWS_WITH(
            Dataset::from_bins({}, bins_of(Dataset::bin(batch, mappers, {})).u16, true,
                               mappers, batch.labels),
            Catch::Matchers::ContainsSubstring("no binned columns"));
    }
}

TEST_CASE("Dataset: a from_bins dataset trains to the same model bytes",
          "[dataset][from_bins]")
{
    auto const       batch   = two_feature_batch(256);
    BinMappers const mappers = BinMappers::fit(batch, BinMapperConfig{});
    Dataset const    raw     = Dataset::bin(batch, mappers, {});
    Bins const       bins    = bins_of(raw);
    Dataset const    rebound =
        Dataset::from_bins(bins.u8, {}, true, mappers, batch.labels);

    Config const cfg          = from_bins_cfg();
    auto         from_raw     = make_booster(cfg);
    auto         from_rebound = make_booster(cfg);
    for (int i = 0; i < 5; ++i)
    {
        from_raw->update_one_iter(raw);
        from_rebound->update_one_iter(rebound);
    }

    CHECK(saved_bytes(*from_raw, mappers, cfg, "bonsai_from_bins_raw.msgpack") ==
          saved_bytes(*from_rebound, mappers, cfg, "bonsai_from_bins_rebound.msgpack"));
}

namespace
{

// Four features whose BINS differ row for row, which two monotone ramps
// would not: any strictly increasing column bins to the same ids as any
// other, so a gathered column that came from the wrong place would compare
// equal to the right one and every value check would pass vacuously.
detail::ColumnBatch four_feature_batch(size_t n)
{
    std::vector<float> labels;
    std::vector<float> down;
    std::vector<float> saw;
    std::vector<float> fold;
    labels.reserve(n);
    down.reserve(n);
    saw.reserve(n);
    fold.reserve(n);
    for (size_t i = 0; i < n; ++i)
    {
        auto const x = static_cast<float>(i);
        labels.push_back(static_cast<float>(i % 5));
        down.push_back(-x);                                 // reversed order
        saw.push_back(static_cast<float>(i % 7));           // repeats, 7 levels
        fold.push_back(std::abs(x - (static_cast<float>(n) / 2.0F))); // v-shaped
    }
    return detail::ColumnBatch{
        .features      = {linspace(0.0F, 1.0F, n), std::move(down), std::move(saw),
                          std::move(fold)},
        .labels        = std::move(labels),
        .weights       = {},
        .feature_names = {"a", "b", "c", "d"},
    };
}

} // namespace

TEST_CASE("Dataset: select_features gathers the kept columns, renumbered",
          "[dataset][select_features]")
{
    auto const       batch   = four_feature_batch(64);
    BinMappers const mappers = BinMappers::fit(batch, BinMapperConfig{});
    Dataset const    full    = Dataset::bin(batch, mappers, {});

    std::vector<feature_id_t> const keep{2, 0};
    Dataset const                   sub = full.select_features(keep);

    REQUIRE(sub.n_features() == 2);
    CHECK(sub.n_rows() == full.n_rows());
    CHECK(sub.view_n_rows() == full.n_rows());
    // Names travel with the columns and land in the order asked for.
    REQUIRE(sub.mappers().feature_names().size() == 2);
    CHECK(sub.mappers().feature_names()[0] == "c");
    CHECK(sub.mappers().feature_names()[1] == "a");
    // Feature k of the subset holds the bins of feature keep[k] of the parent.
    for (size_t k = 0; k < keep.size(); ++k)
    {
        for (size_t r = 0; r < full.n_rows(); ++r)
        {
            CHECK(sub.bin_at(k, r) == full.bin_at(keep[k], r));
        }
        CHECK(sub.n_bins(k) == full.n_bins(keep[k]));
    }
    // The two kept columns disagree somewhere, so the checks above are a
    // claim about which column landed where and not about all columns
    // happening to bin alike.
    bool differs = false;
    for (size_t r = 0; r < sub.n_rows(); ++r)
    {
        differs |= sub.bin_at(0, r) != sub.bin_at(1, r);
    }
    CHECK(differs);
}

TEST_CASE("Dataset: a gathered column equals one binned from its own matrix",
          "[dataset][select_features]")
{
    auto const       batch   = four_feature_batch(128);
    BinMappers const mappers = BinMappers::fit(batch, BinMapperConfig{});
    Dataset const    full    = Dataset::bin(batch, mappers, {});

    // The same two columns taken the long way: a matrix of just those
    // features, binned under the cuts the parent already fit. Identical bins
    // are the claim that gathering does not disturb the binning.
    detail::ColumnBatch const two{
        .features      = {batch.features[3], batch.features[1]},
        .labels        = batch.labels,
        .weights       = {},
        .feature_names = {"d", "b"},
    };
    BinMappers const two_mappers = BinMappers::from_mappers(
        std::vector<BinMapper>{mappers[3], mappers[1]}, {"d", "b"});
    Dataset const direct = Dataset::bin(two, two_mappers, {});

    std::vector<feature_id_t> const keep{3, 1};
    Dataset const                   sub = full.select_features(keep);

    Config const cfg         = from_bins_cfg();
    auto         from_sub    = make_booster(cfg);
    auto         from_direct = make_booster(cfg);
    for (int i = 0; i < 5; ++i)
    {
        from_sub->update_one_iter(sub);
        from_direct->update_one_iter(direct);
    }
    CHECK(saved_bytes(*from_sub, two_mappers, cfg, "bonsai_selfeat_sub.msgpack") ==
          saved_bytes(*from_direct, two_mappers, cfg, "bonsai_selfeat_direct.msgpack"));
}

TEST_CASE("Dataset: keeping every feature in order is the same dataset",
          "[dataset][select_features]")
{
    auto const       batch   = four_feature_batch(96);
    BinMappers const mappers = BinMappers::fit(batch, BinMapperConfig{});
    Dataset const    full    = Dataset::bin(batch, mappers, {});
    Dataset const    same    = full.select_features(std::vector<feature_id_t>{0, 1, 2, 3});

    Config const cfg       = from_bins_cfg();
    auto         from_full = make_booster(cfg);
    auto         from_same = make_booster(cfg);
    for (int i = 0; i < 5; ++i)
    {
        from_full->update_one_iter(full);
        from_same->update_one_iter(same);
    }
    CHECK(saved_bytes(*from_full, mappers, cfg, "bonsai_selfeat_all_full.msgpack") ==
          saved_bytes(*from_same, mappers, cfg, "bonsai_selfeat_all_same.msgpack"));
}

TEST_CASE("Dataset: a column selection serves its own bins, not the parent's",
          "[dataset][select_features][aliasing]")
{
    auto const       batch   = four_feature_batch(64);
    BinMappers const mappers = BinMappers::fit(batch, BinMapperConfig{});
    Dataset const    full    = Dataset::bin(batch, mappers, {});
    Dataset const    sub     = full.select_features(std::vector<feature_id_t>{2});

    // cols_ and row_major_ are shared_ptr members shared across Dataset
    // copies by design. A selection built by copying the parent would serve
    // the parent's four columns here, and feature 0 would read "a" not "c".
    REQUIRE(sub.n_features() == 1);
    bool differs = false;
    for (size_t r = 0; r < full.n_rows(); ++r)
    {
        CHECK(sub.bin_at(0, r) == full.bin_at(2, r));
        differs |= full.bin_at(0, r) != full.bin_at(2, r);
    }
    CHECK(differs); // the two columns really do disagree somewhere

    // The row-major mirror is minted per dataset and must be the subset's
    // own: one feature wide, not the parent's four.
    auto const mirror = sub.row_major_bins();
    REQUIRE(!mirror.empty());
    for (size_t r = 0; r < sub.n_rows(); ++r)
    {
        CHECK(mirror[sub.mirror_index(r, 0)] == full.bin_at(2, r));
    }
}

TEST_CASE("Dataset: a column rewrite spends the row view instead of carrying it",
          "[dataset][select_features][views]")
{
    auto const       batch   = four_feature_batch(64);
    BinMappers const mappers = BinMappers::fit(batch, BinMapperConfig{});
    Dataset const    full    = Dataset::bin(batch, mappers, {});

    std::vector<row_id_t> const     rows{10, 21, 22, 23, 40, 55};
    std::vector<feature_id_t> const keep{1, 3};
    Dataset const view = full.with_rows(RowView::encode(rows, full.n_rows()));

    // The rewrite mints a plane whatever the rows are, so it gathers only the
    // ones the view names: the result holds them numbered from zero and views
    // nothing.
    Dataset const rows_first = view.select_features(keep);
    REQUIRE(rows_first.n_rows() == rows.size());
    CHECK(rows_first.view_n_rows() == rows.size());
    CHECK(rows_first.row_view().is_identity());
    for (size_t k = 0; k < keep.size(); ++k)
    {
        for (size_t i = 0; i < rows.size(); ++i)
        {
            CHECK(rows_first.bin_at(k, i) == full.bin_at(keep[k], rows[i]));
        }
    }
    // Labels follow the same rows, or the objective would score the wrong
    // ones with no error anywhere.
    for (size_t i = 0; i < rows.size(); ++i)
    {
        CHECK(rows_first.labels()[i] == full.labels()[rows[i]]);
    }

    // The other order views the rewrite instead of materializing, and the two
    // must train the same model: same rows, same columns, same bins.
    Dataset const cols_first =
        full.select_features(keep).with_rows(RowView::encode(rows, full.n_rows()));
    CHECK(cols_first.view_n_rows() == rows.size());
    CHECK(cols_first.n_rows() == full.n_rows()); // a view keeps the plane whole

    BinMappers const kept = BinMappers::from_mappers(
        std::vector<BinMapper>{mappers[1], mappers[3]}, {"b", "d"});
    Config const cfg      = from_bins_cfg();
    auto         from_rf  = make_booster(cfg);
    auto         from_cf  = make_booster(cfg);
    for (int i = 0; i < 5; ++i)
    {
        from_rf->update_one_iter(rows_first);
        from_cf->update_one_iter(cols_first);
    }
    CHECK(saved_bytes(*from_rf, kept, cfg, "bonsai_selfeat_rows_first.msgpack") ==
          saved_bytes(*from_cf, kept, cfg, "bonsai_selfeat_cols_first.msgpack"));
}

TEST_CASE("Dataset: select_features refuses a selection it cannot serve",
          "[dataset][select_features][error]")
{
    auto const       batch   = four_feature_batch(32);
    BinMappers const mappers = BinMappers::fit(batch, BinMapperConfig{});
    Dataset const    full    = Dataset::bin(batch, mappers, {});

    SECTION("no features at all")
    {
        CHECK_THROWS_AS(full.select_features(std::vector<feature_id_t>{}),
                        std::invalid_argument);
        CHECK_THROWS_WITH(full.select_features(std::vector<feature_id_t>{}),
                          Catch::Matchers::ContainsSubstring("no features"));
    }

    SECTION("a feature past the last one")
    {
        CHECK_THROWS_AS(full.select_features(std::vector<feature_id_t>{0, 4}),
                        std::invalid_argument);
        CHECK_THROWS_WITH(full.select_features(std::vector<feature_id_t>{0, 4}),
                          Catch::Matchers::ContainsSubstring("feature 4") &&
                              Catch::Matchers::ContainsSubstring("4 features"));
    }
}
