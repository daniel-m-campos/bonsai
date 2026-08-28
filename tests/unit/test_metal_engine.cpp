// Compiled in every build; every case SKIPs at runtime unless
// metal_available(). Metal cells are fixed-point within a chunk and merge
// through float device atomics, so parity with the CPU engine is
// tolerance-based, normalized per feature by the largest |cell sum|: the
// scale a split decision actually compares against.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <random>
#include <vector>

#include "bonsai/dataset.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/metal/histogram_engine.hpp"
#include "bonsai/split.hpp"
#include "bonsai/types.hpp"

using namespace bonsai; // NOLINT

namespace
{

constexpr double k_cell_tolerance = 1e-3;

struct Fixture
{
    Dataset                   ds;
    std::vector<float>        grad, hess;
    std::vector<feature_id_t> selected;
};

Fixture make_fixture(size_t n_rows, size_t n_features, BinMapperConfig const &cfg = {})
{
    std::mt19937                          rng(7);
    std::uniform_real_distribution<float> df(-1.0F, 1.0F);
    detail::ColumnBatch                   batch;
    batch.features.resize(n_features);
    for (auto &col : batch.features)
    {
        col.resize(n_rows);
        for (auto &v : col)
        {
            v = df(rng);
        }
    }
    batch.labels.resize(n_rows);
    for (auto &v : batch.labels)
    {
        v = df(rng);
    }
    batch.feature_names.assign(n_features, "f");

    BinMappers mappers = BinMappers::fit(batch, cfg);
    Fixture    fx{
           .ds = Dataset::bin(batch, mappers, {}), .grad = {}, .hess = {}, .selected = {}};
    fx.grad.resize(n_rows);
    fx.hess.resize(n_rows);
    for (size_t i = 0; i < n_rows; ++i)
    {
        fx.grad[i] = df(rng);
        fx.hess[i] = df(rng) + 2.0F;
    }
    for (feature_id_t f = 0; f < n_features; ++f)
    {
        fx.selected.push_back(f);
    }
    return fx;
}

SplitInput cpu_reference(Fixture const &fx, std::vector<row_id_t> rows, bool identity)
{
    SplitInput node;
    node.rows           = std::move(rows);
    node.shape.identity = identity;
    CpuHistogramEngine engine;
    engine.begin_tree(fx.ds, fx.grad, fx.hess);
    engine.populate(fx.ds, fx.grad, fx.hess, node, fx.selected);
    return node;
}

SplitInput metal_node(Fixture const &fx, floats_view hess, std::vector<row_id_t> rows,
                      bool identity)
{
    SplitInput node;
    node.rows           = std::move(rows);
    node.shape.identity = identity;
    MetalHistogramEngine engine;
    engine.begin_tree(fx.ds, fx.grad, hess);
    engine.populate(fx.ds, fx.grad, hess, node, fx.selected);
    return node;
}

double max_relative_cell_error(Fixture const &fx, SplitInput const &got,
                               SplitInput const &want)
{
    double worst = 0.0;
    for (feature_id_t const fid : fx.selected)
    {
        auto const g     = got.hists[fid].all_cells();
        auto const w     = want.hists[fid].all_cells();
        REQUIRE(g.size() == w.size());
        double scale_g = 0.0, scale_h = 0.0, err_g = 0.0, err_h = 0.0;
        for (size_t b = 0; b < w.size(); ++b)
        {
            scale_g = std::max(scale_g, std::fabs(double{w[b].sum_grad}));
            scale_h = std::max(scale_h, std::fabs(double{w[b].sum_hess}));
            err_g   = std::max(err_g,
                               std::fabs(double{g[b].sum_grad} - double{w[b].sum_grad}));
            err_h   = std::max(err_h,
                               std::fabs(double{g[b].sum_hess} - double{w[b].sum_hess}));
        }
        worst = std::max({worst, err_g / std::max(1e-30, scale_g),
                          err_h / std::max(1e-30, scale_h)});
    }
    return worst;
}

std::vector<row_id_t> iota_rows(size_t n)
{
    std::vector<row_id_t> rows(n);
    for (size_t i = 0; i < n; ++i)
    {
        rows[i] = static_cast<row_id_t>(i);
    }
    return rows;
}

} // namespace

TEST_CASE("MetalEngine: root populate matches the CPU engine cell for cell",
          "[metal][fit]")
{
    if (!metal_available())
    {
        SKIP("no usable Metal device");
    }
    Fixture const fx = make_fixture(20000, 24);

    SplitInput const want = cpu_reference(fx, iota_rows(20000), true);
    SplitInput const got  = metal_node(fx, fx.hess, iota_rows(20000), true);
    CHECK(max_relative_cell_error(fx, got, want) < k_cell_tolerance);
}

TEST_CASE("MetalEngine: subset rows gather through the row list", "[metal][fit]")
{
    if (!metal_available())
    {
        SKIP("no usable Metal device");
    }
    Fixture const fx = make_fixture(20000, 24);

    std::vector<row_id_t> rows;
    for (size_t i = 0; i < 20000; i += 3)
    {
        rows.push_back(static_cast<row_id_t>(i));
    }
    SplitInput const want = cpu_reference(fx, rows, false);
    SplitInput const got  = metal_node(fx, fx.hess, rows, false);
    CHECK(max_relative_cell_error(fx, got, want) < k_cell_tolerance);
}

TEST_CASE("MetalEngine: empty hessian counts rows like the CPU engine",
          "[metal][fit][edge]")
{
    if (!metal_available())
    {
        SKIP("no usable Metal device");
    }
    Fixture const fx = make_fixture(10000, 8);

    SplitInput want;
    want.rows           = iota_rows(10000);
    want.shape.identity = true;
    {
        CpuHistogramEngine engine;
        engine.begin_tree(fx.ds, fx.grad, {});
        engine.populate(fx.ds, fx.grad, {}, want, fx.selected);
    }
    SplitInput const got = metal_node(fx, {}, iota_rows(10000), true);
    CHECK(max_relative_cell_error(fx, got, want) < k_cell_tolerance);
}

TEST_CASE("MetalEngine: populate_many fills a level of nodes in one dispatch",
          "[metal][fit]")
{
    if (!metal_available())
    {
        SKIP("no usable Metal device");
    }
    Fixture const fx = make_fixture(30000, 16);

    std::vector<row_id_t> left, right;
    for (size_t i = 0; i < 30000; ++i)
    {
        (i % 2 == 0 ? left : right).push_back(static_cast<row_id_t>(i));
    }
    SplitInput const want_left  = cpu_reference(fx, left, false);
    SplitInput const want_right = cpu_reference(fx, right, false);

    SplitInput got_left, got_right;
    got_left.rows  = left;
    got_right.rows = right;
    std::array nodes = {std::ref(got_left), std::ref(got_right)};
    MetalHistogramEngine engine;
    engine.begin_tree(fx.ds, fx.grad, fx.hess);
    engine.populate_many(fx.ds, fx.grad, fx.hess, nodes, fx.selected);

    CHECK(max_relative_cell_error(fx, got_left, want_left) < k_cell_tolerance);
    CHECK(max_relative_cell_error(fx, got_right, want_right) < k_cell_tolerance);
}

TEST_CASE("MetalEngine: u16 bin plane routes through the wide kernel",
          "[metal][fit][edge]")
{
    if (!metal_available())
    {
        SKIP("no usable Metal device");
    }
    BinMapperConfig wide;
    wide.max_bin     = 500;
    Fixture const fx = make_fixture(20000, 6, wide);
    REQUIRE_FALSE(fx.ds.bins_are_u8());

    SplitInput const want = cpu_reference(fx, iota_rows(20000), true);
    SplitInput const got  = metal_node(fx, fx.hess, iota_rows(20000), true);
    CHECK(max_relative_cell_error(fx, got, want) < k_cell_tolerance);
}
