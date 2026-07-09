#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstring>
#include <random>
#include <vector>

#include "bonsai/dataset.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/parallel.hpp"
#include "bonsai/split.hpp"
#include "bonsai/types.hpp"
#include "test_grower_helpers.hpp"

using namespace bonsai; // NOLINT

namespace
{

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
        fx.hess[i] = df(rng) + 2.0F; // positive-ish, like a real hessian
    }
    for (feature_id_t f = 0; f < n_features; ++f)
    {
        fx.selected.push_back(f);
    }
    return fx;
}

// Serial reference: per feature, add rows in k order with double
// accumulation — the exact summation order of the feature-parallel fill.
std::vector<std::vector<HistCell>> reference_hists(Fixture const               &fx,
                                                   std::vector<row_id_t> const &rows)
{
    std::vector<std::vector<HistCell>> out(fx.selected.size());
    for (size_t s = 0; s < fx.selected.size(); ++s)
    {
        feature_id_t const fid = fx.selected[s];
        out[s].resize(fx.ds.n_bins(fid));
        for (row_id_t const r : rows)
        {
            HistCell &c = out[s][fx.ds.bin_at(fid, r)];
            c.sum_grad += fx.grad[r];
            c.sum_hess += fx.hess[r];
        }
    }
    return out;
}

SplitInput populate_node(Fixture const &fx, std::vector<row_id_t> rows)
{
    SplitInput node;
    node.rows = std::move(rows);
    CpuHistogramEngine engine;
    engine.populate(fx.ds, fx.grad, fx.hess, node, fx.selected);
    return node;
}

} // namespace

TEST_CASE("row-wise multi-block fill matches serial sums within tolerance",
          "[populate]")
{
    parallel::set_n_threads(4);
    auto const fx = make_fixture(40960, 3);
    REQUIRE(fx.ds.bins_are_u8());

    auto const node = populate_node(fx, test::iota_rows(fx.ds.n_rows()));
    auto const ref  = reference_hists(fx, test::iota_rows(fx.ds.n_rows()));
    for (size_t s = 0; s < fx.selected.size(); ++s)
    {
        auto const cells = node.hists[fx.selected[s]].all_cells();
        REQUIRE(cells.size() == ref[s].size());
        for (size_t b = 0; b < cells.size(); ++b)
        {
            CHECK(cells[b].sum_grad == Catch::Approx(ref[s][b].sum_grad).margin(1e-7));
            CHECK(cells[b].sum_hess == Catch::Approx(ref[s][b].sum_hess).margin(1e-7));
        }
    }
    parallel::set_n_threads(0);
}

TEST_CASE("row-wise single-block fill is bit-identical to the serial order",
          "[populate]")
{
    parallel::set_n_threads(4);
    auto const fx = make_fixture(8192, 3);
    REQUIRE(fx.ds.bins_are_u8());

    // A sparse, non-contiguous subset small enough for one block.
    std::vector<row_id_t> rows;
    for (row_id_t r = 1; r < fx.ds.n_rows(); r += 17)
    {
        rows.push_back(r);
    }
    auto const node = populate_node(fx, rows);
    auto const ref  = reference_hists(fx, rows);
    for (size_t s = 0; s < fx.selected.size(); ++s)
    {
        auto const cells = node.hists[fx.selected[s]].all_cells();
        REQUIRE(std::memcmp(cells.data(), ref[s].data(),
                            cells.size() * sizeof(HistCell)) == 0);
    }
    parallel::set_n_threads(0);
}

TEST_CASE("populate is reproducible at a fixed thread count", "[populate]")
{
    parallel::set_n_threads(3);
    auto const fx = make_fixture(40960, 3);

    auto const a = populate_node(fx, test::iota_rows(fx.ds.n_rows()));
    auto const b = populate_node(fx, test::iota_rows(fx.ds.n_rows()));
    for (feature_id_t const fid : fx.selected)
    {
        auto const ca = a.hists[fid].all_cells();
        auto const cb = b.hists[fid].all_cells();
        REQUIRE(std::memcmp(ca.data(), cb.data(), ca.size() * sizeof(HistCell)) == 0);
    }
    parallel::set_n_threads(0);
}

TEST_CASE("u16 fallback fill is bit-identical to the serial order", "[populate]")
{
    parallel::set_n_threads(4);
    auto const fx = make_fixture(8192, 3, BinMapperConfig{.max_bin = 2048});
    REQUIRE(!fx.ds.bins_are_u8());

    auto const node = populate_node(fx, test::iota_rows(fx.ds.n_rows()));
    auto const ref  = reference_hists(fx, test::iota_rows(fx.ds.n_rows()));
    for (size_t s = 0; s < fx.selected.size(); ++s)
    {
        auto const cells = node.hists[fx.selected[s]].all_cells();
        REQUIRE(std::memcmp(cells.data(), ref[s].data(),
                            cells.size() * sizeof(HistCell)) == 0);
    }
    parallel::set_n_threads(0);
}

TEST_CASE("row-major mirror matches the binned columns", "[dataset]")
{
    auto const fx = make_fixture(4096, 5);
    REQUIRE(fx.ds.bins_are_u8());
    auto const rm = fx.ds.row_major_bins();
    REQUIRE(rm.size() == fx.ds.n_rows() * fx.ds.n_features());
    for (size_t r = 0; r < fx.ds.n_rows(); ++r)
    {
        for (size_t f = 0; f < fx.ds.n_features(); ++f)
        {
            REQUIRE(rm[(r * fx.ds.n_features()) + f] == fx.ds.bin_at(f, r));
        }
    }
}
