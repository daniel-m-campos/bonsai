#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <random>
#include <utility>
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

// The level plane's fill of one node, or — for a nonzero id, as the leaf
// plane drives it — the fused lone fill against a sibling holding the
// parent's histograms.
SplitInput populate_node(Fixture const &fx, std::vector<row_id_t> rows,
                         node_id_t id = 0)
{
    SplitInput node;
    node.id             = id;
    node.rows           = std::move(rows);
    node.shape.identity = rows_are_identity(node.rows, fx.ds.n_rows());
    CpuHistogramEngine engine;
    // As a grower drives it: begin_tree drops the cached selection plan, which
    // is keyed by addresses a second fixture in this process may reuse.
    engine.begin_tree(fx.ds, fx.grad, fx.hess);
    if (id == 0)
    {
        engine.populate(fx.ds, fx.grad, fx.hess, node, fx.selected);
        return node;
    }
    SplitInput parent;
    parent.rows           = test::iota_rows(fx.ds.n_rows());
    parent.shape.identity = true;
    engine.populate(fx.ds, fx.grad, fx.hess, parent, fx.selected);
    engine.populate_lone(fx.ds, fx.grad, fx.hess, node, fx.selected, parent.hists);
    return node;
}

// Block-aware mirror lookup: feature f lives in block f / width at column
// position f % width; one block reproduces the classic row-major layout.
void check_mirror_layout(Fixture const &fx)
{
    auto const   rm    = fx.ds.mirror().bins();
    size_t const width = Dataset::mirror_tile_width();
    REQUIRE(rm.size() == fx.ds.n_rows() * fx.ds.n_features());
    for (size_t r = 0; r < fx.ds.n_rows(); ++r)
    {
        for (size_t f = 0; f < fx.ds.n_features(); ++f)
        {
            size_t const mb      = f / width;
            size_t const width_b = std::min(width, fx.ds.n_features() - (mb * width));
            size_t const idx =
                (fx.ds.n_rows() * mb * width) + (r * width_b) + (f - (mb * width));
            REQUIRE(rm[idx] == fx.ds.bin_at(f, r));
        }
    }
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
            // Blocked f32 accumulation vs the serial f32 order: rounding
            // differs by O(cell_sum * eps); real fill bugs miss whole rows.
            CHECK(cells[b].sum_grad == Catch::Approx(ref[s][b].sum_grad).margin(1e-2));
            CHECK(cells[b].sum_hess == Catch::Approx(ref[s][b].sum_hess).margin(1e-2));
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

TEST_CASE("sparse fill sums its per-thread partials into the node arena", "[populate]")
{
    // The partial-and-reduce arm: 8192 of 40960 rows clears the density gate
    // (rows * 4 < n_rows) so the node routes to the sparse fill, and 8 row
    // chunks of grain 1024 spread over 4 threads, so three of the four
    // write partials that the reduce pass must sum back in.
    auto const fx = make_fixture(40960, 3);
    REQUIRE(fx.ds.bins_are_u8());

    std::vector<row_id_t> rows;
    for (row_id_t r = 0; r < fx.ds.n_rows(); r += 5)
    {
        rows.push_back(r);
    }
    REQUIRE(rows.size() == 8192);
    REQUIRE(rows.size() * 4 < fx.ds.n_rows());

    parallel::set_n_threads(1);
    auto const serial = populate_node(fx, rows); // one chunk range, no partials
    parallel::set_n_threads(4);
    auto const shared = populate_node(fx, rows); // partials plus reduce
    auto const ref    = reference_hists(fx, rows);

    for (size_t s = 0; s < fx.selected.size(); ++s)
    {
        auto const cells = shared.hists[fx.selected[s]].all_cells();
        auto const one   = serial.hists[fx.selected[s]].all_cells();
        REQUIRE(cells.size() == ref[s].size());
        for (size_t b = 0; b < cells.size(); ++b)
        {
            // A dropped or double-counted partial loses or repeats whole
            // chunks of rows; only reassociation rounding is in tolerance.
            CHECK(cells[b].sum_grad == Catch::Approx(one[b].sum_grad).margin(1e-2));
            CHECK(cells[b].sum_hess == Catch::Approx(one[b].sum_hess).margin(1e-2));
            CHECK(cells[b].sum_grad == Catch::Approx(ref[s][b].sum_grad).margin(1e-2));
            CHECK(cells[b].sum_hess == Catch::Approx(ref[s][b].sum_hess).margin(1e-2));
        }
    }
    parallel::set_n_threads(0);
}

TEST_CASE("a lone node fills feature-major in serial order in one row block",
          "[populate]")
{
    // The leaf plane's per-split fill: a node short enough to fill in one row
    // block is cut by feature only, where one worker owns a whole feature's
    // cells, so they sum in ascending row order at any thread count.
    auto const fx = make_fixture(40960, 3);
    REQUIRE(fx.ds.bins_are_u8());

    std::vector<row_id_t> rows;
    for (row_id_t r = 0; r < fx.ds.n_rows(); r += 5)
    {
        rows.push_back(r);
    }
    REQUIRE(rows.size() * 4 < fx.ds.n_rows());

    parallel::set_n_threads(1);
    auto const one = populate_node(fx, rows, 1);
    parallel::set_n_threads(4);
    auto const four = populate_node(fx, rows, 1);
    auto const ref  = reference_hists(fx, rows);

    for (size_t s = 0; s < fx.selected.size(); ++s)
    {
        auto const cells = four.hists[fx.selected[s]].all_cells();
        auto const c1    = one.hists[fx.selected[s]].all_cells();
        REQUIRE(cells.size() == ref[s].size());
        REQUIRE(std::memcmp(cells.data(), ref[s].data(),
                            cells.size() * sizeof(HistCell)) == 0);
        REQUIRE(std::memcmp(cells.data(), c1.data(), cells.size() * sizeof(HistCell)) ==
                0);
    }
    parallel::set_n_threads(0);
}

TEST_CASE("a lone node cut into row blocks is reproducible and serial at one "
          "thread",
          "[populate]")
{
    // A long, narrow node spends its team on row blocks, which regroups the
    // per-cell sums: the bytes are keyed to the thread count (the fixed-N
    // contract), not invariant across it. What holds is that one thread
    // reproduces the serial order exactly, a fixed count repeats itself
    // exactly, and every count lands on the same sums within float
    // tolerance.
    auto const fx = make_fixture(262144, 8);
    REQUIRE(fx.ds.bins_are_u8());

    std::vector<row_id_t> rows;
    for (row_id_t r = 0; r < fx.ds.n_rows(); r += 8)
    {
        rows.push_back(r);
    }
    REQUIRE(rows.size() * 4 < fx.ds.n_rows());

    parallel::set_n_threads(1);
    auto const one = populate_node(fx, rows, 1);
    parallel::set_n_threads(4);
    auto const four  = populate_node(fx, rows, 1);
    auto const again = populate_node(fx, rows, 1);
    auto const ref   = reference_hists(fx, rows);

    for (size_t s = 0; s < fx.selected.size(); ++s)
    {
        auto const c1  = one.hists[fx.selected[s]].all_cells();
        auto const c4  = four.hists[fx.selected[s]].all_cells();
        auto const c4b = again.hists[fx.selected[s]].all_cells();
        REQUIRE(c4.size() == ref[s].size());
        // One thread is one row block, which is the serial order.
        REQUIRE(std::memcmp(c1.data(), ref[s].data(), c1.size() * sizeof(HistCell)) ==
                0);
        // Four threads repeat themselves byte for byte.
        REQUIRE(std::memcmp(c4.data(), c4b.data(), c4.size() * sizeof(HistCell)) == 0);
        for (size_t b = 0; b < c4.size(); ++b)
        {
            CHECK(c4[b].sum_grad == Catch::Approx(ref[s][b].sum_grad).margin(1e-2));
            CHECK(c4[b].sum_hess == Catch::Approx(ref[s][b].sum_hess).margin(1e-2));
        }
    }
    parallel::set_n_threads(0);
}

TEST_CASE("the column fill gathers a lone node in serial order", "[populate]")
{
    // A lone node above the density cut takes the column fill, which reads
    // each feature's bins through the node's row list. Both sides of the
    // prefetch peel run: the wide node is longer than the lookahead, the
    // narrow one shorter than it.
    for (auto const [n_rows, stride] : {std::pair{8192UL, 2UL}, std::pair{128UL, 3UL}})
    {
        auto const fx = make_fixture(n_rows, 3);
        REQUIRE(fx.ds.bins_are_u8());

        std::vector<row_id_t> rows;
        for (row_id_t r = 1; r < fx.ds.n_rows(); r += stride)
        {
            rows.push_back(r);
        }
        REQUIRE(rows.size() * 4 >= fx.ds.n_rows()); // routes to the column fill
        REQUIRE(rows.size() < fx.ds.n_rows());      // and gathers, not in place

        auto const ref = reference_hists(fx, rows);
        for (int const threads : {1, 4})
        {
            parallel::set_n_threads(threads);
            auto const node = populate_node(fx, rows, 1);
            for (size_t s = 0; s < fx.selected.size(); ++s)
            {
                auto const cells = node.hists[fx.selected[s]].all_cells();
                REQUIRE(cells.size() == ref[s].size());
                REQUIRE(std::memcmp(cells.data(), ref[s].data(),
                                    cells.size() * sizeof(HistCell)) == 0);
            }
        }
        parallel::set_n_threads(0);
    }
}

TEST_CASE("a full-cardinality row list that is not the identity still gathers",
          "[populate]")
{
    // Cardinality does not imply identity. A permutation and a
    // with-replacement bootstrap both fill n slots without being [0, n), and
    // a fill that reads bins and grad at the row's POSITION instead of its id
    // returns the identity's histogram for either one: right by luck for the
    // permutation (same multiset of rows, wrong summation order), wrong for
    // the bootstrap (duplicated rows counted once, dropped rows counted at
    // all).
    parallel::set_n_threads(4);
    auto const fx = make_fixture(8192, 3);
    REQUIRE(fx.ds.bins_are_u8());

    std::mt19937 rng(11);
    auto         perm = test::iota_rows(fx.ds.n_rows());
    std::shuffle(perm.begin(), perm.end(), rng);

    std::vector<row_id_t>                   boot(fx.ds.n_rows());
    std::uniform_int_distribution<row_id_t> pick(
        0, static_cast<row_id_t>(fx.ds.n_rows() - 1));
    for (row_id_t &r : boot)
    {
        r = pick(rng);
    }

    for (auto const &rows : {perm, boot})
    {
        REQUIRE(rows.size() == fx.ds.n_rows());
        auto const node = populate_node(fx, rows);
        auto const ref  = reference_hists(fx, rows);
        for (size_t s = 0; s < fx.selected.size(); ++s)
        {
            auto const cells = node.hists[fx.selected[s]].all_cells();
            REQUIRE(cells.size() == ref[s].size());
            for (size_t b = 0; b < cells.size(); ++b)
            {
                // Values first: the bootstrap's are wrong by whole rows,
                // which no rounding tolerance covers.
                CHECK(cells[b].sum_grad ==
                      Catch::Approx(ref[s][b].sum_grad).margin(1e-2));
                CHECK(cells[b].sum_hess ==
                      Catch::Approx(ref[s][b].sum_hess).margin(1e-2));
            }
            // Then the column fill's serial-order contract: one worker owns a
            // feature and adds the node's rows in list order, so the cells
            // reproduce the reference byte for byte.
            CHECK(std::memcmp(cells.data(), ref[s].data(),
                              cells.size() * sizeof(HistCell)) == 0);
        }
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
    check_mirror_layout(fx);
}

TEST_CASE("row-major mirror matches the binned columns across mirror blocks",
          "[dataset]")
{
    // 2048 + 64 features: two mirror blocks, the second one partial.
    auto const fx = make_fixture(256, 2112);
    REQUIRE(fx.ds.bins_are_u8());
    REQUIRE(fx.ds.n_features() > Dataset::mirror_tile_width());
    check_mirror_layout(fx);
}

TEST_CASE("wide multi-slice fill matches serial sums within tolerance", "[populate]")
{
    parallel::set_n_threads(4);
    // 2048 + 64 features span two mirror slices, so the fill runs one pass
    // per slice; max_bin = 16 keeps the two slices' arenas small.
    auto const fx = make_fixture(4096, 2112, BinMapperConfig{.max_bin = 16});
    REQUIRE(fx.ds.bins_are_u8());
    REQUIRE(fx.ds.n_features() > Dataset::mirror_tile_width());

    auto const node = populate_node(fx, test::iota_rows(fx.ds.n_rows()));
    auto const ref  = reference_hists(fx, test::iota_rows(fx.ds.n_rows()));
    // Probe features from both blocks, straddling the boundary.
    for (size_t const s : {0UL, 1024UL, 2046UL, 2047UL, 2048UL, 2049UL, 2111UL})
    {
        auto const cells = node.hists[fx.selected[s]].all_cells();
        REQUIRE(cells.size() == ref[s].size());
        for (size_t b = 0; b < cells.size(); ++b)
        {
            CHECK(cells[b].sum_grad == Catch::Approx(ref[s][b].sum_grad).margin(1e-2));
            CHECK(cells[b].sum_hess == Catch::Approx(ref[s][b].sum_hess).margin(1e-2));
        }
    }
    parallel::set_n_threads(0);
}

TEST_CASE("wide single-block fill with a selection spanning the block boundary "
          "is bit-identical to the serial order",
          "[populate]")
{
    parallel::set_n_threads(4);
    auto fx = make_fixture(4096, 2112);
    REQUIRE(fx.ds.bins_are_u8());

    // colsample-style sparse selection straddling the mirror block boundary.
    fx.selected.clear();
    for (feature_id_t f = 5; f < fx.ds.n_features(); f += 89)
    {
        fx.selected.push_back(f);
    }
    REQUIRE(fx.selected.front() < Dataset::mirror_tile_width());
    REQUIRE(fx.selected.back() >= Dataset::mirror_tile_width());

    // A sparse row subset small enough for one block per slice.
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
