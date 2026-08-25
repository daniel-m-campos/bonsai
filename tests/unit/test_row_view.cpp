#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bonsai/bin_mappers.hpp"
#include "bonsai/config/bin_mapper_config.hpp"
#include "bonsai/config/tree_config.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/detail/column_batch.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/row_view.hpp"
#include "bonsai/tree.hpp"
#include "bonsai/types.hpp"

using namespace bonsai; // NOLINT

namespace
{

std::vector<row_id_t> iota_rows(row_id_t begin, row_id_t end)
{
    std::vector<row_id_t> rows(static_cast<size_t>(end - begin));
    std::iota(rows.begin(), rows.end(), begin);
    return rows;
}

constexpr size_t k_rows     = 400;
constexpr size_t k_features = 4;

// A deterministic plane, its mappers kept so a materialized copy can be binned
// against the SAME cuts: identical bin ids are what makes a view and a copy
// comparable byte for byte.
struct Plane
{
    detail::ColumnBatch batch;
    BinMappers          mappers;
    Dataset             ds;
    std::vector<float>  grad;
    std::vector<float>  hess;
};

Plane make_plane()
{
    std::mt19937                          rng(7);
    std::uniform_real_distribution<float> uni(0.0F, 1.0F);
    detail::ColumnBatch                   batch;
    for (size_t j = 0; j < k_features; ++j)
    {
        std::vector<float> column(k_rows);
        for (float &v : column)
        {
            v = uni(rng);
        }
        batch.features.push_back(std::move(column));
        batch.feature_names.push_back("f" + std::to_string(j));
    }
    batch.labels.assign(k_rows, 0.0F);
    BinMappers         mappers = BinMappers::fit(batch, BinMapperConfig{});
    Dataset            ds      = Dataset::bin(batch, mappers, {});
    std::vector<float> grad(k_rows);
    for (size_t i = 0; i < k_rows; ++i)
    {
        grad[i] = batch.features[0][i] - 0.5F + (0.25F * batch.features[1][i]);
    }
    return Plane{.batch   = std::move(batch),
                 .mappers = std::move(mappers),
                 .ds      = std::move(ds),
                 .grad    = std::move(grad),
                 .hess    = std::vector<float>(k_rows, 1.0F)};
}

// The same rows binned as their own dataset, under the plane's cuts.
Dataset materialize(Plane const &plane, std::span<row_id_t const> rows)
{
    detail::ColumnBatch sub;
    sub.feature_names = plane.batch.feature_names;
    for (size_t j = 0; j < k_features; ++j)
    {
        std::vector<float> column;
        column.reserve(rows.size());
        for (row_id_t const r : rows)
        {
            column.push_back(plane.batch.features[j][r]);
        }
        sub.features.push_back(std::move(column));
    }
    sub.labels.assign(rows.size(), 0.0F);
    return Dataset::bin(sub, plane.mappers, {});
}

std::vector<float> narrow(std::span<float const> values, std::span<row_id_t const> rows)
{
    std::vector<float> out;
    out.reserve(rows.size());
    for (row_id_t const r : rows)
    {
        out.push_back(values[r]);
    }
    return out;
}

// Every node field of a tree, floats compared by their bits: two trees match
// here only if the models are identical, not merely close.
std::string tree_bytes(DenseTree const &tree)
{
    std::ostringstream out;
    out << tree.params().depth << ':' << tree.params().n_leaves;
    for (DenseTree::Node const &n : tree.nodes())
    {
        out << '|' << n.feature_id << ','
            << std::bit_cast<uint32_t>(n.threshold_or_value) << ',' << n.left << ','
            << n.right << ',' << int{n.default_left};
    }
    return out.str();
}

TreeConfig fill_config()
{
    return TreeConfig{.min_child_hess   = 0.0F,
                      .lambda_l2        = 1.0F,
                      .feature_fraction = 1.0F,
                      .max_depth        = 4,
                      .min_data_in_leaf = 1};
}

// One tree over `rows` of `ds`, with the run list the column fill is handed.
std::string grown(Dataset const &ds, std::span<float const> grad,
                  std::span<float const> hess, RowSelection const &sel)
{
    DepthwiseGrower<> grower{fill_config()};
    return tree_bytes(grower.grow(ds, grad, hess, sel).tree);
}

} // namespace

TEST_CASE("all_rows is the identity in constant time")
{
    RowView const view = RowView::all(1000);
    CHECK(view.form() == RowView::Form::Range);
    CHECK(view.size() == 1000);
    CHECK(view.is_identity());
    CHECK(view.n_runs() == 1);
    CHECK(view.parent_rows() == 1000);
}

TEST_CASE("the encoder run-length encodes into the three forms")
{
    SECTION("one ascending run is a range")
    {
        RowView const view = RowView::encode(iota_rows(100, 900), 1000);
        CHECK(view.form() == RowView::Form::Range);
        CHECK(view.size() == 800);
        CHECK_FALSE(view.is_identity());
    }
    SECTION("the whole ascending run is the identity")
    {
        RowView const view = RowView::encode(iota_rows(0, 1000), 1000);
        CHECK(view.form() == RowView::Form::Range);
        CHECK(view.is_identity());
    }
    SECTION("a range short of the parent is not the identity")
    {
        RowView const view = RowView::encode(iota_rows(0, 999), 1000);
        CHECK(view.form() == RowView::Form::Range);
        CHECK_FALSE(view.is_identity());
    }
    SECTION("a few blocks are segments")
    {
        std::vector<row_id_t> rows = iota_rows(0, 100);
        for (row_id_t r : iota_rows(500, 700))
        {
            rows.push_back(r);
        }
        for (row_id_t r : iota_rows(900, 1000))
        {
            rows.push_back(r);
        }
        RowView const view = RowView::encode(rows, 1000);
        CHECK(view.form() == RowView::Form::Segments);
        CHECK(view.n_runs() == 3);
        CHECK(view.size() == 400);
        CHECK_FALSE(view.is_identity());
    }
    SECTION("every-other-row is a gather, not 500 segments")
    {
        std::vector<row_id_t> rows;
        for (row_id_t r = 0; r < 1000; r += 2)
        {
            rows.push_back(r);
        }
        RowView const view = RowView::encode(rows, 1000);
        CHECK(view.form() == RowView::Form::Gather);
        CHECK(view.size() == 500);
    }
    SECTION("a descending list is a gather that keeps its order")
    {
        std::vector<row_id_t> const rows{9, 7, 5, 3, 1};
        RowView const               view = RowView::encode(rows, 10);
        CHECK(view.form() == RowView::Form::Gather);
        CHECK(view.materialize() == rows);
    }
    SECTION("duplicates stay duplicated")
    {
        std::vector<row_id_t> const rows{0, 0, 1, 1, 2};
        RowView const               view = RowView::encode(rows, 3);
        CHECK(view.size() == 5);
        CHECK_FALSE(view.is_identity());
        CHECK(view.materialize() == rows);
    }
}

TEST_CASE("materialize round-trips every form")
{
    std::vector<std::vector<row_id_t>> const cases{iota_rows(0, 64),
                                                   iota_rows(7, 33),
                                                   {1, 2, 3, 40, 41, 42, 90},
                                                   {5, 1, 99, 4},
                                                   {0, 0, 0}};
    for (auto const &rows : cases)
    {
        RowView const view = RowView::encode(rows, 100);
        CHECK(view.size() == rows.size());
        CHECK(view.materialize() == rows);
    }
}

TEST_CASE("density is the occupancy of the bounding span")
{
    CHECK(RowView::all(100).density() == 1.0);
    CHECK(RowView::encode(iota_rows(10, 20), 100).density() == 1.0);
    // 10 rows spread over a span of 20
    std::vector<row_id_t> rows = iota_rows(0, 5);
    for (row_id_t r : iota_rows(15, 20))
    {
        rows.push_back(r);
    }
    CHECK(RowView::encode(rows, 100).density() == 0.5);
}

TEST_CASE("runs describe the contiguous forms and nothing else")
{
    CHECK(RowView::all(100).runs().size() == 1);
    CHECK(rows_in(RowView::all(100).runs()) == 100);
    RowView const range = RowView::encode(iota_rows(10, 90), 100);
    CHECK(range.runs().size() == 1);
    CHECK(range.runs().front().start == 10);
    CHECK(rows_in(range.runs()) == range.size());
    std::vector<row_id_t> segmented = iota_rows(0, 20);
    for (row_id_t const r : iota_rows(60, 90))
    {
        segmented.push_back(r);
    }
    RowView const segments = RowView::encode(segmented, 100);
    CHECK(segments.runs().size() == 2);
    CHECK(rows_in(segments.runs()) == segments.size());
    // A gather has no contiguity to spend, so it offers no runs and the fill
    // keeps its per-row indirection.
    std::vector<row_id_t> const scattered{9, 7, 5, 3, 1};
    CHECK(RowView::encode(scattered, 100).runs().empty());
}

TEST_CASE("a view whose rows leave the dataset is refused, not read")
{
    Plane const  plane = make_plane();
    size_t const n     = k_rows;
    SECTION("a range reaching past the last row")
    {
        std::vector<row_id_t> const rows{
            static_cast<row_id_t>(n - 2), static_cast<row_id_t>(n - 1),
            static_cast<row_id_t>(n), static_cast<row_id_t>(n + 1)};
        RowView const view = RowView::encode(rows, n);
        CHECK(view.form() == RowView::Form::Range);
        CHECK_FALSE(view.can_fit(n));
        CHECK_THROWS_WITH(plane.ds.with_rows(view),
                          Catch::Matchers::ContainsSubstring("past the last"));
    }
    SECTION("a segment run reaching past the last row")
    {
        std::vector<row_id_t> rows = iota_rows(0, 3);
        rows.push_back(static_cast<row_id_t>(n));
        rows.push_back(static_cast<row_id_t>(n + 1));
        RowView const view = RowView::encode(rows, n);
        CHECK(view.form() == RowView::Form::Segments);
        CHECK_FALSE(view.can_fit(n));
        CHECK_THROWS_WITH(plane.ds.with_rows(view),
                          Catch::Matchers::ContainsSubstring("past the last"));
    }
    SECTION("a gathered id past the last row")
    {
        std::vector<row_id_t> const rows{0, static_cast<row_id_t>(n + 5), 2, 1};
        RowView const               view = RowView::encode(rows, n);
        CHECK(view.form() == RowView::Form::Gather);
        CHECK_THROWS_AS(plane.ds.with_rows(view), std::invalid_argument);
    }
    SECTION("a descriptor built against another dataset's row count")
    {
        RowView const view = RowView::encode(iota_rows(0, 10), n + 7);
        CHECK(view.can_fit(n));
        CHECK_THROWS_WITH(plane.ds.with_rows(view),
                          Catch::Matchers::ContainsSubstring("and this dataset holds"));
    }
}

TEST_CASE("a range view fills from subspans and grows the same tree")
{
    Plane const                 plane = make_plane();
    std::vector<row_id_t> const rows  = iota_rows(37, 337);
    Dataset const view = plane.ds.with_rows(RowView::encode(rows, k_rows));
    REQUIRE(view.row_view().form() == RowView::Form::Range);

    std::string const ranged =
        grown(view, plane.grad, plane.hess,
              {rows, {.identity = false, .runs = view.row_view().runs()}});
    std::string const gathered =
        grown(view, plane.grad, plane.hess, {rows, {.identity = false, .runs = {}}});
    CHECK(ranged == gathered);

    Dataset const            copy = materialize(plane, rows);
    std::vector<float> const g    = narrow(plane.grad, rows);
    std::vector<float> const h    = narrow(plane.hess, rows);
    std::string const        copied =
        grown(copy, g, h,
              {iota_rows(0, static_cast<row_id_t>(rows.size())), {.identity = true}});
    CHECK(ranged == copied);
}

TEST_CASE("a segmented view fills run by run and grows the same tree")
{
    Plane const           plane = make_plane();
    std::vector<row_id_t> rows  = iota_rows(11, 120);
    for (row_id_t const r : iota_rows(150, 260))
    {
        rows.push_back(r);
    }
    for (row_id_t const r : iota_rows(300, 390))
    {
        rows.push_back(r);
    }
    Dataset const view = plane.ds.with_rows(RowView::encode(rows, k_rows));
    REQUIRE(view.row_view().form() == RowView::Form::Segments);
    REQUIRE(view.row_view().runs().size() == 3);

    std::string const ranged =
        grown(view, plane.grad, plane.hess,
              {rows, {.identity = false, .runs = view.row_view().runs()}});
    std::string const gathered =
        grown(view, plane.grad, plane.hess, {rows, {.identity = false, .runs = {}}});
    CHECK(ranged == gathered);

    Dataset const            copy = materialize(plane, rows);
    std::vector<float> const g    = narrow(plane.grad, rows);
    std::vector<float> const h    = narrow(plane.hess, rows);
    std::string const        copied =
        grown(copy, g, h,
              {iota_rows(0, static_cast<row_id_t>(rows.size())), {.identity = true}});
    CHECK(ranged == copied);
}

TEST_CASE("a unit hessian takes the same subspan arms")
{
    Plane const                 plane = make_plane();
    std::vector<row_id_t> const rows  = iota_rows(64, 320);
    Dataset const     view = plane.ds.with_rows(RowView::encode(rows, k_rows));
    std::string const ranged =
        grown(view, plane.grad, {},
              {rows, {.identity = false, .runs = view.row_view().runs()}});
    CHECK(ranged ==
          grown(view, plane.grad, {}, {rows, {.identity = false, .runs = {}}}));
}

TEST_CASE("a view of many short runs grows the same tree as a gather")
{
    // Runs shorter than the fill's prefetch distance are not worth subspans,
    // so this view gathers; the answer must not depend on which arm ran.
    Plane const           plane = make_plane();
    std::vector<row_id_t> rows;
    for (row_id_t start = 0; start < static_cast<row_id_t>(k_rows); start += 8)
    {
        for (row_id_t r = start; r < start + 3; ++r)
        {
            rows.push_back(r);
        }
    }
    Dataset const view = plane.ds.with_rows(RowView::encode(rows, k_rows));
    REQUIRE(view.row_view().form() == RowView::Form::Segments);
    REQUIRE(view.row_view().runs().size() == 50);

    std::string const ranged =
        grown(view, plane.grad, plane.hess,
              {rows, {.identity = false, .runs = view.row_view().runs()}});
    CHECK(ranged ==
          grown(view, plane.grad, plane.hess, {rows, {.identity = false, .runs = {}}}));

    Dataset const            copy = materialize(plane, rows);
    std::vector<float> const g    = narrow(plane.grad, rows);
    std::vector<float> const h    = narrow(plane.hess, rows);
    CHECK(ranged == grown(copy, g, h,
                          {iota_rows(0, static_cast<row_id_t>(rows.size())),
                           {.identity = true}}));
}
