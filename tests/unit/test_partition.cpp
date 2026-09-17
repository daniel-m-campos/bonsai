#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <random>
#include <vector>

#include "bonsai/dataset.hpp"
#include "bonsai/detail/column_batch.hpp"
#include "bonsai/parallel.hpp"
#include "bonsai/split.hpp"
#include "bonsai/types.hpp"
#include "level_step.hpp"
#include "step/primitives.hpp"
#include "test_grower_helpers.hpp"

using namespace bonsai; // NOLINT
namespace gd = bonsai::grower_detail;

namespace
{

Dataset make_dataset(size_t n_rows, size_t n_features)
{
    std::mt19937                          rng(11);
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
    batch.labels.assign(n_rows, 0.0F);
    batch.feature_names.assign(n_features, "f");
    return test::build(std::move(batch)).ds;
}

gd::PendingSplit blocked(Dataset const &ds, std::vector<row_id_t> rows,
                         SplitOutput const &s, size_t block_rows, int workers)
{
    gd::LevelPlan      plan;
    gd::DeferredSplit &d = plan.splits.emplace_back();
    d.parent.rows        = std::move(rows);
    d.split              = s;
    d.left_id            = 1;
    d.right_id           = 2;
    gd::host_partition(ds, plan, block_rows, workers);
    return std::move(plan.splits.front().p);
}

} // namespace

TEST_CASE("the blocked partition is the serial order at every decomposition",
          "[partition]")
{
    parallel::set_n_threads(4);
    Dataset const ds   = make_dataset(5000, 2);
    auto const    rows = test::iota_rows(ds.plane_n_rows());
    SplitOutput   s;
    s.feature_id   = 0;
    s.bin_id       = static_cast<bin_id_t>(ds.n_bins(0) / 2);
    s.default_left = true;

    auto const            last_bin = static_cast<bin_id_t>(ds.n_bins(0) - 1);
    std::vector<row_id_t> left;
    std::vector<row_id_t> right;
    for (row_id_t const r : rows)
    {
        bool const goes_left =
            routes_left(ds.bin_at(0, r), last_bin, s.bin_id, s.default_left);
        (goes_left ? left : right).push_back(r);
    }
    REQUIRE(!left.empty());
    REQUIRE(!right.empty());

    // Decomposition is a scheduling choice, never an output: one worker per
    // block, several blocks per worker, and a single block all agree with
    // the row-order scan above.
    for (auto const [block_rows, workers] :
         {std::pair<size_t, int>{5000, 1}, {2500, 2}, {625, 8}, {97, 4}, {1, 3}})
    {
        gd::PendingSplit const got = blocked(ds, rows, s, block_rows, workers);
        REQUIRE(got.left.rows == left);
        REQUIRE(got.right.rows == right);
    }
    parallel::set_n_threads(0);
}

TEST_CASE("a parent partitions on the whole team or on none of it", "[partition]")
{
    size_t const floor_rows = gd::partition_rows_per_worker;
    // No intermediate team exists: a partition region either matches the
    // team every other region uses, or there is no region.
    for (size_t rows :
         {size_t{0}, floor_rows - 1, 2 * floor_rows, (12 * floor_rows) - 1})
    {
        REQUIRE(gd::partition_workers(rows, 12) == 1);
    }
    REQUIRE(gd::partition_workers(12 * floor_rows, 12) == 12);
    REQUIRE(gd::partition_workers(1000 * floor_rows, 12) == 12);
    REQUIRE(gd::partition_workers(8 * floor_rows, 8) == 8);
    REQUIRE(gd::partition_workers(1000 * floor_rows, 1) == 1);
}

// INVARIANT: smaller-child-tie-break-agrees
// On equal child row counts the fresh histogram slot goes to the LEFT child.
// The larger sibling derives by subtracting the smaller from the parent, so
// host and device must pick the same side or a subtraction reads the wrong
// sibling's histogram and the tree silently changes. The device makes the
// same `<=` comparison twice, in publish_small_child (which child's rows
// the partition kernel fills) and in leaf_children (which child holds the
// fresh slot), and those two must agree with each other as well.
TEST_CASE("partition: an equal-count split gives the fresh slot to the left",
          "[partition][invariant]")
{
    grower_detail::PendingSplit p;
    p.left.rows.assign(4, row_id_t{0});
    p.right.rows.assign(4, row_id_t{0});
    CHECK(&grower_detail::smaller_child(p) == &p.left);

    p.right.rows.assign(3, row_id_t{0});
    CHECK(&grower_detail::smaller_child(p) == &p.right);

    p.right.rows.assign(5, row_id_t{0});
    CHECK(&grower_detail::smaller_child(p) == &p.left);
}
