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

    SplitInput parent;
    parent.rows                = rows;
    gd::PendingSplit const ref = gd::partition_rows(ds, std::move(parent), s, 1, 2);
    REQUIRE(!ref.left.rows.empty());
    REQUIRE(!ref.right.rows.empty());

    // Decomposition is a scheduling choice, never an output: one worker per
    // block, several blocks per worker, and a single block all agree.
    for (auto const [block_rows, workers] :
         {std::pair<size_t, int>{5000, 1}, {2500, 2}, {625, 8}, {97, 4}, {1, 3}})
    {
        gd::PendingSplit const got = blocked(ds, rows, s, block_rows, workers);
        REQUIRE(got.left.rows == ref.left.rows);
        REQUIRE(got.right.rows == ref.right.rows);
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
