// CUDA grower parity tests. Compiled in every build; each case SKIPs at
// runtime unless cuda_available(). GPU histograms accumulate per-chunk in
// float (merged in double), and atomics add in arbitrary order, so
// comparisons are tolerance-based rather than bit-exact. A configuration the
// device cannot hold is refused with ConfigError, not moved to the host, so
// the cases that used to assert host-fallback parity assert the error.

#include "bonsai/booster.hpp"
#include "bonsai/config/tree_config.hpp"
#include "bonsai/cuda/grower.hpp"
#include "bonsai/cuda/histogram_engine.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/objective.hpp"
#include "bonsai/split.hpp"
#include "bonsai/types.hpp"
#include "test_grower_helpers.hpp"

#include "bonsai/config/errors.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <utility>
#include <vector>

namespace
{

using namespace bonsai;

// 4096 rows x 4 features with duplicates-heavy value
// ranges and a NaN column so the missing bin is populated; seeded so
// failures reproduce.
test::ScenarioInputs random_scenario()
{
    std::mt19937                          rng(7);
    std::uniform_real_distribution<float> value(0.0F, 1.0F);
    std::normal_distribution<float>       gradient(0.0F, 1.0F);
    size_t const                          n = 4096;

    detail::ColumnBatch batch;
    batch.features.resize(4, std::vector<float>(n));
    batch.feature_names = {"a", "b", "c", "d"};
    batch.labels.assign(n, 0.0F);
    std::vector<float> grad(n);
    std::vector<float> hess(n);
    for (size_t r = 0; r < n; ++r)
    {
        batch.features[0][r] = value(rng);
        batch.features[1][r] = std::round(value(rng) * 8.0F); // few distinct bins
        batch.features[2][r] = value(rng);
        batch.features[3][r] =
            (r % 5 == 0) ? std::numeric_limits<float>::quiet_NaN() : value(rng);
        grad[r] = gradient(rng);
        hess[r] = 0.5F + value(rng);
    }
    return {.built = test::build(std::move(batch)),
            .grad  = std::move(grad),
            .hess  = std::move(hess),
            .rows  = test::iota_rows(n)};
}

TEST_CASE("CudaDepthwiseGrower predictions match DepthwiseGrower", "[cuda][grower]")
{
    if (!cuda_available())
    {
        SKIP("no usable CUDA device");
    }
    auto        scenario = random_scenario();
    auto const &ds       = scenario.built.ds;

    TreeConfig cfg;
    cfg.max_depth        = 5;
    cfg.min_data_in_leaf = 4;

    DepthwiseGrower<CpuHistogramEngine> cpu_grower(cfg);
    CudaDepthwiseGrower                 gpu_grower(cfg);

    auto cpu = cpu_grower.grow(ds, scenario.grad, scenario.hess, scenario.rows);
    auto gpu = gpu_grower.grow(ds, scenario.grad, scenario.hess, scenario.rows);

    REQUIRE(cpu.values.size() == gpu.values.size());
    for (size_t r = 0; r < cpu.values.size(); ++r)
    {
        REQUIRE_THAT(gpu.values[r], Catch::Matchers::WithinAbs(cpu.values[r], 1e-4));
    }
}

// Shared body for the device eval-plane parity cases: grow on the device,
// then require the device walk to match the host binned walk per row at
// rounding level (the walk has no atomics, so 1e-6, not the growers' 1e-4),
// and the device mse loss to match the host objective over the same scores.
template <typename GrowerT> void check_device_eval_parity(uint8_t max_depth)
{
    if (!cuda_available())
    {
        SKIP("no usable CUDA device");
    }
    auto        scenario = random_scenario();
    auto const &ds       = scenario.built.ds;

    TreeConfig cfg;
    cfg.max_depth        = max_depth;
    cfg.min_data_in_leaf = 4;

    GrowerT grower(cfg);
    auto    grown = grower.grow(ds, scenario.grad, scenario.hess, scenario.rows);

    float const        lr = 0.1F;
    std::vector<float> host_scores(ds.plane_n_rows(), 0.25F);
    std::vector<float> dev_scores = host_scores;

    std::optional<float> loss;
    REQUIRE(grower.eval_begin(ds, DeviceObjectiveKind::none, dev_scores));
    REQUIRE(grower.eval_accumulate(grown.tree, ds, lr, dev_scores, loss));
    REQUIRE(!loss.has_value());

    auto const sb = internal::split_bins(grown.tree, ds);
    for (size_t r = 0; r < host_scores.size(); ++r)
    {
        host_scores[r] += lr * internal::value_binned(grown.tree, sb, [&](size_t f)
                                                      { return ds.bin_at(f, r); });
        REQUIRE_THAT(dev_scores[r], Catch::Matchers::WithinAbs(host_scores[r], 1e-6));
    }

    // With a device-capable kind the loss reduces on device and the scores
    // stay there; the value must match the host objective over the same
    // walked scores.
    std::vector<float> dev_scores2(ds.plane_n_rows(), 0.25F);
    REQUIRE(grower.eval_begin(ds, DeviceObjectiveKind::mse, dev_scores2));
    std::optional<float> dev_loss;
    REQUIRE(grower.eval_accumulate(grown.tree, ds, lr, dev_scores2, dev_loss));
    REQUIRE(dev_loss.has_value());
    float const host_loss = MSEObjective::eval(
        floats_view{host_scores.data(), host_scores.size()}, ds.labels());
    REQUIRE_THAT(*dev_loss, Catch::Matchers::WithinAbs(host_loss, 1e-5));
}

TEST_CASE("CudaDepthwiseGrower device eval walk matches the host binned walk",
          "[cuda][grower][eval]")
{
    check_device_eval_parity<CudaDepthwiseGrower>(5);
}

// The oblivious flatten synthesizes the perfect-tree numbering; the same
// bound applies against the host's oblivious binned walk.
TEST_CASE("CudaObliviousGrower device eval walk matches the host binned walk",
          "[cuda][grower][eval]")
{
    check_device_eval_parity<CudaObliviousGrower>(4);
}

// Twelve features: more than one bin tile at the shipping width, so a fit
// crosses a full tile and a narrow tail one. The narrow tail is the layout's
// one asymmetric case and the full tile is the only one whose strip loads as
// a single aligned vector, so both paths need a fit over them.
test::ScenarioInputs wide_scenario()
{
    std::mt19937                          rng(13);
    std::uniform_real_distribution<float> value(0.0F, 1.0F);
    std::normal_distribution<float>       gradient(0.0F, 1.0F);
    size_t const                          n     = 4096;
    size_t const                          feats = 12;

    detail::ColumnBatch batch;
    batch.features.resize(feats, std::vector<float>(n));
    batch.feature_names = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"};
    batch.labels.assign(n, 0.0F);
    std::vector<float> grad(n);
    std::vector<float> hess(n);
    for (size_t r = 0; r < n; ++r)
    {
        for (size_t f = 0; f < feats; ++f)
        {
            batch.features[f][r] = f == 3 ? std::round(value(rng) * 8.0F) : value(rng);
        }
        // A NaN column in the tail tile, so the missing bin is populated there.
        batch.features[9][r] =
            (r % 5 == 0) ? std::numeric_limits<float>::quiet_NaN() : value(rng);
        grad[r] = gradient(rng);
        hess[r] = 0.5F + value(rng);
    }
    return {.built = test::build(std::move(batch)),
            .grad  = std::move(grad),
            .hess  = std::move(hess),
            .rows  = test::iota_rows(n)};
}

TEST_CASE("CudaDepthwiseGrower matches CPU across bin tiles", "[cuda][grower]")
{
    if (!cuda_available())
    {
        SKIP("no usable CUDA device");
    }
    auto        scenario = wide_scenario();
    auto const &ds       = scenario.built.ds;

    TreeConfig cfg;
    cfg.max_depth        = 5;
    cfg.min_data_in_leaf = 4;

    DepthwiseGrower<CpuHistogramEngine> cpu_grower(cfg);
    CudaDepthwiseGrower                 gpu_grower(cfg);

    auto cpu = cpu_grower.grow(ds, scenario.grad, scenario.hess, scenario.rows);
    auto gpu = gpu_grower.grow(ds, scenario.grad, scenario.hess, scenario.rows);

    REQUIRE(cpu.values.size() == gpu.values.size());
    for (size_t r = 0; r < cpu.values.size(); ++r)
    {
        REQUIRE_THAT(gpu.values[r], Catch::Matchers::WithinAbs(cpu.values[r], 1e-4));
    }
}

TEST_CASE("CudaDepthwiseGrower matches CPU under feature subsampling", "[cuda][grower]")
{
    if (!cuda_available())
    {
        SKIP("no usable CUDA device");
    }
    // The bin plane groups every feature into tiles, so a tree that selects a
    // subset walks the same tiles and skips lanes through the slot map. Both
    // growers draw the same features from the same seed.
    auto        scenario = wide_scenario();
    auto const &ds       = scenario.built.ds;

    TreeConfig cfg;
    cfg.max_depth        = 5;
    cfg.min_data_in_leaf = 4;
    cfg.feature_fraction = 0.5F;
    cfg.feature_seed     = 11;

    DepthwiseGrower<CpuHistogramEngine> cpu_grower(cfg);
    CudaDepthwiseGrower                 gpu_grower(cfg);

    auto cpu = cpu_grower.grow(ds, scenario.grad, scenario.hess, scenario.rows);
    auto gpu = gpu_grower.grow(ds, scenario.grad, scenario.hess, scenario.rows);

    REQUIRE(cpu.values.size() == gpu.values.size());
    for (size_t r = 0; r < cpu.values.size(); ++r)
    {
        REQUIRE_THAT(gpu.values[r], Catch::Matchers::WithinAbs(cpu.values[r], 1e-4));
    }
}

TEST_CASE("CudaObliviousGrower predictions match ObliviousGrower", "[cuda][grower]")
{
    if (!cuda_available())
    {
        SKIP("no usable CUDA device");
    }
    auto        scenario = random_scenario();
    auto const &ds       = scenario.built.ds;

    TreeConfig cfg;
    cfg.max_depth        = 5;
    cfg.min_data_in_leaf = 4;

    ObliviousGrower<CpuHistogramEngine> cpu_grower(cfg);
    CudaObliviousGrower                 gpu_grower(cfg);

    auto cpu = cpu_grower.grow(ds, scenario.grad, scenario.hess, scenario.rows);
    auto gpu = gpu_grower.grow(ds, scenario.grad, scenario.hess, scenario.rows);

    REQUIRE(cpu.values.size() == gpu.values.size());
    for (size_t r = 0; r < cpu.values.size(); ++r)
    {
        REQUIRE_THAT(gpu.values[r], Catch::Matchers::WithinAbs(cpu.values[r], 1e-4));
    }
}

TEST_CASE("CudaObliviousGrower survives frontiers wider than one node chunk",
          "[cuda][grower]")
{
    // The device level-find processes the frontier in 32-node chunks; depth 7
    // on 4096 rows exercises multi-chunk levels (64+ nodes) that the register
    // -tiled first implementation silently truncated.
    if (!cuda_available())
    {
        SKIP("no usable CUDA device");
    }
    auto        scenario = random_scenario();
    auto const &ds       = scenario.built.ds;

    TreeConfig cfg;
    cfg.max_depth        = 7;
    cfg.min_data_in_leaf = 1;

    ObliviousGrower<CpuHistogramEngine> cpu_grower(cfg);
    CudaObliviousGrower                 gpu_grower(cfg);

    auto cpu = cpu_grower.grow(ds, scenario.grad, scenario.hess, scenario.rows);
    auto gpu = gpu_grower.grow(ds, scenario.grad, scenario.hess, scenario.rows);

    REQUIRE(cpu.values.size() == gpu.values.size());
    for (size_t r = 0; r < cpu.values.size(); ++r)
    {
        REQUIRE_THAT(gpu.values[r], Catch::Matchers::WithinAbs(cpu.values[r], 1e-4));
    }
}

TEST_CASE("CudaObliviousGrower matches CPU when deep nodes go infeasible (issue #60)",
          "[cuda][grower]")
{
    // The CPU level-find lets an infeasible node contribute its parent score
    // (zero gain) rather than veto the whole level candidate (split.cpp,
    // issue #60). The device level-find originally kept the veto, so at depth
    // >= 5 — where some frontier node is always near-empty — GPU levelwise
    // chose worse splits than its own CPU grower and silently lost accuracy at
    // scale (0.011 test r2 at 16M). A high min_child_hess forces that
    // infeasibility at shallow depth so the divergence reproduces on 4k rows:
    // pre-fix this REQUIRE fails; with the parent-score port it holds.
    if (!cuda_available())
    {
        SKIP("no usable CUDA device");
    }
    auto        scenario = random_scenario();
    auto const &ds       = scenario.built.ds;

    TreeConfig cfg;
    cfg.max_depth        = 7;
    cfg.min_data_in_leaf = 1;
    cfg.min_child_hess   = 6.0; // deep children fall under this, going infeasible

    ObliviousGrower<CpuHistogramEngine> cpu_grower(cfg);
    CudaObliviousGrower                 gpu_grower(cfg);

    auto cpu = cpu_grower.grow(ds, scenario.grad, scenario.hess, scenario.rows);
    auto gpu = gpu_grower.grow(ds, scenario.grad, scenario.hess, scenario.rows);

    REQUIRE(cpu.values.size() == gpu.values.size());
    for (size_t r = 0; r < cpu.values.size(); ++r)
    {
        REQUIRE_THAT(gpu.values[r], Catch::Matchers::WithinAbs(cpu.values[r], 1e-4));
    }
}

// ---- The leaf plane (docs/invariants.md) ------------------

// Every leafwise parity case asserts the same contract as the level plane's:
// tolerance-equal predictions, never tree equality.
void require_values_match(std::vector<float> const &cpu, std::vector<float> const &gpu)
{
    REQUIRE(cpu.size() == gpu.size());
    for (size_t r = 0; r < cpu.size(); ++r)
    {
        REQUIRE_THAT(gpu[r], Catch::Matchers::WithinAbs(cpu[r], 1e-4));
    }
}

TEST_CASE("CudaLeafwiseGrower predictions match LeafwiseGrower", "[cuda][grower][fit]")
{
    if (!cuda_available())
    {
        SKIP("no usable CUDA device");
    }
    auto        scenario = random_scenario();
    auto const &ds       = scenario.built.ds;

    TreeConfig cfg;
    cfg.max_depth        = 5;
    cfg.max_leaves       = 31;
    cfg.min_data_in_leaf = 4;

    LeafwiseGrower<CpuHistogramEngine> cpu_grower(cfg);
    CudaLeafwiseGrower                 gpu_grower(cfg);

    auto cpu = cpu_grower.grow(ds, scenario.grad, scenario.hess, scenario.rows);
    auto gpu = gpu_grower.grow(ds, scenario.grad, scenario.hess, scenario.rows);
    require_values_match(cpu.values, gpu.values);

    // Row subset: the root segment uploads instead of restoring the cached
    // identity, and route_unsampled fills the rows growth never reached.
    std::vector<row_id_t> half;
    for (row_id_t r = 0; r < scenario.rows.size(); r += 2)
    {
        half.push_back(r);
    }
    auto cpu_sub = cpu_grower.grow(ds, scenario.grad, scenario.hess, half);
    auto gpu_sub = gpu_grower.grow(ds, scenario.grad, scenario.hess, half);
    require_values_match(cpu_sub.values, gpu_sub.values);
}

TEST_CASE("CudaLeafwiseGrower matches CPU under a leaf budget with no depth cap",
          "[cuda][grower][fit]")
{
    if (!cuda_available())
    {
        SKIP("no usable CUDA device");
    }
    // The shape where leaf-wise growth structurally differs from depth-wise:
    // the budget, not the depth, is what stops it, so the tree grows ragged
    // and the slot pool is sized off max_leaves.
    auto        scenario = random_scenario();
    auto const &ds       = scenario.built.ds;

    TreeConfig cfg;
    cfg.max_depth        = 255;
    cfg.max_leaves       = 40;
    cfg.min_data_in_leaf = 4;

    LeafwiseGrower<CpuHistogramEngine> cpu_grower(cfg);
    CudaLeafwiseGrower                 gpu_grower(cfg);

    auto cpu = cpu_grower.grow(ds, scenario.grad, scenario.hess, scenario.rows);
    auto gpu = gpu_grower.grow(ds, scenario.grad, scenario.hess, scenario.rows);
    REQUIRE(gpu.tree.params().n_leaves == cpu.tree.params().n_leaves);
    require_values_match(cpu.values, gpu.values);
}

TEST_CASE("CudaLeafwiseGrower honors the depth cap", "[cuda][grower][fit]")
{
    if (!cuda_available())
    {
        SKIP("no usable CUDA device");
    }
    // The cap is the host grower's, not the plane's: with the leaf budget off,
    // depth alone must stop growth (LightGBM's CUDA learner ignores it; ours
    // cannot, or the two leafwise growers would disagree by construction).
    auto        scenario = random_scenario();
    auto const &ds       = scenario.built.ds;

    TreeConfig cfg;
    cfg.max_depth        = 3;
    cfg.max_leaves       = 0;
    cfg.min_data_in_leaf = 4;

    LeafwiseGrower<CpuHistogramEngine> cpu_grower(cfg);
    CudaLeafwiseGrower                 gpu_grower(cfg);

    auto cpu = cpu_grower.grow(ds, scenario.grad, scenario.hess, scenario.rows);
    auto gpu = gpu_grower.grow(ds, scenario.grad, scenario.hess, scenario.rows);
    REQUIRE(gpu.tree.params().depth <= 3);
    REQUIRE(gpu.tree.params().n_leaves <= 8);
    require_values_match(cpu.values, gpu.values);
}

TEST_CASE("CudaLeafwiseGrower matches CPU with constraints on the leaf plane",
          "[cuda][grower][fit]")
{
    if (!cuda_available())
    {
        SKIP("no usable CUDA device");
    }
    // Both constraint families ride the same staging the level find uses, but
    // here a find covers two nodes, not a level: the per-node bounds box and
    // the interaction mask must be indexed by node, the histogram by slot.
    auto        scenario = random_scenario();
    auto const &ds       = scenario.built.ds;

    TreeConfig cfg;
    cfg.max_depth               = 5;
    cfg.max_leaves              = 24;
    cfg.min_data_in_leaf        = 4;
    cfg.monotone_constraints    = {+1, 0, -1, 0};
    cfg.interaction_constraints = {"0,1", "2,3"};

    LeafwiseGrower<CpuHistogramEngine> cpu_grower(cfg);
    CudaLeafwiseGrower                 gpu_grower(cfg);

    auto cpu = cpu_grower.grow(ds, scenario.grad, scenario.hess, scenario.rows);
    auto gpu = gpu_grower.grow(ds, scenario.grad, scenario.hess, scenario.rows);
    require_values_match(cpu.values, gpu.values);
}

TEST_CASE("CudaLeafwiseGrower handles consecutive trees and datasets",
          "[cuda][grower][fit]")
{
    if (!cuda_available())
    {
        SKIP("no usable CUDA device");
    }
    // The slot pool is per tree: it zeroes once and its slot counter restarts
    // at the root, so a second grow that reuses the same buffer must not read
    // the first tree's cells.
    auto scenario_a = random_scenario();
    auto scenario_b = test::separable_4row();

    TreeConfig cfg;
    cfg.max_depth        = 4;
    cfg.max_leaves       = 12;
    cfg.min_data_in_leaf = 4;

    CudaLeafwiseGrower                 grower(cfg);
    LeafwiseGrower<CpuHistogramEngine> cpu_grower(cfg);
    auto const first  = grower.grow(scenario_a.built.ds, scenario_a.grad,
                                    scenario_a.hess, scenario_a.rows);
    auto const second = grower.grow(scenario_a.built.ds, scenario_a.grad,
                                    scenario_a.hess, scenario_a.rows);
    require_values_match(first.values, second.values);
    auto const cpu = cpu_grower.grow(scenario_a.built.ds, scenario_a.grad,
                                     scenario_a.hess, scenario_a.rows);
    require_values_match(cpu.values, second.values);

    // A second dataset re-uploads the binned matrix and resizes the pool.
    auto const other = grower.grow(scenario_b.built.ds, scenario_b.grad,
                                   scenario_b.hess, scenario_b.rows);
    REQUIRE(other.values.size() == scenario_b.built.ds.plane_n_rows());
}

TEST_CASE("CudaLeafwiseGrower matches CPU on a deep unconstrained tree",
          "[cuda][grower][fit]")
{
    if (!cuda_available())
    {
        SKIP("no usable CUDA device");
    }
    // Deep, small-node growth: the shape where a split's partition can empty
    // one child (subtraction noise scores a degenerate cut, decision 50) and
    // the round must demote back to a leaf without spending a pool slot. It
    // also drives every small child below the 512-row kernel cutoff.
    auto        scenario = random_scenario();
    auto const &ds       = scenario.built.ds;

    TreeConfig cfg;
    cfg.max_depth         = 8;
    cfg.max_leaves        = 64;
    cfg.min_data_in_leaf  = 4;
    cfg.min_gain_to_split = 0.0F;

    LeafwiseGrower<CpuHistogramEngine> cpu_grower(cfg);
    CudaLeafwiseGrower                 gpu_grower(cfg);

    auto cpu = cpu_grower.grow(ds, scenario.grad, scenario.hess, scenario.rows);
    auto gpu = gpu_grower.grow(ds, scenario.grad, scenario.hess, scenario.rows);
    require_values_match(cpu.values, gpu.values);
}

TEST_CASE("CudaLeafwiseGrower refuses an oversized leaf budget", "[cuda][grower][fit]")
{
    if (!cuda_available())
    {
        SKIP("no usable CUDA device");
    }
    // A leaf budget whose slot pool no device can hold. The tree is refused
    // where the pool is sized, naming the budget: the alternative, training it
    // on the host plane, is a silent 10x that reads as a working fit.
    auto        scenario = random_scenario();
    auto const &ds       = scenario.built.ds;

    TreeConfig cfg;
    cfg.max_depth        = 4;
    cfg.max_leaves       = 1U << 30U;
    cfg.min_data_in_leaf = 4;

    CudaLeafwiseGrower gpu_grower(cfg);
    REQUIRE_THROWS_AS(gpu_grower.grow(ds, scenario.grad, scenario.hess, scenario.rows),
                      ConfigError);
}

TEST_CASE("CudaDepthwiseGrower handles consecutive trees and datasets",
          "[cuda][grower]")
{
    if (!cuda_available())
    {
        SKIP("no usable CUDA device");
    }
    // One engine instance across two grows on one dataset, then a switch
    // to a second dataset — exercises the upload cache in begin_tree.
    auto scenario_a = random_scenario();
    auto scenario_b = test::separable_4row();

    TreeConfig cfg;
    cfg.max_depth = 3;

    CudaDepthwiseGrower grower(cfg);
    auto const          first  = grower.grow(scenario_a.built.ds, scenario_a.grad,
                                             scenario_a.hess, scenario_a.rows);
    auto const          second = grower.grow(scenario_a.built.ds, scenario_a.grad,
                                             scenario_a.hess, scenario_a.rows);
    // Not bit-exact: atomic accumulation order differs between runs, so the
    // two grows may disagree in the last ulps even on identical inputs.
    for (size_t r = 0; r < first.values.size(); ++r)
    {
        REQUIRE_THAT(second.values[r],
                     Catch::Matchers::WithinAbs(first.values[r], 1e-4));
    }

    auto const other = grower.grow(scenario_b.built.ds, scenario_b.grad,
                                   scenario_b.hess, scenario_b.rows);
    REQUIRE(other.values.size() == scenario_b.built.ds.plane_n_rows());
}

} // namespace

TEST_CASE("CudaDepthwiseGrower matches CPU past the 48KiB shared-memory budget",
          "[cuda][grower]")
{
    if (!cuda_available())
    {
        SKIP("no usable CUDA device");
    }
    // ~4095 bins per feature: over the static 48KiB budget (3072 bins), so
    // this exercises the dynamic shared-memory opt-in. A device that does not
    // grant an opt-in this wide refuses the fit instead (see the max_bin case
    // below); every device this suite runs on does.
    std::mt19937                          rng(11);
    std::uniform_real_distribution<float> value(0.0F, 1.0F);
    std::normal_distribution<float>       gradient(0.0F, 1.0F);
    size_t const                          n = 16384;

    detail::ColumnBatch batch;
    batch.features.resize(3, std::vector<float>(n));
    batch.feature_names = {"a", "b", "c"};
    batch.labels.assign(n, 0.0F);
    std::vector<float> grad(n);
    std::vector<float> hess(n);
    for (size_t r = 0; r < n; ++r)
    {
        batch.features[0][r] = value(rng);
        batch.features[1][r] = value(rng);
        batch.features[2][r] = value(rng);
        grad[r]              = gradient(rng);
        hess[r]              = 0.5F + value(rng);
    }
    BinMapperConfig bm;
    bm.max_bin         = 4096;
    BinMappers mappers = BinMappers::fit(batch, bm);
    Dataset    ds      = Dataset::bin(batch, mappers, {});
    REQUIRE(ds.n_bins(0) > 3072); // the scenario must actually cross the cliff

    TreeConfig cfg;
    cfg.max_depth        = 4;
    cfg.min_data_in_leaf = 4;

    DepthwiseGrower<CpuHistogramEngine> cpu_grower(cfg);
    CudaDepthwiseGrower                 gpu_grower(cfg);

    auto rows = test::iota_rows(n);
    auto cpu  = cpu_grower.grow(ds, grad, hess, rows);
    auto gpu  = gpu_grower.grow(ds, grad, hess, rows);

    REQUIRE(cpu.values.size() == gpu.values.size());
    for (size_t r = 0; r < cpu.values.size(); ++r)
    {
        REQUIRE_THAT(gpu.values[r], Catch::Matchers::WithinAbs(cpu.values[r], 1e-4));
    }
}

TEST_CASE("A max_bin past the shared-memory ceiling is refused", "[cuda][grower]")
{
    if (!cuda_available())
    {
        SKIP("no usable CUDA device");
    }
    // ~16k bins per feature: 4*16384*4B = 256KiB of shared memory, past every
    // device's opt-in ceiling. begin_root refuses the tree and names the
    // limit; the host fallback this replaced trained a wrong model once
    // (issue #12, fixed in cd4e726) and hid a large slowdown the rest of the
    // time.
    std::mt19937                          rng(13);
    std::uniform_real_distribution<float> value(0.0F, 1.0F);
    std::normal_distribution<float>       gradient(0.0F, 1.0F);
    size_t const                          n = 40960;

    detail::ColumnBatch batch;
    batch.features.resize(2, std::vector<float>(n));
    batch.feature_names = {"a", "b"};
    batch.labels.assign(n, 0.0F);
    std::vector<float> grad(n);
    std::vector<float> hess(n);
    for (size_t r = 0; r < n; ++r)
    {
        batch.features[0][r] = value(rng);
        batch.features[1][r] = value(rng);
        grad[r]              = gradient(rng);
        hess[r]              = 0.5F + value(rng);
    }
    BinMapperConfig bm;
    // 24576, not 16384: decision 51's ceiling stride caps cuts at the budget,
    // and 40960 rows at 16384 now bin to ~13.7k bins, below the smem opt-in
    // ceiling this test must exceed.
    bm.max_bin         = 24576;
    BinMappers mappers = BinMappers::fit(batch, bm);
    Dataset    ds      = Dataset::bin(batch, mappers, {});
    REQUIRE(4 * ds.n_bins(0) * sizeof(float) > 227UL * 1024UL); // past every ceiling

    TreeConfig cfg;
    cfg.max_depth        = 4;
    cfg.min_data_in_leaf = 4;

    auto rows = test::iota_rows(n);
    // Every device plane refuses it: level-batched growth and best-first
    // growth gate on the same histogram budget.
    CudaObliviousGrower oblivious(cfg);
    REQUIRE_THROWS_AS(oblivious.grow(ds, grad, hess, rows), ConfigError);
    CudaDepthwiseGrower depthwise(cfg);
    REQUIRE_THROWS_AS(depthwise.grow(ds, grad, hess, rows), ConfigError);
    CudaLeafwiseGrower leafwise(cfg);
    REQUIRE_THROWS_AS(leafwise.grow(ds, grad, hess, rows), ConfigError);

    // The host plane trains the same dataset without complaint: the refusal is
    // a device-capacity fact, not a rejected dataset.
    ObliviousGrower<CpuHistogramEngine> cpu_grower(cfg);
    auto const                          cpu = cpu_grower.grow(ds, grad, hess, rows);
    REQUIRE(cpu.values.size() == n);
}

// ---- The ingest transaction (decision 54) ------------------------------------

TEST_CASE("cuda_ingest bins bit-identically to the host fill", "[cuda][ingest]")
{
    if (!cuda_available())
    {
        SKIP("no usable CUDA device");
    }
    std::mt19937                          rng(11);
    std::uniform_real_distribution<float> value(0.0F, 1.0F);
    size_t const                          n = 4096;

    detail::ColumnBatch batch;
    batch.features.resize(3, std::vector<float>(n));
    batch.feature_names = {"a", "b", "c"};
    batch.labels.assign(n, 0.0F);
    for (size_t r = 0; r < n; ++r)
    {
        batch.features[0][r] = value(rng);
        batch.features[1][r] = std::round(value(rng) * 8.0F); // few distinct bins
        batch.features[2][r] =
            (r % 5 == 0) ? std::numeric_limits<float>::quiet_NaN() : value(rng);
    }
    auto const mappers = BinMappers::fit(batch, BinMapperConfig{});
    auto const host_ds = Dataset::bin(batch, mappers, {});

    SECTION("feature-major arm (ColumnBatch)")
    {
        auto plane = cuda_ingest(batch, mappers);
        REQUIRE(plane != nullptr);
        auto const dev_ds = Dataset::bin(batch, mappers, {}, std::move(plane));
        REQUIRE(dev_ds.bins_are_u8() == host_ds.bins_are_u8());
        for (size_t f = 0; f < host_ds.n_features(); ++f)
        {
            for (size_t r = 0; r < n; ++r)
            {
                REQUIRE(dev_ds.bin_at(f, r) == host_ds.bin_at(f, r));
            }
        }
    }

    SECTION("row-major arm (features_view)")
    {
        std::vector<float> rowmajor(n * batch.features.size());
        for (size_t r = 0; r < n; ++r)
        {
            for (size_t f = 0; f < batch.features.size(); ++f)
            {
                rowmajor[(r * batch.features.size()) + f] = batch.features[f][r];
            }
        }
        features_view const X{rowmajor.data(), n, batch.features.size()};
        auto                plane = cuda_ingest(X, mappers);
        REQUIRE(plane != nullptr);
        auto const dev_ds =
            Dataset::bin(X, floats_view{batch.labels}, mappers, {}, std::move(plane));
        for (size_t f = 0; f < host_ds.n_features(); ++f)
        {
            for (size_t r = 0; r < n; ++r)
            {
                REQUIRE(dev_ds.bin_at(f, r) == host_ds.bin_at(f, r));
            }
        }
    }

    SECTION("u16 bins (max_bin past 256)")
    {
        BinMapperConfig wide;
        wide.max_bin      = 1000;
        auto const m16    = BinMappers::fit(batch, wide);
        auto const host16 = Dataset::bin(batch, m16, {});
        auto       plane  = cuda_ingest(batch, m16);
        REQUIRE(plane != nullptr);
        auto const dev16 = Dataset::bin(batch, m16, {}, std::move(plane));
        REQUIRE(dev16.bins_are_u8() == host16.bins_are_u8());
        REQUIRE_FALSE(dev16.bins_are_u8());
        for (size_t f = 0; f < host16.n_features(); ++f)
        {
            for (size_t r = 0; r < n; ++r)
            {
                REQUIRE(dev16.bin_at(f, r) == host16.bin_at(f, r));
            }
        }
    }
}

TEST_CASE("CudaDepthwiseGrower trains identically on a device-binned dataset",
          "[cuda][ingest][grower]")
{
    if (!cuda_available())
    {
        SKIP("no usable CUDA device");
    }
    auto        scenario = random_scenario();
    auto const &host_ds  = scenario.built.ds;
    auto        plane    = cuda_ingest(scenario.built.batch, scenario.built.mappers);
    REQUIRE(plane != nullptr);
    auto const dev_ds = Dataset::bin(scenario.built.batch, scenario.built.mappers, {},
                                     std::move(plane));

    TreeConfig cfg;
    cfg.max_depth        = 5;
    cfg.min_data_in_leaf = 4;
    CudaDepthwiseGrower grower(cfg);

    auto host_out = grower.grow(host_ds, scenario.grad, scenario.hess, scenario.rows);
    auto dev_out  = grower.grow(dev_ds, scenario.grad, scenario.hess, scenario.rows);

    // Bins are bit-identical (previous case), but two GPU fits differ in
    // float-atomic accumulation order run to run — the suite's standard
    // 1e-4 tolerance, same as the CPU-parity cases.
    REQUIRE(host_out.values.size() == dev_out.values.size());
    for (size_t r = 0; r < host_out.values.size(); ++r)
    {
        REQUIRE_THAT(dev_out.values[r],
                     Catch::Matchers::WithinAbs(host_out.values[r], 1e-4));
    }

    // Row subset: route_unsampled walks the tree via bin_at, forcing the
    // lazy host materialization from the plane.
    std::vector<row_id_t> half;
    for (row_id_t r = 0; r < scenario.rows.size(); r += 2)
    {
        half.push_back(r);
    }
    auto host_sub = grower.grow(host_ds, scenario.grad, scenario.hess, half);
    auto dev_sub  = grower.grow(dev_ds, scenario.grad, scenario.hess, half);
    for (size_t r = 0; r < host_sub.values.size(); ++r)
    {
        REQUIRE_THAT(dev_sub.values[r],
                     Catch::Matchers::WithinAbs(host_sub.values[r], 1e-4));
    }
}

// ---- Constraints on the device plane (issue #149) ----------------------------

TEST_CASE("CudaDepthwiseGrower: monotone +1 forces non-decreasing predictions",
          "[cuda][grower][monotone]")
{
    if (!cuda_available())
    {
        SKIP("no usable CUDA device");
    }
    // Three value groups whose gradient means swing down-up (-1, +2, -2), so
    // the unconstrained tree is non-monotone in the feature — the CPU monotone
    // test's data shape, sized to 4096 rows so the device find really runs.
    std::mt19937                          rng(17);
    std::uniform_real_distribution<float> jitter(0.0F, 0.8F);
    size_t const                          n = 4096;

    detail::ColumnBatch batch;
    batch.features.resize(1, std::vector<float>(n));
    batch.feature_names = {"a"};
    batch.labels.assign(n, 0.0F);
    std::vector<float>         grad(n);
    std::vector<float>         hess(n, 1.0F);
    std::array<float, 3> const group_grad{-1.0F, +2.0F, -2.0F};
    for (size_t r = 0; r < n; ++r)
    {
        size_t const g       = r % 3;
        batch.features[0][r] = static_cast<float>(g) + jitter(rng);
        grad[r]              = group_grad[g];
    }
    auto built = test::build(std::move(batch));
    auto rows  = test::iota_rows(n);

    TreeConfig unconstrained;
    unconstrained.max_depth          = 4;
    unconstrained.min_data_in_leaf   = 4;
    TreeConfig constrained           = unconstrained;
    constrained.monotone_constraints = {+1};

    auto predict_curve = [&](DenseTree const &tree)
    {
        std::vector<float> out;
        for (float x : {0.4F, 1.4F, 2.4F})
        {
            out.push_back(test::predict_one(tree, std::vector<float>{x}));
        }
        return out;
    };

    // Sanity: the unconstrained GPU tree is non-monotone on this data,
    // otherwise the constrained assertion below would pass vacuously.
    CudaDepthwiseGrower free_grower(unconstrained);
    auto                free_out   = free_grower.grow(built.ds, grad, hess, rows);
    auto const          free_curve = predict_curve(free_out.tree);
    REQUIRE((free_curve[1] < free_curve[0] || free_curve[2] < free_curve[1]));

    CudaDepthwiseGrower gpu_grower(constrained);
    auto                gpu   = gpu_grower.grow(built.ds, grad, hess, rows);
    auto const          curve = predict_curve(gpu.tree);
    CHECK(curve[0] <= curve[1]);
    CHECK(curve[1] <= curve[2]);

    // Same constrained config on the CPU plane: the device find mirrors the
    // CPU bound-propagation scheme, so predictions agree to the suite band.
    DepthwiseGrower<CpuHistogramEngine> cpu_grower(constrained);
    auto cpu = cpu_grower.grow(built.ds, grad, hess, rows);
    REQUIRE(cpu.values.size() == gpu.values.size());
    for (size_t r = 0; r < cpu.values.size(); ++r)
    {
        REQUIRE_THAT(gpu.values[r], Catch::Matchers::WithinAbs(cpu.values[r], 1e-4));
    }
}

TEST_CASE("CudaDepthwiseGrower: interaction constraints keep groups on separate paths",
          "[cuda][grower][interaction]")
{
    if (!cuda_available())
    {
        SKIP("no usable CUDA device");
    }
    auto        scenario = random_scenario();
    auto const &ds       = scenario.built.ds;

    TreeConfig cfg;
    cfg.max_depth               = 5;
    cfg.min_data_in_leaf        = 4;
    cfg.interaction_constraints = {"0,1", "2,3"};

    CudaDepthwiseGrower gpu_grower(cfg);
    auto gpu = gpu_grower.grow(ds, scenario.grad, scenario.hess, scenario.rows);

    // Walk every root-to-leaf path; no path may mix features across groups.
    auto const &nodes = gpu.tree.nodes();
    REQUIRE(gpu.tree.params().n_leaves > 1); // the walk must not be vacuous
    std::vector<std::pair<node_id_t, std::set<feature_id_t>>> stack{{0, {}}};
    while (!stack.empty())
    {
        auto [id, used] = stack.back();
        stack.pop_back();
        auto const &node = nodes[id];
        if (DenseTree::is_leaf(node))
        {
            bool const mixes_groups = (used.contains(0) || used.contains(1)) &&
                                      (used.contains(2) || used.contains(3));
            CHECK(!mixes_groups);
            continue;
        }
        used.insert(node.feature_id);
        stack.push_back({node.left, used});
        stack.push_back({node.right, used});
    }

    DepthwiseGrower<CpuHistogramEngine> cpu_grower(cfg);
    auto cpu = cpu_grower.grow(ds, scenario.grad, scenario.hess, scenario.rows);
    REQUIRE(cpu.values.size() == gpu.values.size());
    for (size_t r = 0; r < cpu.values.size(); ++r)
    {
        REQUIRE_THAT(gpu.values[r], Catch::Matchers::WithinAbs(cpu.values[r], 1e-4));
    }
}

TEST_CASE("CudaObliviousGrower rejects constraints at construction",
          "[cuda][grower][monotone]")
{
    // Construction-time contract (shared with the CPU levelwise grower); the
    // engine allocates lazily, so this pins the ConfigError on every build,
    // device or not — no SKIP.
    TreeConfig mono;
    mono.monotone_constraints = {+1};
    REQUIRE_THROWS_AS(CudaObliviousGrower(mono), ConfigError);

    TreeConfig inter;
    inter.interaction_constraints = {"0", "1"};
    REQUIRE_THROWS_AS(CudaObliviousGrower(inter), ConfigError);
}

TEST_CASE("cuda_select_device: rejects an out-of-range device id", "[cuda][edge]")
{
    // Deterministic on every build: the stub throws for any nonzero id, and
    // the real backend throws when the id is at or above the visible count.
    REQUIRE_THROWS_AS(bonsai::cuda_select_device(10000), bonsai::ConfigError);
}

TEST_CASE("cuda_select_device: id 0 is accepted everywhere", "[cuda][edge]")
{
    // The config default: a no-op without a device, cudaSetDevice(0) with
    // one. Either way it must not throw, so CPU configs and GPU-less hosts
    // are untouched by the knob existing.
    REQUIRE_NOTHROW(bonsai::cuda_select_device(0));
}

TEST_CASE("cuda_select_device: a second device trains when present", "[cuda][grower]")
{
    if (!bonsai::cuda_available())
    {
        SKIP("no usable CUDA device");
    }
    try
    {
        bonsai::cuda_select_device(1);
    }
    catch (bonsai::ConfigError const &)
    {
        SKIP("single-GPU host");
    }
    // Placed on device 1: a small fit must match the CPU grower end to end
    // there, the same parity bar as the device-0 cases above.
    auto        scenario = random_scenario();
    auto const &ds       = scenario.built.ds;
    TreeConfig  cfg;
    cfg.max_depth        = 5;
    cfg.min_data_in_leaf = 4;
    DepthwiseGrower<CpuHistogramEngine> cpu_grower(cfg);
    CudaDepthwiseGrower                 gpu_grower(cfg);
    auto cpu = cpu_grower.grow(ds, scenario.grad, scenario.hess, scenario.rows);
    auto gpu = gpu_grower.grow(ds, scenario.grad, scenario.hess, scenario.rows);
    REQUIRE(cpu.values.size() == gpu.values.size());
    for (size_t r = 0; r < cpu.values.size(); ++r)
    {
        REQUIRE_THAT(gpu.values[r], Catch::Matchers::WithinAbs(cpu.values[r], 1e-4));
    }
    bonsai::cuda_select_device(0); // restore for subsequent [cuda] cases
}
