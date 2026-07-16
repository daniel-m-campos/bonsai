// CUDA histogram-backend parity tests. Compiled in every build; each case
// SKIPs at runtime unless cuda_available(). They exercise the real device:
// scenarios are sized above the engine's CPU-fallback cutoff so populate
// really launches kernels. GPU histograms accumulate per-chunk in float
// (merged in double), and atomics add in arbitrary order, so comparisons
// are tolerance-based rather than bit-exact.

#include "bonsai/config/tree_config.hpp"
#include "bonsai/cuda/grower.hpp"
#include "bonsai/cuda/histogram_engine.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/grower.hpp"
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
#include <random>
#include <set>
#include <utility>
#include <vector>

namespace
{

using namespace bonsai;

// 4096 rows (comfortably above the GPU engine's CPU-fallback cutoff, so
// the device path actually runs) x 4 features with duplicates-heavy value
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

// Float per-chunk accumulation bounds the error at ~1e-5 relative for the
// few-thousand-row sums these scenarios produce.
void require_hists_match(SplitInput const &cpu, SplitInput const &gpu)
{
    REQUIRE(cpu.hists.size() == gpu.hists.size());
    for (size_t f = 0; f < cpu.hists.size(); ++f)
    {
        REQUIRE(cpu.hists[f].size() == gpu.hists[f].size());
        for (size_t b = 0; b < cpu.hists[f].size(); ++b)
        {
            auto const bin = static_cast<bin_id_t>(b);
            REQUIRE_THAT(
                gpu.hists[f][bin].sum_grad,
                Catch::Matchers::WithinRel(cpu.hists[f][bin].sum_grad, 1e-4F) ||
                    Catch::Matchers::WithinAbs(cpu.hists[f][bin].sum_grad, 1e-5));
            REQUIRE_THAT(
                gpu.hists[f][bin].sum_hess,
                Catch::Matchers::WithinRel(cpu.hists[f][bin].sum_hess, 1e-4F) ||
                    Catch::Matchers::WithinAbs(cpu.hists[f][bin].sum_hess, 1e-5));
        }
    }
}

TEST_CASE("CudaHistogramEngine matches CPU histograms on all rows", "[cuda][histogram]")
{
    if (!cuda_available())
    {
        SKIP("no usable CUDA device");
    }
    auto        scenario = random_scenario();
    auto const &ds       = scenario.built.ds;

    std::vector<feature_id_t> selected(ds.n_features());
    std::iota(selected.begin(), selected.end(), feature_id_t{0});

    CpuHistogramEngine  cpu_engine;
    CudaHistogramEngine gpu_engine;
    gpu_engine.begin_tree(ds, scenario.grad, scenario.hess);

    SplitInput cpu_node;
    cpu_node.rows = scenario.rows;
    cpu_engine.populate(ds, scenario.grad, scenario.hess, cpu_node, selected);

    SplitInput gpu_node;
    gpu_node.rows = scenario.rows;
    gpu_engine.populate(ds, scenario.grad, scenario.hess, gpu_node, selected);

    require_hists_match(cpu_node, gpu_node);
}

TEST_CASE("CudaHistogramEngine matches CPU histograms on a row subset",
          "[cuda][histogram]")
{
    if (!cuda_available())
    {
        SKIP("no usable CUDA device");
    }
    auto        scenario = random_scenario();
    auto const &ds       = scenario.built.ds;

    // Every third row, plus a feature subset with a placeholder gap.
    std::vector<row_id_t> rows;
    for (row_id_t r = 0; r < ds.n_rows(); r += 3)
    {
        rows.push_back(r);
    }
    std::vector<feature_id_t> const selected{0, 2, 3};

    CpuHistogramEngine  cpu_engine;
    CudaHistogramEngine gpu_engine;
    gpu_engine.begin_tree(ds, scenario.grad, scenario.hess);

    SplitInput cpu_node;
    cpu_node.rows = rows;
    cpu_engine.populate(ds, scenario.grad, scenario.hess, cpu_node, selected);

    SplitInput gpu_node;
    gpu_node.rows = rows;
    gpu_engine.populate(ds, scenario.grad, scenario.hess, gpu_node, selected);

    REQUIRE(gpu_node.hists[1].size() == 0); // unselected placeholder
    require_hists_match(cpu_node, gpu_node);
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
    // >= 5 — where some frontier node is always near-empty — GPU oblivious
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
    REQUIRE(other.values.size() == scenario_b.built.ds.n_rows());
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
    // this exercises the dynamic shared-memory opt-in on devices that grant
    // it and the CPU fallback on devices that don't — parity must hold
    // either way.
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

TEST_CASE("CudaObliviousGrower host fallback matches ObliviousGrower (issue #12)",
          "[cuda][grower]")
{
    if (!cuda_available())
    {
        SKIP("no usable CUDA device");
    }
    // ~16k bins per feature: 4*16384*4B = 256KiB of shared memory, past every
    // device's opt-in ceiling, so begin_root declines and the whole tree runs
    // the host-fallback plane. The fallback used to skip leaf stamping and
    // silently train garbage (issue #12).
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
    // and 40960 rows at 16384 now bin to ~13.7k bins — below the smem opt-in
    // ceiling this test must exceed to guarantee the fallback path.
    bm.max_bin         = 24576;
    BinMappers mappers = BinMappers::fit(batch, bm);
    Dataset    ds      = Dataset::bin(batch, mappers, {});
    REQUIRE(4 * ds.n_bins(0) * sizeof(float) > 227UL * 1024UL); // must force fallback

    TreeConfig cfg;
    cfg.max_depth        = 4;
    cfg.min_data_in_leaf = 4;

    ObliviousGrower<CpuHistogramEngine> cpu_grower(cfg);
    CudaObliviousGrower                 gpu_grower(cfg);

    auto rows = test::iota_rows(n);
    auto cpu  = cpu_grower.grow(ds, grad, hess, rows);
    auto gpu  = gpu_grower.grow(ds, grad, hess, rows);

    REQUIRE(cpu.values.size() == gpu.values.size());
    for (size_t r = 0; r < cpu.values.size(); ++r)
    {
        REQUIRE_THAT(gpu.values[r], Catch::Matchers::WithinAbs(cpu.values[r], 1e-4));
    }
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
    // Construction-time contract (shared with the CPU oblivious grower); the
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
