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

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <random>
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
                Catch::Matchers::WithinRel(cpu.hists[f][bin].sum_grad, 1e-4) ||
                    Catch::Matchers::WithinAbs(cpu.hists[f][bin].sum_grad, 1e-5));
            REQUIRE_THAT(
                gpu.hists[f][bin].sum_hess,
                Catch::Matchers::WithinRel(cpu.hists[f][bin].sum_hess, 1e-4) ||
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
