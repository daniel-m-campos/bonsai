// MultiCudaHistogramEngine parity tests. Compiled in every build; the device
// cases SKIP at runtime unless cuda_available(). The multi engine is a
// tolerance-match to the single engine (atomics accumulate in arbitrary order
// and the shard reduction re-sums in a different order), so comparisons use the
// suite's standard 1e-4 band. The single-GPU validation harness for the whole
// multi path is N contexts on ONE device (duplicate ids), so tests 2 and 3 run
// on any GPU host; two distinct real devices (test 5) SKIP on a single-GPU box.

#include "bonsai/config/errors.hpp"
#include "bonsai/config/tree_config.hpp"
#include "bonsai/cuda/grower.hpp"
#include "bonsai/cuda/histogram_engine.hpp"
#include "bonsai/cuda/multi_engine.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/split.hpp"
#include "bonsai/types.hpp"
#include "test_grower_helpers.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>
#include <span>
#include <vector>

namespace
{

using namespace bonsai;

using MultiDepthwiseGrower = DepthwiseGrower<MultiCudaHistogramEngine>;
using MultiObliviousGrower =
    ObliviousGrower<MultiCudaHistogramEngine, HistogramLevelSplitFinder>;

// Resets the process-wide device set on scope exit so a leaked selection never
// bleeds into a later [cuda] case.
struct DeviceSetGuard
{
    ~DeviceSetGuard()
    {
        cuda_select_devices(std::span<uint32_t const>{});
    }
};

// 4096 rows (above the GPU engine's CPU-fallback cutoff, so the device path
// runs) x 4 features with a NaN column so the missing bin is populated; seeded.
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
        batch.features[1][r] = std::round(value(rng) * 8.0F);
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

template <typename A, typename B> void require_values_close(A const &a, B const &b)
{
    REQUIRE(a.values.size() == b.values.size());
    for (size_t r = 0; r < a.values.size(); ++r)
    {
        REQUIRE_THAT(b.values[r], Catch::Matchers::WithinAbs(a.values[r], 1e-4));
    }
}

TreeConfig parity_config()
{
    TreeConfig cfg;
    cfg.max_depth        = 5;
    cfg.min_data_in_leaf = 4;
    return cfg;
}

TEST_CASE("MultiCudaHistogramEngine: matches CudaHistogramEngine on one device",
          "[cuda][grower]")
{
    if (!cuda_available())
    {
        SKIP("no usable CUDA device");
    }
    DeviceSetGuard guard;
    cuda_select_devices(std::span<uint32_t const>{}); // default: current device
    auto        scenario = random_scenario();
    auto const &ds       = scenario.built.ds;
    auto const  cfg      = parity_config();

    SECTION("depthwise")
    {
        CudaDepthwiseGrower  single(cfg);
        MultiDepthwiseGrower multi(cfg);
        auto s = single.grow(ds, scenario.grad, scenario.hess, scenario.rows);
        auto m = multi.grow(ds, scenario.grad, scenario.hess, scenario.rows);
        require_values_close(s, m);
    }
    SECTION("oblivious")
    {
        CudaObliviousGrower  single(cfg);
        MultiObliviousGrower multi(cfg);
        auto s = single.grow(ds, scenario.grad, scenario.hess, scenario.rows);
        auto m = multi.grow(ds, scenario.grad, scenario.hess, scenario.rows);
        require_values_close(s, m);
    }
}

TEST_CASE("MultiCudaHistogramEngine: two contexts on one device reproduce the "
          "single-engine model",
          "[cuda][grower]")
{
    if (!cuda_available())
    {
        SKIP("no usable CUDA device");
    }
    DeviceSetGuard                guard;
    std::array<uint32_t, 2> const ids{0, 0};
    cuda_select_devices(ids); // two contexts on device 0: the shard harness
    auto        scenario = random_scenario();
    auto const &ds       = scenario.built.ds;
    auto const  cfg      = parity_config();

    SECTION("depthwise")
    {
        MultiDepthwiseGrower multi(cfg);
        // The single-engine reference runs on the default device set.
        cuda_select_devices(std::span<uint32_t const>{});
        CudaDepthwiseGrower single(cfg);
        auto m = multi.grow(ds, scenario.grad, scenario.hess, scenario.rows);
        auto s = single.grow(ds, scenario.grad, scenario.hess, scenario.rows);
        require_values_close(s, m);
    }
    SECTION("oblivious")
    {
        MultiObliviousGrower multi(cfg);
        cuda_select_devices(std::span<uint32_t const>{});
        CudaObliviousGrower single(cfg);
        auto m = multi.grow(ds, scenario.grad, scenario.hess, scenario.rows);
        auto s = single.grow(ds, scenario.grad, scenario.hess, scenario.rows);
        require_values_close(s, m);
    }
}

TEST_CASE("MultiCudaHistogramEngine: host-staged reduction matches the peer path",
          "[cuda][grower]")
{
    if (!cuda_available())
    {
        SKIP("no usable CUDA device");
    }
    DeviceSetGuard guard;
    // The env hook forces peer_ok false at construction, exercising the pinned
    // host bounce path; set it before the multi engine (grower member) is built.
    setenv("BONSAI_MULTI_HOST_STAGED", "1", 1);
    std::array<uint32_t, 2> const ids{0, 0};
    cuda_select_devices(ids);
    auto        scenario = random_scenario();
    auto const &ds       = scenario.built.ds;
    auto const  cfg      = parity_config();

    MultiDepthwiseGrower multi(cfg);
    unsetenv("BONSAI_MULTI_HOST_STAGED");
    cuda_select_devices(std::span<uint32_t const>{});
    CudaDepthwiseGrower single(cfg);

    auto m = multi.grow(ds, scenario.grad, scenario.hess, scenario.rows);
    auto s = single.grow(ds, scenario.grad, scenario.hess, scenario.rows);
    require_values_close(s, m);
}

TEST_CASE("cuda_select_devices: rejects an out-of-range device id", "[cuda][edge]")
{
    // Deterministic on every build: the stub throws for any nonzero id, and the
    // real backend throws when the id is at or above the visible count.
    DeviceSetGuard                guard;
    std::array<uint32_t, 1> const ids{10000};
    REQUIRE_THROWS_AS(cuda_select_devices(ids), ConfigError);
}

TEST_CASE(
    "MultiCudaHistogramEngine: two real devices reproduce the single-device model",
    "[cuda][grower]")
{
    if (!cuda_available())
    {
        SKIP("no usable CUDA device");
    }
    DeviceSetGuard                guard;
    std::array<uint32_t, 2> const ids{0, 1};
    try
    {
        cuda_select_devices(ids);
    }
    catch (ConfigError const &)
    {
        SKIP("single-GPU host");
    }
    auto        scenario = random_scenario();
    auto const &ds       = scenario.built.ds;
    auto const  cfg      = parity_config();

    MultiDepthwiseGrower multi(cfg);
    cuda_select_devices(std::span<uint32_t const>{});
    CudaDepthwiseGrower single(cfg);

    auto m = multi.grow(ds, scenario.grad, scenario.hess, scenario.rows);
    auto s = single.grow(ds, scenario.grad, scenario.hess, scenario.rows);
    require_values_close(s, m);
}

} // namespace
