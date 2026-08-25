// Device TreeSHAP parity. The device case is compiled in every build and SKIPs
// at runtime unless cuda_available(); it fits on the host so the only thing
// under test is the device walk, then requires the device contributions to
// match the host bin walk's to a tolerance. Unlike device predict this is not
// a bit-equal contract: the kernel builds its path polynomials in fp32 where
// the host evaluator is fp64 throughout, and the per-feature accumulation is
// an atomicAdd whose order the device picks.
//
// The merged path length picks the kernel, so the fixtures span the three
// arms: a path merges one element per distinct feature, so the feature count
// caps it and the depth reaches it.
//
// The decline cases run on every host: without a usable device the plan is
// null and the caller keeps its host walk.

#include "bonsai/bin_mapper.hpp"
#include "bonsai/bin_mappers.hpp"
#include "bonsai/booster.hpp"
#include "bonsai/config/bin_mapper_config.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/config/data_config.hpp"
#include "bonsai/cuda/histogram_engine.hpp"
#include "bonsai/cuda/shap.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/detail/column_batch.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/objective.hpp"
#include "bonsai/sampler.hpp"
#include "bonsai/tree.hpp"
#include "bonsai/types.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <print>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace
{

using namespace bonsai;

using MseBooster =
    Booster<MSEObjective, DepthwiseGrower<CpuHistogramEngine>, AllRowsSampler>;

// The levelwise arm reaches the device through the same seam, by way of the
// dense equivalents its plan input owns.
using ObliviousMseBooster =
    Booster<MSEObjective, ObliviousGrower<CpuHistogramEngine>, AllRowsSampler>;

// The fp32 walk against the fp64 host evaluator. Worst measured on a Jetson
// Orin Nano (sm_87) over the three fixtures below: element gap 1.4e-7
// relative, additivity residual 2.2e-7 relative, both two orders inside this
// pin.
constexpr double k_tol = 1e-5;

// A NaN column keeps the missing bin populated, so the element's missing_ok
// arm is exercised beside the ordinary interval test. Column 1 is a coarse
// staircase of column 0, so the grower splits on both and paths that merge two
// constraints on one feature appear.
detail::ColumnBatch shap_batch(size_t n, size_t nf)
{
    std::mt19937                          rng(23);
    std::uniform_real_distribution<float> u(0.0F, 1.0F);
    detail::ColumnBatch                   batch;
    batch.features.assign(nf, std::vector<float>(n));
    batch.feature_names.resize(nf);
    batch.labels.assign(n, 0.0F);
    for (size_t j = 0; j < nf; ++j)
    {
        batch.feature_names[j] = "f" + std::to_string(j);
    }
    for (size_t r = 0; r < n; ++r)
    {
        float s = 0.0F;
        for (size_t j = 0; j < nf; ++j)
        {
            float const v        = u(rng);
            batch.features[j][r] = (j == nf - 1 && r % 7 == 0)
                                       ? std::numeric_limits<float>::quiet_NaN()
                                       : v;
            s += v * static_cast<float>(j + 1);
        }
        batch.features[1][r] = std::floor(batch.features[0][r] * 4.0F);
        batch.labels[r]      = s;
    }
    return batch;
}

Config shap_cfg(size_t max_depth)
{
    Config cfg{};
    cfg.tree_config.max_depth        = max_depth;
    cfg.tree_config.min_data_in_leaf = 4;
    cfg.booster_config.learning_rate = 0.1F;
    cfg.booster_config.random_seed   = 11;
    return cfg;
}

// Nodes a densified oblivious tree carries with no training evidence behind
// them. A levelwise fixture that has none is not testing what it claims to.
size_t count_zero_cover(std::span<DenseTree const> trees)
{
    size_t dead = 0;
    for (DenseTree const &tree : trees)
    {
        dead += static_cast<size_t>(
            std::count(tree.covers().begin(), tree.covers().end(), 0.0F));
    }
    return dead;
}

// The worst relative element gap and the worst relative additivity residual
// over one fixture, both reported so the tolerance above stays a measurement.
// Returns the zero-cover node count so a levelwise caller can assert the dead
// slots it means to exercise were really there.
template <typename BoosterT = MseBooster>
size_t check_parity(size_t n_rows, size_t n_feats, size_t max_depth, char const *what)
{
    auto const batch   = shap_batch(n_rows, n_feats);
    auto const mappers = BinMappers::fit(batch, BinMapperConfig{});
    auto const plane   = cuda_ingest(batch, mappers);
    REQUIRE(plane);
    auto const ds = Dataset::bin(batch, mappers, DataConfig{}, plane);

    BoosterT booster{shap_cfg(max_depth)};
    for (size_t i = 0; i < 8; ++i)
    {
        booster.update_one_iter(ds);
    }

    auto const in = booster.predict_plan_input();
    auto const plan =
        cuda_shap_plan(in.trees, mappers, in.learning_rate, in.init_score);
    REQUIRE(plan);

    size_t const        n    = ds.n_rows();
    size_t const        cols = n_feats + 1;
    std::vector<double> host(n * cols, 0.0);
    std::vector<double> dev(n * cols, 0.0);
    REQUIRE(cuda_pred_contribs(*plan, *plane, n, n_feats, dev));
    booster.pred_contribs_binned(ds, host, n_feats);

    std::vector<float> margin(n, 0.0F);
    booster.predict_at_binned(ds, margin, 0);

    double worst_elem = 0.0;
    double worst_add  = 0.0;
    for (size_t r = 0; r < n; ++r)
    {
        double max_abs = 0.0;
        for (size_t c = 0; c < cols; ++c)
        {
            max_abs = std::max(max_abs, std::abs(host[(r * cols) + c]));
        }
        double const scale = std::max(1.0, max_abs);
        double       sum   = 0.0;
        for (size_t c = 0; c < cols; ++c)
        {
            double const gap =
                std::abs(dev[(r * cols) + c] - host[(r * cols) + c]) / scale;
            worst_elem = std::max(worst_elem, gap);
            sum += dev[(r * cols) + c];
        }
        double const m = margin[r];
        worst_add = std::max(worst_add, std::abs(sum - m) / std::max(1.0, std::abs(m)));
    }
    std::println("cuda-shap parity ({}): worst element gap {:.2e}, worst additivity "
                 "residual {:.2e}",
                 what, worst_elem, worst_add);
    CHECK(worst_elem <= k_tol);
    CHECK(worst_add <= k_tol);
    return count_zero_cover(in.trees);
}

} // namespace

TEST_CASE("cuda_pred_contribs matches the host binned walk", "[cuda][shap]")
{
    if (!cuda_available())
    {
        SKIP("device TreeSHAP needs a usable CUDA device");
    }
    SECTION("paths of at most 8 elements")
    {
        check_parity(2048, 5, 5, "K=8");
    }
    SECTION("paths of at most 16 elements")
    {
        check_parity(2048, 24, 12, "K=16");
    }
    SECTION("paths of at most 32 elements")
    {
        check_parity(4096, 48, 20, "K=32");
    }
}

TEST_CASE("cuda_pred_contribs matches the host binned walk for a levelwise model",
          "[cuda][shap][oblivious]")
{
    if (!cuda_available())
    {
        SKIP("device TreeSHAP needs a usable CUDA device");
    }
    // Densification mints a perfect tree, so a levelwise model reaches the
    // kernel carrying nodes no training row ever visited. The host walk skips
    // those branches and the device closed form multiplies them out instead,
    // which is the arithmetic this case exists to check; the dead-slot count
    // is asserted so the fixture cannot quietly stop producing them.
    SECTION("paths of at most 8 elements")
    {
        CHECK(check_parity<ObliviousMseBooster>(2048, 5, 5, "levelwise K=8") > 0);
    }
    SECTION("paths of at most 16 elements")
    {
        CHECK(check_parity<ObliviousMseBooster>(2048, 24, 12, "levelwise K=16") > 0);
    }
}

TEST_CASE("cuda_pred_contribs declines a plane of the wrong shape", "[cuda][shap]")
{
    if (!cuda_available())
    {
        SKIP("device TreeSHAP needs a usable CUDA device");
    }
    auto const batch   = shap_batch(512, 5);
    auto const mappers = BinMappers::fit(batch, BinMapperConfig{});
    auto const plane   = cuda_ingest(batch, mappers);
    REQUIRE(plane);
    auto const ds = Dataset::bin(batch, mappers, DataConfig{}, plane);

    MseBooster booster{shap_cfg(5)};
    booster.update_one_iter(ds);
    auto const in = booster.predict_plan_input();
    auto const plan =
        cuda_shap_plan(in.trees, mappers, in.learning_rate, in.init_score);
    REQUIRE(plan);

    size_t const        n = ds.n_rows();
    std::vector<double> dev(n * (ds.n_features() + 2), 0.0);
    REQUIRE(!cuda_pred_contribs(*plan, *plane, n, ds.n_features() + 1, dev));
}

TEST_CASE("the device shap plan declines without a device", "[shap]")
{
    if (cuda_available())
    {
        SKIP("this host has a usable CUDA device; the plan is built, not refused");
    }
    std::vector<BinMapper> one;
    one.push_back(BinMapper::from_cuts({0.5F, 1.0F}));
    auto const mappers = BinMappers::from_mappers(std::move(one), {"a"});

    std::vector<DenseTree> trees;
    trees.emplace_back(DenseTree::Nodes{DenseTree::leaf(1.0F)},
                       DenseTree::Params{.depth = 0, .n_leaves = 1});
    REQUIRE(cuda_shap_plan(trees, mappers, 0.1F, 0.0F) == nullptr);
}

TEST_CASE("the device shap plan declines a model with no covers", "[cuda][shap]")
{
    if (!cuda_available())
    {
        SKIP("the cover check runs after the device check");
    }
    std::vector<BinMapper> one;
    one.push_back(BinMapper::from_cuts({0.5F, 1.0F}));
    auto const mappers = BinMappers::from_mappers(std::move(one), {"a"});

    // Hand-built: no per-node covers, which the packer refuses.
    std::vector<DenseTree> trees;
    trees.emplace_back(DenseTree::Nodes{DenseTree::internal(0, 0.5F, 1, 2, true),
                                        DenseTree::leaf(1.0F), DenseTree::leaf(2.0F)},
                       DenseTree::Params{.depth = 1, .n_leaves = 2});
    REQUIRE(cuda_shap_plan(trees, mappers, 0.1F, 0.0F) == nullptr);
}
