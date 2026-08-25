// Device predict parity. The device case is compiled in every build and SKIPs
// at runtime unless cuda_available(); it fits on the host so the only thing
// under test is the walk, then requires the device scores to equal the host
// bin walk's bit for bit. The walk has no atomics and sums the trees in the
// same order the host does, so equality is the contract, not a tolerance;
// only the device's fused multiply-add on init + lr * sum could break it, and
// that shows on a GPU host, not here.
//
// The decline case runs on every host: without a usable device the plan is
// null and the caller keeps its host walk.

#include "bonsai/bin_mapper.hpp"
#include "bonsai/bin_mappers.hpp"
#include "bonsai/booster.hpp"
#include "bonsai/config/bin_mapper_config.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/config/data_config.hpp"
#include "bonsai/cuda/histogram_engine.hpp"
#include "bonsai/cuda/predict.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/detail/column_batch.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/objective.hpp"
#include "bonsai/sampler.hpp"
#include "bonsai/tree.hpp"
#include "bonsai/types.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <limits>
#include <memory>
#include <random>
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

// A NaN column keeps the missing bin populated, so the walk's default_left
// arm is exercised beside the ordinary comparison.
detail::ColumnBatch predict_batch(size_t n, size_t nf)
{
    std::mt19937                          rng(19);
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
        batch.labels[r] = s;
    }
    return batch;
}

Config predict_cfg()
{
    Config cfg{};
    cfg.tree_config.max_depth        = 5;
    cfg.tree_config.min_data_in_leaf = 8;
    cfg.booster_config.learning_rate = 0.1F;
    cfg.booster_config.random_seed   = 7;
    return cfg;
}

} // namespace

TEST_CASE("cuda_predict matches the host binned walk", "[cuda][predict]")
{
    if (!cuda_available())
    {
        SKIP("device predict needs a usable CUDA device");
    }
    auto const batch   = predict_batch(4096, 5);
    auto const mappers = BinMappers::fit(batch, BinMapperConfig{});
    auto const plane   = cuda_ingest(batch, mappers);
    REQUIRE(plane);
    auto const ds = Dataset::bin(batch, mappers, DataConfig{}, plane);

    MseBooster booster{predict_cfg()};
    for (size_t i = 0; i < 8; ++i)
    {
        booster.update_one_iter(ds);
    }

    auto const in = booster.predict_plan_input();
    REQUIRE(in.trees.size() == 8);
    auto const plan =
        cuda_predict_plan(in.trees, mappers, in.learning_rate, in.init_score);
    REQUIRE(plan);

    size_t const       n = ds.n_rows();
    std::vector<float> host(n, 0.0F);
    std::vector<float> dev(n, 0.0F);

    SECTION("the whole ensemble")
    {
        REQUIRE(cuda_predict(*plan, *plane, n, ds.n_features(), 0, dev));
        booster.predict_at_binned(ds, host, 0);
        for (size_t r = 0; r < n; ++r)
        {
            REQUIRE(dev[r] == host[r]);
        }
    }

    SECTION("a truncated ensemble")
    {
        REQUIRE(cuda_predict(*plan, *plane, n, ds.n_features(), 3, dev));
        booster.predict_at_binned(ds, host, 3);
        for (size_t r = 0; r < n; ++r)
        {
            REQUIRE(dev[r] == host[r]);
        }
    }

    SECTION("a plane of the wrong shape declines")
    {
        REQUIRE(!cuda_predict(*plan, *plane, n, ds.n_features() + 1, 0, dev));
    }
}

TEST_CASE("cuda_predict matches the host binned walk for a levelwise model",
          "[cuda][predict][oblivious]")
{
    if (!cuda_available())
    {
        SKIP("device predict needs a usable CUDA device");
    }
    auto const batch   = predict_batch(4096, 5);
    auto const mappers = BinMappers::fit(batch, BinMapperConfig{});
    auto const plane   = cuda_ingest(batch, mappers);
    REQUIRE(plane);
    auto const ds = Dataset::bin(batch, mappers, DataConfig{}, plane);

    ObliviousMseBooster booster{predict_cfg()};
    for (size_t i = 0; i < 8; ++i)
    {
        booster.update_one_iter(ds);
    }

    // The trees the device packs are the plan input's dense equivalents, not
    // the booster's own; the walk they describe must still be the same one.
    auto const in = booster.predict_plan_input();
    REQUIRE(in.keep_alive != nullptr);
    REQUIRE(in.trees.size() == 8);
    auto const plan =
        cuda_predict_plan(in.trees, mappers, in.learning_rate, in.init_score);
    REQUIRE(plan);

    size_t const       n = ds.n_rows();
    std::vector<float> host(n, 0.0F);
    std::vector<float> dev(n, 0.0F);

    SECTION("the whole ensemble")
    {
        REQUIRE(cuda_predict(*plan, *plane, n, ds.n_features(), 0, dev));
        booster.predict_at_binned(ds, host, 0);
        for (size_t r = 0; r < n; ++r)
        {
            REQUIRE(dev[r] == host[r]);
        }
    }

    SECTION("a truncated ensemble")
    {
        REQUIRE(cuda_predict(*plan, *plane, n, ds.n_features(), 3, dev));
        booster.predict_at_binned(ds, host, 3);
        for (size_t r = 0; r < n; ++r)
        {
            REQUIRE(dev[r] == host[r]);
        }
    }
}

TEST_CASE("the device predict plan declines without a device", "[predict]")
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
    REQUIRE(cuda_predict_plan(trees, mappers, 0.1F, 0.0F) == nullptr);
}

TEST_CASE("a host-binned Dataset carries no plane for the device walk", "[predict]")
{
    auto const batch   = predict_batch(64, 2);
    auto const mappers = BinMappers::fit(batch, BinMapperConfig{});
    auto const ds      = Dataset::bin(batch, mappers, DataConfig{});
    REQUIRE(ds.ingest_plane() == nullptr);
}
