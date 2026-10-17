#include "bonsai/bin_mappers.hpp"
#include "bonsai/booster.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/config/errors.hpp"
#include "bonsai/cuda/histogram_engine.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/io/model.hpp"
#include "bonsai/registry/make_booster.hpp"
#include "bonsai/tree.hpp"
#include "bonsai/types.hpp"
#include "test_grower_helpers.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

using namespace bonsai; // NOLINT

namespace
{
constexpr size_t k_rows     = 2048;
constexpr size_t k_features = 4;
constexpr size_t k_targets  = 3;
constexpr size_t k_rounds   = 4;

struct Table
{
    std::vector<float> X;
    std::vector<float> block;
    features_view      view() const
    {
        return features_view{X.data(), k_rows, k_features};
    }
    std::vector<float> column(size_t k) const
    {
        std::vector<float> out(k_rows);
        for (size_t i = 0; i < k_rows; ++i)
        {
            out[i] = block[(i * k_targets) + k];
        }
        return out;
    }
};

Table make_table()
{
    Table    t;
    uint32_t state = 12345U;
    auto     next  = [&]
    {
        state = (state * 1664525U) + 1013904223U;
        return static_cast<float>(state >> 8) / static_cast<float>(1U << 24);
    };
    t.X.resize(k_rows * k_features);
    for (float &x : t.X)
    {
        x = next();
    }
    t.block.resize(k_rows * k_targets);
    for (size_t i = 0; i < k_rows; ++i)
    {
        float const a                = t.X[i * k_features];
        float const b                = t.X[(i * k_features) + 1];
        float const c                = t.X[(i * k_features) + 2];
        t.block[(i * k_targets) + 0] = (2.0F * a) + b + (0.1F * next());
        t.block[(i * k_targets) + 1] = std::sin(6.0F * b) - c + (0.1F * next());
        t.block[(i * k_targets) + 2] = (a > 0.5F ? 1.0F : -1.0F) * c + (0.1F * next());
    }
    return t;
}

Config config_for(std::string_view objective, std::string_view grower,
                  std::string_view sampler, uint32_t n_targets)
{
    Config cfg{};
    cfg.dispatch.objective_name      = std::string{objective};
    cfg.dispatch.grower_name         = std::string{grower};
    cfg.dispatch.sampler_name        = std::string{sampler};
    cfg.objective.n_targets          = n_targets;
    cfg.tree_config.max_depth        = 3;
    cfg.tree_config.max_leaves       = 8;
    cfg.tree_config.min_data_in_leaf = 2;
    cfg.booster_config.learning_rate = 0.3F;
    return cfg;
}

BinMappers mappers_for(Table const &t)
{
    return BinMappers::fit(t.view(), {"a", "b", "c", "d"}, BinMapperConfig{});
}

std::unique_ptr<ITrainableBooster> fit(Config const &cfg, Dataset const &train)
{
    auto booster = make_booster(cfg);
    for (size_t r = 0; r < k_rounds; ++r)
    {
        booster->update_one_iter(train);
    }
    return booster;
}

std::vector<float> predictions(IBooster const &booster, Table const &t, size_t width)
{
    std::vector<float> out(k_rows * width);
    booster.predict(t.view(), out);
    return out;
}

void require_matches_single_fits(std::string_view grower, std::string_view objective,
                                 bool singles_through_host_objective)
{
    test::ScopedEnv const hatch = test::host_objective(singles_through_host_objective);
    Table const           t     = make_table();
    BinMappers const      mappers = mappers_for(t);
    Dataset const         wide    = Dataset::bin(t.view(), t.block, mappers, {});
    REQUIRE(wide.n_targets() == k_targets);

    auto const multi = fit(config_for(objective, grower, "all_rows", k_targets), wide);
    REQUIRE(multi->n_iters() == k_rounds);
    REQUIRE(multi->n_trees() == k_rounds * k_targets);
    REQUIRE(multi->score_width() == k_targets);
    REQUIRE(multi->prediction_width() == k_targets);
    std::vector<float> const got = predictions(*multi, t, k_targets);

    for (size_t k = 0; k < k_targets; ++k)
    {
        std::vector<float> const column = t.column(k);
        Dataset const            narrow = Dataset::bin(t.view(), column, mappers, {});
        auto const single = fit(config_for(objective, grower, "all_rows", 1), narrow);
        std::vector<float> const want = predictions(*single, t, 1);
        for (size_t i = 0; i < k_rows; ++i)
        {
            REQUIRE(got[(i * k_targets) + k] == want[i]);
        }
    }
}
} // namespace

// INVARIANT: multi-target-fit-matches-single-target-fits
// A multi-target regression grows one tree per target per round from that
// target's column of the label block and score matrix, through the same
// objective, renewal and score update a single-target fit runs, so with
// every row selected its prediction for target k equals, bit for bit, the
// prediction of a single-target fit on column k alone. Any engine or
// objective change that holds the three model hashes therefore holds the
// multi-target fit too.
TEST_CASE("MultiOutputBooster: a K-target fit matches K single-target fits bit for bit",
          "[booster][multi_target][fit]")
{
    auto const grower =
        GENERATE(as<std::string_view>{}, "depthwise", "levelwise", "leafwise");
    auto const objective = GENERATE(as<std::string_view>{}, "mse", "huber", "quantile");
    CAPTURE(grower, objective);
    require_matches_single_fits(grower, objective, false);
}

// A device fit is bit-reproducible on one device and build, so the equality
// holds on the CUDA growers on both of their paths: under the host-objective
// hatch every fit takes the host round trip (host gradients up, leaf ids
// down), and with the hatch unset every fit arms the resident objective,
// whose gradients differ from the host's at rounding level
// (test_cuda_resident.cpp pins that gap), the K-target fit arming once with
// the whole block and growing tree k from the device's column k.
TEST_CASE("MultiOutputBooster: a K-target fit on a CUDA grower matches K device fits",
          "[cuda][booster][multi_target][fit]")
{
    if (!cuda_available())
    {
        SKIP("no usable CUDA device");
    }
    auto const grower    = GENERATE(as<std::string_view>{}, "cuda_depthwise",
                                    "cuda_levelwise", "cuda_leafwise");
    auto const objective = GENERATE(as<std::string_view>{}, "mse", "huber");
    CAPTURE(grower, objective);
    require_matches_single_fits(grower, objective, true);
}

TEST_CASE("MultiOutputBooster: a K-target resident fit matches K resident fits",
          "[cuda][resident][booster][multi_target][fit]")
{
    if (!cuda_available())
    {
        SKIP("no usable CUDA device");
    }
    auto const grower    = GENERATE(as<std::string_view>{}, "cuda_depthwise",
                                    "cuda_levelwise", "cuda_leafwise");
    auto const objective = GENERATE(as<std::string_view>{}, "mse", "huber", "quantile");
    CAPTURE(grower, objective);
    require_matches_single_fits(grower, objective, false);
}

TEST_CASE("MultiOutputBooster: a K-target model hands the device one plan per target",
          "[booster][multi_target][predict]")
{
    Table const      t       = make_table();
    BinMappers const mappers = mappers_for(t);
    Dataset const    wide    = Dataset::bin(t.view(), t.block, mappers, {});
    auto const grower = GENERATE(as<std::string_view>{}, "depthwise", "levelwise");
    CAPTURE(grower);
    auto const multi = fit(config_for("mse", grower, "all_rows", k_targets), wide);
    std::vector<float> const want = predictions(*multi, t, k_targets);

    for (size_t k = 0; k < k_targets; ++k)
    {
        DevicePlanInput const in = multi->device_plan_input(k);
        REQUIRE(in.trees.size() == k_rounds);
        REQUIRE(in.epoch == multi->device_plan_input(0).epoch);
        std::vector<float> raw(k_rows, 0.0F);
        for (DenseTree const &tree : in.trees)
        {
            tree.predict(t.view(), raw);
        }
        for (size_t i = 0; i < k_rows; ++i)
        {
            float const got = in.init_score + (in.learning_rate * raw[i]);
            REQUIRE(got == want[(i * k_targets) + k]);
        }
    }
    CHECK(multi->device_plan_input(0).trees.data() !=
          multi->device_plan_input(1).trees.data());
}

TEST_CASE("MultiOutputBooster: eval is the objective's loss pooled over the block",
          "[booster][multi_target]")
{
    Table const      t       = make_table();
    BinMappers const mappers = mappers_for(t);
    Dataset const    wide    = Dataset::bin(t.view(), t.block, mappers, {});
    auto const multi = fit(config_for("mse", "depthwise", "all_rows", k_targets), wide);
    std::vector<float> const got = predictions(*multi, t, k_targets);
    double                   sum = 0.0;
    for (size_t i = 0; i < got.size(); ++i)
    {
        double const d = static_cast<double>(got[i]) - static_cast<double>(t.block[i]);
        sum += d * d;
    }
    float const pooled = static_cast<float>(sum / static_cast<double>(got.size()));
    CHECK(multi->eval(t.view(), t.block) == Catch::Approx(pooled).epsilon(1e-5));
}

TEST_CASE("MultiOutputBooster: a saved K-target model loads and predicts the same",
          "[booster][multi_target][io]")
{
    Table const      t       = make_table();
    BinMappers const mappers = mappers_for(t);
    Dataset const    wide    = Dataset::bin(t.view(), t.block, mappers, {});
    Config const     cfg     = config_for("huber", "levelwise", "all_rows", k_targets);
    auto const       multi   = fit(cfg, wide);

    auto const bytes  = io::save_booster_bytes(*multi, mappers, cfg);
    auto const loaded = io::load_booster_bytes(bytes, "");
    REQUIRE(loaded.booster->n_iters() == k_rounds);
    REQUIRE(loaded.booster->score_width() == k_targets);
    CHECK(predictions(*loaded.booster, t, k_targets) ==
          predictions(*multi, t, k_targets));

    SECTION("and a file saved before the knob existed reads as one target")
    {
        auto const old = io::load_booster(std::string{BONSAI_TESTS_DATA_DIR} +
                                          "/fixture_v7_regressor.bonsai");
        CHECK(old.cfg.objective.n_targets == 1);
        CHECK(old.booster->score_width() == 1);
        CHECK(loaded.cfg.objective.n_targets == k_targets);
    }

    SECTION("and keeps training as the same model")
    {
        loaded.booster->update_one_iter(wide);
        multi->update_one_iter(wide);
        CHECK(predictions(*loaded.booster, t, k_targets) ==
              predictions(*multi, t, k_targets));
    }
}

TEST_CASE("MultiOutputBooster: a sampled K-target fit is deterministic under a seed",
          "[booster][multi_target]")
{
    Table const      t       = make_table();
    BinMappers const mappers = mappers_for(t);
    Dataset const    wide    = Dataset::bin(t.view(), t.block, mappers, {});
    Config           cfg     = config_for("mse", "leafwise", "bernoulli", k_targets);
    cfg.sampler.subsample    = 0.7F;
    auto const first         = fit(cfg, wide);
    auto const second        = fit(cfg, wide);
    CHECK(predictions(*first, t, k_targets) == predictions(*second, t, k_targets));
}

TEST_CASE("MultiOutputBooster: the route refuses what it cannot serve",
          "[booster][multi_target][edge]")
{
    Table const      t       = make_table();
    BinMappers const mappers = mappers_for(t);

    SECTION("a classification objective with a target count")
    {
        CHECK_THROWS_AS(make_booster(config_for("logloss", "depthwise", "all_rows", 2)),
                        ConfigError);
        CHECK_THROWS_AS(make_booster(config_for("softmax", "depthwise", "all_rows", 2)),
                        ConfigError);
    }

    SECTION("a label block narrower than the configured targets")
    {
        Dataset const narrow = Dataset::bin(t.view(), t.column(0), mappers, {});
        auto          booster =
            make_booster(config_for("mse", "depthwise", "all_rows", k_targets));
        CHECK_THROWS_WITH(booster->update_one_iter(narrow),
                          Catch::Matchers::ContainsSubstring("1 targets per row"));
    }
}
