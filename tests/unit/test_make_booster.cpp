#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <utility>
#include <vector>

#include "bonsai/bin_mappers.hpp"
#include "bonsai/booster.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/detail/column_batch.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/objective.hpp"
#include "bonsai/registry/make_booster.hpp"
#include "bonsai/sampler.hpp"
#include "bonsai/split.hpp"
#include "bonsai/types.hpp"

using namespace bonsai; // NOLINT

namespace
{

Config tiny_cfg()
{
    Config cfg{};
    cfg.tree_config.min_data_in_leaf = 0;
    cfg.tree_config.min_child_hess   = 0.0F;
    cfg.tree_config.max_depth        = 2;
    return cfg;
}

detail::ColumnBatch separable_batch()
{
    return detail::ColumnBatch{
        .features      = {{0.0F, 0.1F, 0.9F, 1.0F}},
        .labels        = {-1.0F, -1.0F, +1.0F, +1.0F},
        .weights       = {},
        .feature_names = {"a"},
    };
}

Dataset make_dataset(detail::ColumnBatch const &batch)
{
    BinMappers const mappers = BinMappers::fit(batch, {});
    return Dataset::bin(batch, mappers, {});
}

struct RawFeatures
{
    std::vector<float> data;
    size_t n_rows;
    size_t n_features;
    features_view view() const
    {
        return features_view{data.data(), n_rows, n_features};
    }
};

RawFeatures to_raw(detail::ColumnBatch const &batch)
{
    size_t const n_features = batch.features.size();
    size_t const n_rows     = n_features == 0 ? 0 : batch.features[0].size();
    std::vector<float> data(n_rows * n_features);
    for (size_t f = 0; f < n_features; ++f)
    {
        for (size_t r = 0; r < n_rows; ++r)
        {
            data[(r * n_features) + f] = batch.features[f][r];
        }
    }
    return RawFeatures{
        .data = std::move(data), .n_rows = n_rows, .n_features = n_features};
}

} // namespace

TEST_CASE("make_booster: default config returns non-null IBooster",
          "[registry][make_booster]")
{
    Config const cfg   = tiny_cfg();
    auto const booster = make_booster(cfg);
    REQUIRE(booster != nullptr);
}

TEST_CASE("make_booster: returned booster runs update_one_iter + predict",
          "[registry][make_booster][smoke]")
{
    auto const batch      = separable_batch();
    Dataset const train   = make_dataset(batch);
    RawFeatures const raw = to_raw(batch);

    Config const cfg   = tiny_cfg();
    auto const booster = make_booster(cfg);
    booster->update_one_iter(train);

    std::vector<float> y_hat(raw.n_rows);
    booster->predict(raw.view(), y_hat);
    CHECK(y_hat.size() == raw.n_rows);
    CHECK(booster->n_iters() == 1);
}

TEST_CASE("make_booster: unknown objective name throws UnknownImplError",
          "[registry][make_booster][error]")
{
    Config cfg                  = tiny_cfg();
    cfg.dispatch.objective_name = "no_such_objective";
    CHECK_THROWS_AS(make_booster(cfg), UnknownImplError);
}

TEST_CASE("make_booster: unknown sampler name throws UnknownImplError",
          "[registry][make_booster][error]")
{
    Config cfg                = tiny_cfg();
    cfg.dispatch.sampler_name = "goss";
    CHECK_THROWS_AS(make_booster(cfg), UnknownImplError);
}

TEST_CASE("make_booster: MSE/Depthwise/AllRows parity with direct instantiation",
          "[registry][make_booster][parity]")
{
    auto const batch      = separable_batch();
    Dataset const train   = make_dataset(batch);
    RawFeatures const raw = to_raw(batch);

    Config const cfg = tiny_cfg();

    using DirectBooster =
        Booster<MSEObjective, DepthwiseGrower<HistogramNodeSplitFinder>,
                AllRowsSampler>;
    DirectBooster direct{cfg};

    auto const dispatched = make_booster(cfg);

    int constexpr n_iters = 5;
    for (int i = 0; i < n_iters; ++i)
    {
        direct.update_one_iter(train);
        dispatched->update_one_iter(train);
    }

    std::vector<float> y_direct(raw.n_rows);
    std::vector<float> y_dispatched(raw.n_rows);
    direct.predict(raw.view(), y_direct);
    dispatched->predict(raw.view(), y_dispatched);

    REQUIRE(y_direct.size() == y_dispatched.size());
    for (size_t i = 0; i < y_direct.size(); ++i)
    {
        CHECK(y_direct[i] == Catch::Approx(y_dispatched[i]).epsilon(1e-6));
    }
}

TEST_CASE("make_booster: LogLoss/Depthwise/AllRows parity with direct instantiation",
          "[registry][make_booster][parity][logloss]")
{
    detail::ColumnBatch const batch{
        .features      = {{0.0F, 0.1F, 0.9F, 1.0F}},
        .labels        = {0.0F, 0.0F, 1.0F, 1.0F},
        .weights       = {},
        .feature_names = {"a"},
    };
    Dataset const train   = make_dataset(batch);
    RawFeatures const raw = to_raw(batch);

    Config cfg                  = tiny_cfg();
    cfg.dispatch.objective_name = "logloss";

    using DirectBooster =
        Booster<LogLossObjective, DepthwiseGrower<HistogramNodeSplitFinder>,
                AllRowsSampler>;
    DirectBooster direct{cfg};

    auto const dispatched = make_booster(cfg);

    int constexpr n_iters = 5;
    for (int i = 0; i < n_iters; ++i)
    {
        direct.update_one_iter(train);
        dispatched->update_one_iter(train);
    }

    std::vector<float> y_direct(raw.n_rows);
    std::vector<float> y_dispatched(raw.n_rows);
    direct.predict(raw.view(), y_direct);
    dispatched->predict(raw.view(), y_dispatched);

    REQUIRE(y_direct.size() == y_dispatched.size());
    for (size_t i = 0; i < y_direct.size(); ++i)
    {
        CHECK(y_direct[i] == Catch::Approx(y_dispatched[i]).epsilon(1e-6));
    }
}
