#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "bonsai/bin_mappers.hpp"
#include "bonsai/booster.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/detail/column_batch.hpp"
#include "bonsai/objective.hpp"
#include "bonsai/registry/typelists.hpp"
#include "bonsai/sampler.hpp"
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

template <typename G>
using MseBooster = Booster<MSEObjective, G, AllRowsSampler>;

template <typename G>
using LogLossBooster = Booster<LogLossObjective, G, AllRowsSampler>;

detail::ColumnBatch separable_batch()
{
    return detail::ColumnBatch{
        .features      = {{0.0F, 0.1F, 0.9F, 1.0F}},
        .labels        = {-1.0F, -1.0F, +1.0F, +1.0F},
        .weights       = {},
        .feature_names = {"a"},
    };
}

detail::ColumnBatch separable_binary_batch()
{
    return detail::ColumnBatch{
        .features      = {{0.0F, 0.1F, 0.9F, 1.0F}},
        .labels        = {0.0F, 0.0F, 1.0F, 1.0F},
        .weights       = {},
        .feature_names = {"a"},
    };
}

Dataset make_dataset(detail::ColumnBatch const &batch)
{
    BinMappers const mappers = BinMappers::fit(batch, {});
    return Dataset::bin(batch, mappers, {});
}

// Owning row-major copy of ColumnBatch.features (column-major) + mdspan view.
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

// Predict inputs for `separable_batch`: rows 0,1 take a value strictly below
// the inner cut, rows 2,3 strictly above. Avoids hitting any cut value so the
// route is unambiguous regardless of inclusive/exclusive boundary conventions.
RawFeatures separable_raw_midpoints()
{
    std::vector<float> const data = {0.05F, 0.05F, 0.95F, 0.95F};
    return RawFeatures{.data = data, .n_rows = 4, .n_features = 1};
}

} // namespace

TEMPLATE_LIST_TEST_CASE("Booster: ctor doesn't allocate per-row state", "[booster][ctor]",
                   Growers)
{
    Config const cfg = tiny_cfg();
    MseBooster<TestType> booster{cfg};
    SUCCEED();
}

TEMPLATE_LIST_TEST_CASE("Booster: predict shape + finite after 1 iter",
                   "[booster][predict][smoke]", Growers)
{
    auto const batch      = separable_batch();
    Dataset const train   = make_dataset(batch);
    RawFeatures const raw = to_raw(batch);

    Config const cfg = tiny_cfg();
    MseBooster<TestType> booster{cfg};
    booster.update_one_iter(train);

    std::vector<float> y_hat(raw.n_rows);
    booster.predict(raw.view(), y_hat);

    CHECK(y_hat.size() == raw.n_rows);
    for (float v : y_hat)
    {
        CHECK(std::isfinite(v));
    }
}

TEMPLATE_LIST_TEST_CASE("Booster: MSE eval decreases monotonically over iters",
                   "[booster][eval][convergence]", Growers)
{
    auto const batch      = separable_batch();
    Dataset const train   = make_dataset(batch);
    RawFeatures const raw = to_raw(batch);
    floats_view const labels{batch.labels};

    Config const cfg = tiny_cfg();
    MseBooster<TestType> booster{cfg};

    booster.update_one_iter(train);
    float const initial_eval = booster.eval(raw.view(), labels);
    float prev_eval          = initial_eval;

    for (int i = 0; i < 4; ++i)
    {
        booster.update_one_iter(train);
        float const cur_eval = booster.eval(raw.view(), labels);
        CHECK(cur_eval <= prev_eval);
        prev_eval = cur_eval;
    }

    CHECK(prev_eval < initial_eval);
}

TEMPLATE_LIST_TEST_CASE("Booster: eval == MSE(predict, labels) by construction",
                   "[booster][eval][predict][contract]", Growers)
{
    auto const batch      = separable_batch();
    Dataset const train   = make_dataset(batch);
    RawFeatures const raw = to_raw(batch);
    floats_view const labels{batch.labels};

    Config const cfg = tiny_cfg();
    MseBooster<TestType> booster{cfg};
    booster.update_one_iter(train);

    float const eval_result = booster.eval(raw.view(), labels);

    std::vector<float> scores(raw.n_rows);
    booster.predict(raw.view(), scores);
    float const recomputed = MSEObjective::eval(scores, labels);

    CHECK(eval_result == Catch::Approx(recomputed).epsilon(1e-6));
}

TEMPLATE_LIST_TEST_CASE("Booster: weights scale grad/hess and shift leaf values",
                   "[booster][update][weights]", Growers)
{
    auto const batch_unw    = separable_batch();
    Dataset const train_unw = make_dataset(batch_unw);

    detail::ColumnBatch batch_w = separable_batch();
    batch_w.weights             = {2.0F, 2.0F, 2.0F, 2.0F};
    Dataset const train_w       = make_dataset(batch_w);

    Config const cfg = tiny_cfg(); // lambda_l2 default = 1.0 ⇒ weights don't cancel.
    MseBooster<TestType> booster_unw{cfg};
    MseBooster<TestType> booster_w{cfg};
    booster_unw.update_one_iter(train_unw);
    booster_w.update_one_iter(train_w);

    RawFeatures const raw = separable_raw_midpoints();
    std::vector<float> pred_unw(raw.n_rows);
    std::vector<float> pred_w(raw.n_rows);
    booster_unw.predict(raw.view(), pred_unw);
    booster_w.predict(raw.view(), pred_w);

    // Unweighted leaf = -G/(H+λ); weighted (w=2) leaf = -2G/(2H+λ).
    // λ>0 ⇒ |weighted| > |unweighted|. Rows 0,1 are in the negative leaf,
    // rows 2,3 in the positive leaf, so weighted prediction is strictly
    // more negative (less) on the left side and strictly more positive
    // (greater) on the right side.
    CHECK(pred_w[0] < pred_unw[0]);
    CHECK(pred_w[1] < pred_unw[1]);
    CHECK(pred_w[2] > pred_unw[2]);
    CHECK(pred_w[3] > pred_unw[3]);
}

TEMPLATE_LIST_TEST_CASE("Booster: predict matches analytic leaf after 1 iter",
                   "[booster][predict][analytic]", Growers)
{
    auto const batch      = separable_batch();
    Dataset const train   = make_dataset(batch);
    RawFeatures const raw = separable_raw_midpoints();

    Config const cfg = tiny_cfg();
    MseBooster<TestType> booster{cfg};
    booster.update_one_iter(train);

    std::vector<float> y_hat(raw.n_rows);
    booster.predict(raw.view(), y_hat);

    // init_score = mean(y) = 0. Grad after init = pred - y = {1,1,-1,-1}, hess =
    // {1,1,1,1}. Single split separates rows {0,1} from {2,3}; left leaf = -2/(2+λ),
    // right = +2/(2+λ). Predict = init + lr * leaf.
    float const lambda = cfg.tree_config.lambda_l2;
    float const lr     = cfg.booster_config.learning_rate;
    float const left   = lr * (-2.0F / (2.0F + lambda));
    float const right  = lr * (+2.0F / (2.0F + lambda));

    CHECK(y_hat[0] == Catch::Approx(left).epsilon(1e-5));
    CHECK(y_hat[1] == Catch::Approx(left).epsilon(1e-5));
    CHECK(y_hat[2] == Catch::Approx(right).epsilon(1e-5));
    CHECK(y_hat[3] == Catch::Approx(right).epsilon(1e-5));
}

TEMPLATE_LIST_TEST_CASE("Booster: LogLoss eval decreases monotonically over iters",
                   "[booster][logloss][convergence]", Growers)
{
    auto const batch      = separable_binary_batch();
    Dataset const train   = make_dataset(batch);
    RawFeatures const raw = to_raw(batch);
    floats_view const labels{batch.labels};

    Config const cfg = tiny_cfg();
    LogLossBooster<TestType> booster{cfg};

    booster.update_one_iter(train);
    float const initial_eval = booster.eval(raw.view(), labels);
    float prev_eval          = initial_eval;

    for (int i = 0; i < 4; ++i)
    {
        booster.update_one_iter(train);
        float const cur_eval = booster.eval(raw.view(), labels);
        CHECK(cur_eval <= prev_eval);
        prev_eval = cur_eval;
    }

    CHECK(prev_eval < initial_eval);
}

TEMPLATE_LIST_TEST_CASE("Booster: LogLoss predict produces raw scores separating classes",
                   "[booster][logloss][predict]", Growers)
{
    auto const batch      = separable_binary_batch();
    Dataset const train   = make_dataset(batch);
    RawFeatures const raw = separable_raw_midpoints();

    Config const cfg = tiny_cfg();
    LogLossBooster<TestType> booster{cfg};
    for (int i = 0; i < 10; ++i)
    {
        booster.update_one_iter(train);
    }

    std::vector<float> y_hat(raw.n_rows);
    booster.predict(raw.view(), y_hat);

    for (float v : y_hat)
    {
        CHECK(std::isfinite(v));
    }
    // Class 0 → negative score, class 1 → positive score.
    CHECK(y_hat[0] < 0.0F);
    CHECK(y_hat[1] < 0.0F);
    CHECK(y_hat[2] > 0.0F);
    CHECK(y_hat[3] > 0.0F);
}

TEMPLATE_LIST_TEST_CASE("Booster: n_iters tracks update count", "[booster][n_iters]",
                   Growers)
{
    auto const batch    = separable_batch();
    Dataset const train = make_dataset(batch);
    Config const cfg    = tiny_cfg();
    MseBooster<TestType> booster{cfg};

    CHECK(booster.n_iters() == 0);
    booster.update_one_iter(train);
    CHECK(booster.n_iters() == 1);
    booster.update_one_iter(train);
    CHECK(booster.n_iters() == 2);
}
