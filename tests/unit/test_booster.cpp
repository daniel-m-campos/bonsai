#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <utility>
#include <vector>

#include "bonsai/bin_mappers.hpp"
#include "bonsai/booster.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/detail/column_batch.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/io/model.hpp"
#include "bonsai/multiclass_booster.hpp"
#include "bonsai/objective.hpp"
#include "bonsai/registry/typelists.hpp"
#include "bonsai/row_view.hpp"
#include "bonsai/sampler.hpp"
#include "bonsai/types.hpp"
#include "test_grower_helpers.hpp"

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

template <typename G> using MseBooster = Booster<MSEObjective, G, AllRowsSampler>;

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
    size_t             n_rows;
    size_t             n_features;
    features_view      view() const
    {
        return features_view{data.data(), n_rows, n_features};
    }
};

RawFeatures to_raw(detail::ColumnBatch const &batch)
{
    size_t const       n_features = batch.features.size();
    size_t const       n_rows     = n_features == 0 ? 0 : batch.features[0].size();
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

using HostGrowers = TypeList<DepthwiseGrower<>, LeafwiseGrower<>, ObliviousGrower<>>;

constexpr size_t k_stepped_rows = 16;

// Sixteen rows on one feature whose labels rise in steps a depth-2 tree can
// cut between. A binary objective reads the two highest steps as positives.
template <typename Obj> detail::ColumnBatch stepped_batch()
{
    constexpr std::array k_steps{0.0F, 0.0F, 1.0F, 1.0F, 3.0F, 3.0F, 6.0F, 6.0F};
    detail::ColumnBatch  batch{
         .features      = {std::vector<float>(k_stepped_rows)},
         .labels        = std::vector<float>(k_stepped_rows),
         .weights       = {},
         .feature_names = {"a"},
    };
    for (size_t r = 0; r < k_stepped_rows; ++r)
    {
        float const step     = k_steps[r / 2];
        batch.features[0][r] = static_cast<float>(r);
        batch.labels[r]      = std::same_as<Obj, LogLossObjective>
                                   ? static_cast<float>(step > 1.0F)
                                   : step;
    }
    return batch;
}

template <typename Obj, typename G>
std::vector<float> fitted_predictions(Config const &cfg, Dataset const &train,
                                      detail::ColumnBatch const &batch)
{
    auto const                      raw = to_raw(batch);
    Booster<Obj, G, AllRowsSampler> booster{cfg};
    for (int round = 0; round < 3; ++round)
    {
        booster.update_one_iter(train);
    }
    std::vector<float> pred(raw.n_rows);
    booster.predict(raw.view(), pred);
    return pred;
}

template <typename Obj, typename G>
std::vector<float> fitted_predictions(Config const              &cfg,
                                      detail::ColumnBatch const &batch)
{
    return fitted_predictions<Obj, G>(cfg, make_dataset(batch), batch);
}

Config stepped_cfg()
{
    Config cfg                       = tiny_cfg();
    cfg.booster_config.learning_rate = 0.5F;
    cfg.objective.quantile_alpha     = 0.8F;
    return cfg;
}

} // namespace

TEMPLATE_LIST_TEST_CASE("Booster: ctor doesn't allocate per-row state",
                        "[booster][ctor]", Growers)
{
    Config const         cfg = tiny_cfg();
    MseBooster<TestType> booster{cfg};
    SUCCEED();
}

TEMPLATE_LIST_TEST_CASE("Booster: predict shape + finite after 1 iter",
                        "[booster][predict][smoke]", Growers)
{
    test::skip_without_cuda<TestType>();
    auto const        batch = separable_batch();
    Dataset const     train = make_dataset(batch);
    RawFeatures const raw   = to_raw(batch);

    Config const         cfg = tiny_cfg();
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
    test::skip_without_cuda<TestType>();
    auto const        batch = separable_batch();
    Dataset const     train = make_dataset(batch);
    RawFeatures const raw   = to_raw(batch);
    floats_view const labels{batch.labels};

    Config const         cfg = tiny_cfg();
    MseBooster<TestType> booster{cfg};

    booster.update_one_iter(train);
    float const initial_eval = booster.eval(raw.view(), labels);
    float       prev_eval    = initial_eval;

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
    test::skip_without_cuda<TestType>();
    auto const        batch = separable_batch();
    Dataset const     train = make_dataset(batch);
    RawFeatures const raw   = to_raw(batch);
    floats_view const labels{batch.labels};

    Config const         cfg = tiny_cfg();
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
    test::skip_without_cuda<TestType>();
    auto const    batch_unw = separable_batch();
    Dataset const train_unw = make_dataset(batch_unw);

    detail::ColumnBatch batch_w = separable_batch();
    batch_w.weights             = {2.0F, 2.0F, 2.0F, 2.0F};
    Dataset const train_w       = make_dataset(batch_w);

    Config const cfg = tiny_cfg(); // lambda_l2 default = 1.0 ⇒ weights don't cancel.
    MseBooster<TestType> booster_unw{cfg};
    MseBooster<TestType> booster_w{cfg};
    booster_unw.update_one_iter(train_unw);
    booster_w.update_one_iter(train_w);

    RawFeatures const  raw = separable_raw_midpoints();
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

TEMPLATE_TEST_CASE("Booster: unit weights reproduce the unweighted fit",
                   "[booster][fit][weights]", MSEObjective, LogLossObjective,
                   PoissonObjective, MAEObjective, QuantileObjective)
{
    Config const              cfg   = stepped_cfg();
    detail::ColumnBatch const plain = stepped_batch<TestType>();
    detail::ColumnBatch       unit  = plain;
    unit.weights.assign(k_stepped_rows, 1.0F);

    for_each_type<HostGrowers>(
        [&]<typename G>()
        {
            INFO(name_of<G>);
            auto const unweighted = fitted_predictions<TestType, G>(cfg, plain);
            CHECK(unweighted.front() != unweighted.back());
            CHECK(fitted_predictions<TestType, G>(cfg, unit) == unweighted);
        });
}

// INVARIANT: zero-weight-labels-never-reach-the-fit
// The label of a row whose weight is zero never reaches the fit: two fits
// whose labels differ only on such rows predict the same on every row, on
// each host grower and under each objective, for the base score, the splits
// and the renewed leaves alike.
TEMPLATE_TEST_CASE("Booster: a zero-weight row's label never reaches the fit",
                   "[booster][fit][weights]", MSEObjective, LogLossObjective,
                   PoissonObjective, MAEObjective, HuberObjective, QuantileObjective)
{
    Config const        cfg  = stepped_cfg();
    detail::ColumnBatch high = stepped_batch<TestType>();
    high.weights.assign(k_stepped_rows, 1.0F);
    detail::ColumnBatch low = high;
    for (size_t r = 1; r < k_stepped_rows; r += 2)
    {
        high.weights[r] = 0.0F;
        low.weights[r]  = 0.0F;
        high.labels[r]  = std::same_as<TestType, LogLossObjective> ? 1.0F : 9.0F;
        low.labels[r]   = 0.0F;
    }

    for_each_type<HostGrowers>(
        [&]<typename G>()
        {
            INFO(name_of<G>);
            auto const fitted = fitted_predictions<TestType, G>(cfg, high);
            CHECK(fitted.front() != fitted.back());
            CHECK(fitted_predictions<TestType, G>(cfg, low) == fitted);
        });
}

TEST_CASE("Booster: weighted MAE fits the weighted medians",
          "[booster][renewal][weights]")
{
    // The weighted median of the labels is 10 where the plain median is 20.
    // The one split leaves residuals {-10, -10, 0} on the left, whose weighted
    // median is 0 where the plain one is -10, and {10, 20, 20, 10} on the
    // right, whose median is 20 either way.
    detail::ColumnBatch const batch{
        .features      = {{0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F, 1.0F}},
        .labels        = {0.0F, 0.0F, 10.0F, 20.0F, 30.0F, 30.0F, 20.0F},
        .weights       = {1.0F, 1.0F, 4.0F, 1.0F, 1.0F, 1.0F, 1.0F},
        .feature_names = {"a"},
    };
    Dataset const train = make_dataset(batch);
    auto const    raw   = to_raw(batch);

    Config cfg                       = tiny_cfg();
    cfg.booster_config.learning_rate = 1.0F;
    cfg.tree_config.lambda_l2        = 1.0F;
    cfg.tree_config.max_depth        = 1;

    for_each_type<HostGrowers>(
        [&]<typename G>()
        {
            INFO(name_of<G>);
            Booster<MAEObjective, G, AllRowsSampler> booster{cfg};
            booster.update_one_iter(train);
            CHECK(booster.init_score() == 10.0F);

            std::vector<float> pred(raw.n_rows);
            booster.predict(raw.view(), pred);
            CHECK(pred == std::vector{10.0F, 10.0F, 10.0F, 30.0F, 30.0F, 30.0F, 30.0F});
        });
}

TEMPLATE_TEST_CASE("Booster: a row outside the view weighs what a zero-weight row does",
                   "[booster][fit][weights]", MSEObjective, MAEObjective,
                   QuantileObjective)
{
    Config const          cfg     = stepped_cfg();
    detail::ColumnBatch   weighed = stepped_batch<TestType>();
    std::vector<row_id_t> viewed;
    for (size_t r = 0; r < k_stepped_rows; ++r)
    {
        weighed.weights.push_back(r < k_stepped_rows / 2 ? 4.0F : 1.0F);
        if (r % 4 != 3)
        {
            viewed.push_back(static_cast<row_id_t>(r));
        }
    }
    detail::ColumnBatch masked = weighed;
    for (size_t r = 3; r < k_stepped_rows; r += 4)
    {
        masked.weights[r] = 0.0F;
    }
    Dataset const view =
        make_dataset(weighed).with_rows(RowView::encode(viewed, k_stepped_rows));

    for_each_type<HostGrowers>(
        [&]<typename G>()
        {
            INFO(name_of<G>);
            auto const on_view = fitted_predictions<TestType, G>(cfg, view, weighed);
            CHECK(on_view.front() != on_view.back());
            CHECK(fitted_predictions<TestType, G>(cfg, masked) == on_view);
        });
}

// INVARIANT: weighted-renewal-runs-on-host
// The device renewal selects over rows counted once each, so a renewal fit
// whose rows carry weights renews on the host, where the weighted selection
// lives. An oblivious tree under a monotone constraint does too. Either one
// closes the resident round.
TEST_CASE("Booster: a weighted renewal fit renews on the host",
          "[booster][renewal][weights]")
{
    std::vector<float> const weights{1.0F, 2.0F};
    std::vector<int> const   unconstrained{0, 0};
    std::vector<int> const   rising{+1, 0};
    SECTION("weights send the renewal of either tree type to the host")
    {
        REQUIRE(renew_leaf_on_host<DenseTree>(true, unconstrained, weights));
        REQUIRE(renew_leaf_on_host<ObliviousTree>(true, unconstrained, weights));
    }
    SECTION("an unweighted renewal leaves the device unless the lattice reprojects")
    {
        REQUIRE_FALSE(renew_leaf_on_host<DenseTree>(true, unconstrained, {}));
        REQUIRE_FALSE(renew_leaf_on_host<ObliviousTree>(true, unconstrained, {}));
        REQUIRE_FALSE(renew_leaf_on_host<DenseTree>(true, rising, {}));
        REQUIRE(renew_leaf_on_host<ObliviousTree>(true, rising, {}));
    }
    SECTION("an objective that renews nothing never asks for the host")
    {
        REQUIRE_FALSE(renew_leaf_on_host<DenseTree>(false, unconstrained, weights));
        REQUIRE_FALSE(renew_leaf_on_host<ObliviousTree>(false, rising, weights));
    }
    SECTION("a renewal on the host closes the resident round")
    {
        REQUIRE_FALSE(resident_round_open(0.0F, true));
    }
}

TEST_CASE("MulticlassBooster: class priors count each row by its weight",
          "[booster][multiclass][weights]")
{
    Config cfg              = tiny_cfg();
    cfg.objective.n_classes = 3;
    detail::ColumnBatch batch{
        .features      = {{0.0F, 0.1F, 0.5F, 0.6F, 0.9F, 1.0F}},
        .labels        = {0.0F, 0.0F, 1.0F, 1.0F, 2.0F, 2.0F},
        .weights       = {},
        .feature_names = {"a"},
    };
    auto const priors_under = [&](std::vector<float> labels, std::vector<float> weights)
    {
        batch.labels  = std::move(labels);
        batch.weights = std::move(weights);
        MulticlassBooster<DepthwiseGrower<>, AllRowsSampler> booster{cfg};
        booster.update_one_iter(make_dataset(batch));
        return booster.init_scores();
    };
    std::vector<float> const labels{0.0F, 0.0F, 1.0F, 1.0F, 2.0F, 2.0F};

    SECTION("a prior is the log of the class's share of the weight")
    {
        auto const priors = priors_under(labels, {3.0F, 0.0F, 1.0F, 1.0F, 1.0F, 2.0F});
        CHECK(priors == std::vector{static_cast<float>(std::log(3.0 / 8.0)),
                                    static_cast<float>(std::log(2.0 / 8.0)),
                                    static_cast<float>(std::log(3.0 / 8.0))});
    }
    SECTION("one constant weight gives the unweighted priors, an empty class included")
    {
        std::vector<float> const two_classes{0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F};
        CHECK(priors_under(two_classes, std::vector<float>(6, 2.0F)) ==
              priors_under(two_classes, {}));
    }
    SECTION("no positive weight is refused")
    {
        CHECK_THROWS_AS(priors_under(labels, std::vector<float>(6, 0.0F)),
                        std::invalid_argument);
    }
}

TEMPLATE_LIST_TEST_CASE("Booster: predict matches analytic leaf after 1 iter",
                        "[booster][predict][analytic]", Growers)
{
    test::skip_without_cuda<TestType>();
    auto const        batch = separable_batch();
    Dataset const     train = make_dataset(batch);
    RawFeatures const raw   = separable_raw_midpoints();

    Config const         cfg = tiny_cfg();
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
    test::skip_without_cuda<TestType>();
    auto const        batch = separable_binary_batch();
    Dataset const     train = make_dataset(batch);
    RawFeatures const raw   = to_raw(batch);
    floats_view const labels{batch.labels};

    Config const             cfg = tiny_cfg();
    LogLossBooster<TestType> booster{cfg};

    booster.update_one_iter(train);
    float const initial_eval = booster.eval(raw.view(), labels);
    float       prev_eval    = initial_eval;

    for (int i = 0; i < 4; ++i)
    {
        booster.update_one_iter(train);
        float const cur_eval = booster.eval(raw.view(), labels);
        CHECK(cur_eval <= prev_eval);
        prev_eval = cur_eval;
    }

    CHECK(prev_eval < initial_eval);
}

TEMPLATE_LIST_TEST_CASE(
    "Booster: LogLoss predict produces raw scores separating classes",
    "[booster][logloss][predict]", Growers)
{
    test::skip_without_cuda<TestType>();
    auto const        batch = separable_binary_batch();
    Dataset const     train = make_dataset(batch);
    RawFeatures const raw   = separable_raw_midpoints();

    Config const             cfg = tiny_cfg();
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
    test::skip_without_cuda<TestType>();
    auto const           batch = separable_batch();
    Dataset const        train = make_dataset(batch);
    Config const         cfg   = tiny_cfg();
    MseBooster<TestType> booster{cfg};

    CHECK(booster.n_iters() == 0);
    booster.update_one_iter(train);
    CHECK(booster.n_iters() == 1);
    booster.update_one_iter(train);
    CHECK(booster.n_iters() == 2);
}

TEST_CASE("Booster: DART trains, stays finite, and differs from plain GBDT",
          "[booster][dart]")
{
    auto const    batch = separable_batch();
    Dataset const train = make_dataset(batch);
    auto const    raw   = to_raw(batch);

    Config cfg                             = tiny_cfg();
    cfg.booster_config.learning_rate       = 0.3F;
    Config dart_cfg                        = cfg;
    dart_cfg.booster_config.dart_drop_rate = 0.5F;

    MseBooster<DepthwiseGrower<>> plain{cfg};
    MseBooster<DepthwiseGrower<>> dart{dart_cfg};
    for (int i = 0; i < 10; ++i)
    {
        plain.update_one_iter(train);
        dart.update_one_iter(train);
    }

    std::vector<float> y_plain(raw.n_rows);
    std::vector<float> y_dart(raw.n_rows);
    plain.predict(raw.view(), y_plain);
    dart.predict(raw.view(), y_dart);

    bool differs = false;
    for (size_t i = 0; i < raw.n_rows; ++i)
    {
        REQUIRE(std::isfinite(y_dart[i]));
        differs |= y_dart[i] != y_plain[i];
    }
    CHECK(differs); // dropout must have fired at rate 0.5 over 10 iters
}

TEST_CASE("Booster: DART is deterministic per seed", "[booster][dart][determinism]")
{
    auto const    batch = separable_batch();
    Dataset const train = make_dataset(batch);
    auto const    raw   = to_raw(batch);

    Config cfg                        = tiny_cfg();
    cfg.booster_config.dart_drop_rate = 0.5F;
    cfg.booster_config.random_seed    = 123;

    auto run = [&]
    {
        MseBooster<DepthwiseGrower<>> b{cfg};
        for (int i = 0; i < 8; ++i)
        {
            b.update_one_iter(train);
        }
        std::vector<float> y(raw.n_rows);
        b.predict(raw.view(), y);
        return y;
    };
    CHECK(run() == run());
}

// INVARIANT: random-seed-varies-feature-draws
// The booster seeds the grower's feature_fraction draws from random_seed and
// feature_seed together, so two fits that differ only in random_seed draw
// different columns under the default all_rows sampler, and a fit with both
// fixed is reproducible. Before this pin random_seed reached the row sampler
// only, and a seed set of five fits under all_rows was five copies.
TEST_CASE("Booster: random_seed varies the feature_fraction draws under all_rows",
          "[booster][colsample][determinism]")
{
    detail::ColumnBatch const batch{
        .features      = {{0.0F, 0.1F, 0.2F, 0.3F, 0.7F, 0.8F, 0.9F, 1.0F},
                          {0.0F, 1.0F, 0.0F, 1.0F, 0.0F, 1.0F, 0.0F, 1.0F},
                          {1.0F, 0.0F, 1.0F, 0.0F, 1.0F, 0.0F, 1.0F, 0.0F},
                          {0.5F, 0.2F, 0.9F, 0.1F, 0.6F, 0.3F, 0.8F, 0.4F}},
        .labels        = {-1.0F, -1.0F, -1.0F, -1.0F, +1.0F, +1.0F, +1.0F, +1.0F},
        .weights       = {},
        .feature_names = {"a", "b", "c", "d"},
    };
    Dataset const train = make_dataset(batch);
    auto const    raw   = to_raw(batch);

    Config cfg                       = tiny_cfg();
    cfg.booster_config.learning_rate = 0.3F;
    cfg.tree_config.feature_fraction = 0.25F;
    cfg.tree_config.feature_seed     = 7;

    auto run = [&](uint32_t random_seed)
    {
        cfg.booster_config.random_seed = random_seed;
        MseBooster<DepthwiseGrower<>> b{cfg};
        for (int i = 0; i < 8; ++i)
        {
            b.update_one_iter(train);
        }
        std::vector<float> y(raw.n_rows);
        b.predict(raw.view(), y);
        return y;
    };
    SECTION("two random seeds draw different columns")
    {
        CHECK(run(1) != run(2));
    }
    SECTION("one random seed refits the same model")
    {
        CHECK(run(1) == run(1));
    }
}

TEST_CASE("Booster: feature_importance accumulates gain and split counts",
          "[booster][importance]")
{
    auto const    batch = separable_batch();
    Dataset const train = make_dataset(batch);

    Config                        cfg = tiny_cfg();
    MseBooster<DepthwiseGrower<>> b{cfg};
    for (int i = 0; i < 5; ++i)
    {
        b.update_one_iter(train);
    }

    auto const gain  = b.feature_importance(ImportanceType::gain);
    auto const split = b.feature_importance(ImportanceType::split);
    REQUIRE(!gain.empty());
    REQUIRE(!split.empty());
    // Single-feature dataset: everything lands on feature 0.
    CHECK(gain[0] > 0.0);
    CHECK(split[0] >= 5.0); // at least the root split of each of 5 trees
    // Split count equals the total number of internal nodes across trees.
    double n_internal = 0;
    for (auto const &t : b.trees())
    {
        for (auto const &n : t.nodes())
        {
            n_internal += DenseTree::is_leaf(n) ? 0 : 1;
        }
    }
    CHECK(split[0] == n_internal);
}

TEST_CASE("Booster: MAE leaf renewal sets leaves to residual medians",
          "[booster][renewal]")
{
    // 7 rows, feature separates labels {-5 x3 | +5 x4}. init = median = 5;
    // left residuals are all -10, right all 0. With renewal and lr = 1, one
    // iteration reproduces the labels exactly: leaf = median(residuals).
    // The plain Newton step would give left = -G/(H+lambda) = 3/4 instead.
    detail::ColumnBatch batch{
        .features      = {{0.0F, 0.1F, 0.2F, 1.0F, 1.1F, 1.2F, 1.3F}},
        .labels        = {-5.0F, -5.0F, -5.0F, 5.0F, 5.0F, 5.0F, 5.0F},
        .weights       = {},
        .feature_names = {"a"},
    };
    Dataset const train = make_dataset(batch);
    auto const    raw   = to_raw(batch);

    Config cfg                       = tiny_cfg();
    cfg.booster_config.learning_rate = 1.0F;
    cfg.tree_config.lambda_l2        = 1.0F;

    Booster<MAEObjective, DepthwiseGrower<>, AllRowsSampler> b{cfg};
    b.update_one_iter(train);

    std::vector<float> pred(raw.n_rows);
    b.predict(raw.view(), pred);
    for (size_t i = 0; i < raw.n_rows; ++i)
    {
        CHECK(pred[i] == Catch::Approx(batch.labels[i]).margin(1e-5));
    }
}

template <typename Obj, typename G>
using RenewingBooster = Booster<Obj, G, AllRowsSampler>;

// INVARIANT: renewal-keeps-monotone
// A depthwise or leafwise fit under a monotone constraint stays ordered by
// that feature after leaf renewal. Renewal replaces each leaf's Newton step
// with a residual statistic (median, Huber mean, quantile) that can reverse
// the order the growth-time fence enforced, so the renewed value is clamped
// back into the fence the grower left for that leaf.
TEMPLATE_TEST_CASE("Booster: leaf renewal keeps a monotone constraint",
                   "[booster][renewal][monotone][invariant]",
                   (RenewingBooster<MAEObjective, DepthwiseGrower<>>),
                   (RenewingBooster<MAEObjective, LeafwiseGrower<>>),
                   (RenewingBooster<HuberObjective, DepthwiseGrower<>>),
                   (RenewingBooster<HuberObjective, LeafwiseGrower<>>),
                   (RenewingBooster<QuantileObjective, DepthwiseGrower<>>),
                   (RenewingBooster<QuantileObjective, LeafwiseGrower<>>) )
{
    // Eight rows on one feature, one split, lr = 1. init = 3, so the split at
    // x < 2 leaves residuals {1, -2} on the left (gradient sum 0, Newton leaf
    // 0) and {1, 0, 0, 0, 1, 0} on the right (gradient sum -2 for MAE and
    // Huber, -3 for the quantile, whose tie rule charges a zero residual as
    // -alpha; Newton leaf 2/7 or 3/7): ordered, so the split passes the +1
    // veto. The renewed values are not ordered: medians 1 > 0, Huber means
    // 0.5 > 1/3, quantiles 1 > 0.
    detail::ColumnBatch batch{
        .features      = {{0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F}},
        .labels        = {4.0F, 1.0F, 4.0F, 3.0F, 3.0F, 3.0F, 4.0F, 3.0F},
        .weights       = {},
        .feature_names = {"a"},
    };
    Dataset const train = make_dataset(batch);
    auto const    raw   = to_raw(batch);

    Config cfg                           = tiny_cfg();
    cfg.tree_config.max_depth            = 1;
    cfg.tree_config.lambda_l2            = 1.0F;
    cfg.tree_config.monotone_constraints = {+1};
    cfg.booster_config.learning_rate     = 1.0F;

    auto predict_rows = [&](Config const &c)
    {
        TestType b{c};
        b.update_one_iter(train);
        std::vector<float> pred(raw.n_rows);
        b.predict(raw.view(), pred);
        return pred;
    };

    Config unconstrained = cfg;
    unconstrained.tree_config.monotone_constraints.clear();
    auto const free_pred = predict_rows(unconstrained);
    CHECK(free_pred.front() > free_pred.back());

    auto const pred = predict_rows(cfg);
    for (size_t i = 1; i < pred.size(); ++i)
    {
        CHECK(pred[i - 1] <= pred[i]);
    }
}

TEST_CASE("Booster: predict_at, staged, and leaf predictions are consistent",
          "[booster][predict_extras]")
{
    auto const    batch = separable_batch();
    Dataset const train = make_dataset(batch);
    auto const    raw   = to_raw(batch);

    Config                        cfg = tiny_cfg();
    MseBooster<DepthwiseGrower<>> b{cfg};
    for (int i = 0; i < 6; ++i)
    {
        b.update_one_iter(train);
    }
    size_t const n = raw.n_rows;
    size_t const k = b.n_iters();
    CHECK(b.n_trees() == k); // one tree per round for a width-1 objective

    std::vector<float> full(n);
    std::vector<float> at_all(n);
    std::vector<float> at_2(n);
    b.predict(raw.view(), full);
    b.predict_at(raw.view(), at_all, 0);
    b.predict_at(raw.view(), at_2, 2);
    CHECK(full == at_all);
    CHECK(full != at_2);

    std::vector<float> staged(k * n);
    b.predict_staged(raw.view(), staged);
    for (size_t i = 0; i < n; ++i)
    {
        CHECK(staged[((k - 1) * n) + i] == full[i]); // last stage == full
        CHECK(staged[(1 * n) + i] == at_2[i]);       // stage 2 == predict_at(2)
    }

    std::vector<node_id_t> leaves(n * k);
    b.predict_leaf(raw.view(), leaves);
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t t = 0; t < k; ++t)
        {
            auto const id = leaves[(i * k) + t];
            REQUIRE(id < b.trees()[t].nodes().size());
            CHECK(DenseTree::is_leaf(b.trees()[t].nodes()[id]));
        }
    }

    auto const text = b.dump(std::vector<std::string>{"a"});
    CHECK(text.find("tree 0:") != std::string::npos);
    CHECK(text.find("leaf=") != std::string::npos);
    CHECK(text.find('a') != std::string::npos);
}

TEST_CASE("MulticlassBooster: predict_leaf fills one column per tree",
          "[booster][multiclass][predict_extras]")
{
    detail::ColumnBatch batch{
        .features      = {{0.0F, 0.1F, 0.5F, 0.6F, 0.9F, 1.0F}},
        .labels        = {0.0F, 0.0F, 1.0F, 1.0F, 2.0F, 2.0F},
        .weights       = {},
        .feature_names = {"a"},
    };
    Dataset const train = make_dataset(batch);
    auto const    raw   = to_raw(batch);

    Config cfg              = tiny_cfg();
    cfg.objective.n_classes = 3;

    MulticlassBooster<DepthwiseGrower<>, AllRowsSampler> b{cfg};
    for (int i = 0; i < 5; ++i)
    {
        b.update_one_iter(train);
    }
    // One tree per class per round, so the leaf columns are trees, not rounds.
    REQUIRE(b.n_iters() == 5);
    REQUIRE(b.n_trees() == 15);

    size_t const           n = raw.n_rows;
    size_t const           k = b.n_trees();
    std::vector<node_id_t> leaves(n * k);
    b.predict_leaf(raw.view(), leaves);
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t t = 0; t < k; ++t)
        {
            auto const id = leaves[(i * k) + t];
            REQUIRE(id < b.trees()[t].nodes().size());
            CHECK(DenseTree::is_leaf(b.trees()[t].nodes()[id]));
        }
    }
}

// INVARIANT: row-list-follows-the-fit
// The row list a round trains on belongs to the fit it was materialized for,
// keyed on the fit's minted token and never on its size: a second fit on a
// same-sized view of different rows grows the tree those rows grow, not the
// previous view's.
TEMPLATE_LIST_TEST_CASE("Booster: a same-sized view of other rows refills the row list",
                        "[booster][view][invariant]", Growers)
{
    test::skip_without_cuda<TestType>();
    detail::ColumnBatch const batch{
        .features      = {{0.5F, 0.5F, 0.5F, 0.5F, 0.0F, 0.1F, 0.9F, 1.0F}},
        .labels        = {0.0F, 0.0F, 0.0F, 0.0F, -1.0F, -1.0F, 1.0F, 1.0F},
        .weights       = {},
        .feature_names = {"a"},
    };
    Dataset const               full = make_dataset(batch);
    std::vector<row_id_t> const flat{0, 1, 2, 3};
    std::vector<row_id_t> const separable{4, 5, 6, 7};
    Dataset const               flat_view =
        full.with_rows(RowView::encode(flat, batch.labels.size()));
    Dataset const separable_view =
        full.with_rows(RowView::encode(separable, batch.labels.size()));
    Config const cfg = tiny_cfg();

    MseBooster<TestType> continued{cfg};
    continued.update_one_iter(flat_view);
    continued.update_one_iter(separable_view);
    MseBooster<TestType> fresh{cfg};
    fresh.update_one_iter(separable_view);

    REQUIRE(continued.n_trees() == 2);
    RawFeatures const  raw = to_raw(batch);
    std::vector<float> second(raw.n_rows);
    std::vector<float> first(raw.n_rows);
    continued.trees()[1].predict(raw.view(), second);
    fresh.trees()[0].predict(raw.view(), first);
    CHECK(second == first);
    CHECK(second[4] != second[7]);
}

TEST_CASE("MulticlassBooster: separable 3-class data reaches perfect accuracy",
          "[booster][multiclass]")
{
    detail::ColumnBatch batch{
        .features      = {{0.0F, 0.1F, 0.5F, 0.6F, 0.9F, 1.0F}},
        .labels        = {0.0F, 0.0F, 1.0F, 1.0F, 2.0F, 2.0F},
        .weights       = {},
        .feature_names = {"a"},
    };
    Dataset const train = make_dataset(batch);
    auto const    raw   = to_raw(batch);

    Config cfg                       = tiny_cfg();
    cfg.objective.n_classes          = 3;
    cfg.booster_config.learning_rate = 0.5F;

    MulticlassBooster<DepthwiseGrower<>, AllRowsSampler> b{cfg};
    float                                                loss_before = 0.0F;
    for (int i = 0; i < 20; ++i)
    {
        b.update_one_iter(train);
        if (i == 0)
        {
            loss_before = b.eval(raw.view(), batch.labels);
        }
    }
    CHECK(b.n_iters() == 20);
    CHECK(b.eval(raw.view(), batch.labels) < loss_before); // mlogloss falls

    std::vector<float> pred(raw.n_rows);
    b.predict(raw.view(), pred);
    for (size_t i = 0; i < raw.n_rows; ++i)
    {
        CHECK(pred[i] == batch.labels[i]); // argmax class ids
    }

    // Round-trip through the registry-driven model IO.
    Config io_cfg                  = cfg;
    io_cfg.dispatch.objective_name = "softmax";
    BinMappers const mappers       = BinMappers::fit(batch, {});
    io::save_booster(b, "/tmp/bonsai_mc_test.msgpack", mappers, io_cfg);
    auto loaded = io::load_booster("/tmp/bonsai_mc_test.msgpack");
    REQUIRE(loaded.booster->n_iters() == 20);
    std::vector<float> pred2(raw.n_rows);
    loaded.booster->predict(raw.view(), pred2);
    CHECK(pred2 == pred);
}

// INVARIANT: init-scores-length-pins-output-count
// A multiclass booster's init scores are one per output or none: the score
// broadcast copies that vector into each row, so a vector of another length
// is refused at the boundary rather than written past a row.
TEST_CASE("MulticlassBooster: init scores of the wrong length are refused",
          "[booster][multiclass][edge]")
{
    Config cfg              = tiny_cfg();
    cfg.objective.n_classes = 3;
    MulticlassBooster<DepthwiseGrower<>, AllRowsSampler> b{cfg};
    CHECK_THROWS_WITH(b.load_state({}, {0.1F, 0.2F}),
                      Catch::Matchers::ContainsSubstring("disagrees"));
    CHECK_NOTHROW(b.load_state({}, {0.1F, 0.2F, 0.3F}));
    CHECK_NOTHROW(b.load_state({}, {}));
}

// The device predict seam. Both device routes read DevicePlanInput, and a
// dense booster can lend a view of its own trees where an oblivious one has
// to convert first. What follows pins the lifetime contract that makes the
// oblivious arm safe.

TEST_CASE("device_plan_input: a dense booster lends a view of its own trees",
          "[booster][predict]")
{
    auto const                                      batch = separable_batch();
    Dataset const                                   train = make_dataset(batch);
    MseBooster<DepthwiseGrower<CpuHistogramEngine>> b{tiny_cfg()};
    b.update_one_iter(train);

    auto const in = b.device_plan_input(0);
    CHECK(in.keep_alive == nullptr);
    CHECK(in.trees.data() == b.trees().data());
    CHECK(in.trees.size() == b.trees().size());
    CHECK(in.learning_rate == tiny_cfg().booster_config.learning_rate);
    CHECK(in.epoch != 0); // 0 is the plan cache's never-packed sentinel
}

TEST_CASE("device_plan_input: an oblivious booster lends its dense equivalent",
          "[booster][predict][oblivious]")
{
    auto const                                      batch = separable_batch();
    Dataset const                                   train = make_dataset(batch);
    MseBooster<ObliviousGrower<CpuHistogramEngine>> b{tiny_cfg()};
    b.update_one_iter(train);
    b.update_one_iter(train);

    auto const in = b.device_plan_input(0);
    REQUIRE(in.keep_alive != nullptr);
    REQUIRE(in.trees.size() == b.trees().size());
    CHECK(in.trees.data() == in.keep_alive->data());
    CHECK(in.learning_rate == tiny_cfg().booster_config.learning_rate);

    SECTION("a mutation re-converts under a new epoch")
    {
        b.update_one_iter(train);
        auto const after = b.device_plan_input(0);
        CHECK(after.epoch > in.epoch);
        CHECK(after.trees.size() == b.trees().size());
        CHECK(after.trees.data() != in.trees.data());
    }
}

TEST_CASE("device_plan_input: the trees outlive the booster that lent them",
          "[booster][predict][oblivious]")
{
    // The whole reason the input carries an owner. Without it the span points
    // into a cache the booster took with it, and this reads freed memory,
    // which is what the sanitizer build is for.
    auto const      batch = separable_batch();
    Dataset const   train = make_dataset(batch);
    DevicePlanInput in{};
    float           expected = 0.0F;
    {
        MseBooster<ObliviousGrower<CpuHistogramEngine>> b{tiny_cfg()};
        b.update_one_iter(train);
        in       = b.device_plan_input(0);
        expected = test::predict_one(in.trees[0], {0.95F});
    }
    REQUIRE(!in.trees.empty());
    CHECK(test::predict_one(in.trees[0], {0.95F}) == expected);
}

TEST_CASE("device_plan_input: an untrained booster still declines",
          "[booster][predict][oblivious]")
{
    // An empty ensemble densifies to an empty vector, so the caller's
    // trees.empty() decline fires without a special case.
    MseBooster<ObliviousGrower<CpuHistogramEngine>> b{tiny_cfg()};
    CHECK(b.device_plan_input(0).trees.empty());
}
