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
#include "bonsai/io/model.hpp"
#include "bonsai/multiclass_booster.hpp"
#include "bonsai/objective.hpp"
#include "bonsai/registry/typelists.hpp"
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
