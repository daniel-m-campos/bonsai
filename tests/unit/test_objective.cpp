#include "bonsai/booster.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/detail/column_batch.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/registry/objective_dispatch.hpp"
#include "bonsai/sampler.hpp"
#include <algorithm>
#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include "bonsai/objective.hpp"
#include "bonsai/types.hpp"

using namespace bonsai; // NOLINT

TEST_CASE("MSEObjective: compute writes grad = p - y and hess = 1",
          "[mse_objective][compute][basic]")
{
    // Dyadic-fraction inputs so the subtraction is bit-exact in float.
    std::vector<float> const preds   = {1.0F, 2.5F, -0.5F, 4.0F};
    std::vector<float> const targets = {0.5F, 3.0F, 0.5F, 4.0F};
    std::vector<float>       grad(4);
    std::vector<float>       hess(4);

    MSEObjective::compute(preds, targets, grad, hess);

    std::array<float, 4> const expected_grad{0.5F, -0.5F, -1.0F, 0.0F};
    CHECK(std::ranges::equal(grad, expected_grad));
    CHECK(std::ranges::all_of(hess, [](float h) { return h == 1.0F; }));
}

TEST_CASE("MSEObjective: compute overwrites grad and hess buffers",
          "[mse_objective][compute][overwrite]")
{
    // Pre-fill grad/hess with junk; compute must overwrite, not accumulate.
    // Pins decision 24 (4-objective.md).
    std::vector<float> const preds(8, 0.5F);
    std::vector<float> const targets(8, 0.25F);
    std::vector<float>       grad(8, 99.0F);
    std::vector<float>       hess(8, 99.0F);

    MSEObjective::compute(preds, targets, grad, hess);

    // grad[i] = 0.5 - 0.25 = 0.25 (bit-exact), not 99 + 0.25.
    CHECK(std::ranges::all_of(grad, [](float g) { return g == 0.25F; }));
    CHECK(std::ranges::all_of(hess, [](float h) { return h == 1.0F; }));
}

TEST_CASE("MSEObjective: compute handles zero residuals",
          "[mse_objective][compute][edge]")
{
    std::vector<float> const preds   = {1.0F, 2.0F, 3.0F, 4.0F};
    std::vector<float> const targets = preds; // exact equality
    std::vector<float>       grad(4, 99.0F);
    std::vector<float>       hess(4, 99.0F);

    MSEObjective::compute(preds, targets, grad, hess);

    CHECK(std::ranges::all_of(grad, [](float g) { return g == 0.0F; }));
    CHECK(std::ranges::all_of(hess, [](float h) { return h == 1.0F; }));
}

TEST_CASE("MSEObjective: compute on single-element input",
          "[mse_objective][compute][edge]")
{
    std::vector<float> const preds{2.5F};
    std::vector<float> const targets{1.5F};
    std::vector<float>       grad(1, 99.0F);
    std::vector<float>       hess(1, 99.0F);

    MSEObjective::compute(preds, targets, grad, hess);

    CHECK(grad[0] == 1.0F);
    CHECK(hess[0] == 1.0F);
}

TEST_CASE("MSEObjective: eval returns mean squared error",
          "[mse_objective][eval][basic]")
{
    // Residuals: {2, -1, 0, 3}. Squared sum = 4 + 1 + 0 + 9 = 14. MSE = 3.5.
    std::vector<float> const preds   = {3.0F, 1.0F, 4.0F, 7.0F};
    std::vector<float> const targets = {1.0F, 2.0F, 4.0F, 4.0F};

    CHECK(MSEObjective::eval(preds, targets) == 3.5F);
}

TEST_CASE("MSEObjective: eval is zero when preds match targets",
          "[mse_objective][eval][edge]")
{
    std::vector<float> const preds   = {1.0F, -2.0F, 3.5F, 100.0F};
    std::vector<float> const targets = preds;

    CHECK(MSEObjective::eval(preds, targets) == 0.0F);
}

TEST_CASE("MSEObjective: eval matches dyadic-fraction reference",
          "[mse_objective][eval][basic]")
{
    // Residuals: {0.5, -1.5, 2.0, -0.25, 1.25}. All dyadic.
    // Squares:    0.25, 2.25, 4.0, 0.0625, 1.5625. Sum = 8.125. n = 5. MSE = 1.625.
    std::vector<float> const preds   = {1.5F, 0.5F, 2.0F, -0.25F, 2.25F};
    std::vector<float> const targets = {1.0F, 2.0F, 0.0F, 0.0F, 1.0F};

    CHECK(MSEObjective::eval(preds, targets) == 1.625F);
}

// =========================================================================
// LogLossObjective
// =========================================================================

TEST_CASE("LogLossObjective: compute matches sigmoid math at score = 0",
          "[log_loss_objective][compute][basic]")
{
    // score = 0 -> sigmoid(0) = 0.5 exactly in float (1/(1+1)).
    // grad = 0.5 - y; hess = 0.5 * 0.5 = 0.25. Bit-exact.
    std::vector<float> const scores = {0.0F, 0.0F, 0.0F, 0.0F};
    std::vector<float> const labels = {0.0F, 1.0F, 0.0F, 1.0F};
    std::vector<float>       grad(4);
    std::vector<float>       hess(4);

    LogLossObjective::compute(scores, labels, grad, hess);

    std::array<float, 4> const expected_grad{0.5F, -0.5F, 0.5F, -0.5F};
    CHECK(std::ranges::equal(grad, expected_grad));
    CHECK(std::ranges::all_of(hess, [](float h) { return h == 0.25F; }));
}

TEST_CASE("LogLossObjective: compute matches sigmoid math at non-zero scores",
          "[log_loss_objective][compute][basic]")
{
    // score = log(3) ~ 1.0986 -> sigmoid = 3/(3+1) = 0.75.
    // score = -log(3) -> sigmoid = 1/(1+3) = 0.25.
    float const              log3   = std::log(3.0F);
    std::vector<float> const scores = {log3, -log3};
    std::vector<float> const labels = {1.0F, 0.0F};
    std::vector<float>       grad(2);
    std::vector<float>       hess(2);

    LogLossObjective::compute(scores, labels, grad, hess);

    // p=0.75, y=1: grad = -0.25, hess = 0.75 * 0.25 = 0.1875
    // p=0.25, y=0: grad = +0.25, hess = 0.25 * 0.75 = 0.1875
    CHECK(grad[0] == Catch::Approx(-0.25F).margin(1e-6));
    CHECK(grad[1] == Catch::Approx(+0.25F).margin(1e-6));
    CHECK(hess[0] == Catch::Approx(0.1875F).margin(1e-6));
    CHECK(hess[1] == Catch::Approx(0.1875F).margin(1e-6));
}

TEST_CASE("LogLossObjective: compute is numerically stable for extreme scores",
          "[log_loss_objective][compute][edge]")
{
    // Extreme raw scores must not produce NaN or inf. At |score| >> 1, the
    // simple sigmoid form saturates (1.0 or 0.0 via exp under/overflow); hess
    // collapses to 0, which the splitter's min_child_hess gate catches.
    std::vector<float> const scores = {+100.0F, -100.0F, +1000.0F, -1000.0F};
    std::vector<float> const labels = {1.0F, 0.0F, 0.0F, 1.0F};
    std::vector<float>       grad(4);
    std::vector<float>       hess(4);

    LogLossObjective::compute(scores, labels, grad, hess);

    CHECK(std::ranges::all_of(grad, [](float g) { return std::isfinite(g); }));
    CHECK(std::ranges::all_of(hess, [](float h) { return std::isfinite(h); }));
    CHECK(grad[0] == Catch::Approx(0.0F).margin(1e-6));  // p~1, y=1
    CHECK(grad[1] == Catch::Approx(0.0F).margin(1e-6));  // p~0, y=0
    CHECK(grad[2] == Catch::Approx(1.0F).margin(1e-6));  // p~1, y=0
    CHECK(grad[3] == Catch::Approx(-1.0F).margin(1e-6)); // p~0, y=1
}

TEST_CASE("LogLossObjective: compute overwrites grad and hess buffers",
          "[log_loss_objective][compute][overwrite]")
{
    std::vector<float> const scores(4, 0.0F);
    std::vector<float> const labels(4, 1.0F);
    std::vector<float>       grad(4, 99.0F);
    std::vector<float>       hess(4, 99.0F);

    LogLossObjective::compute(scores, labels, grad, hess);

    CHECK(std::ranges::all_of(grad, [](float g) { return g == -0.5F; }));
    CHECK(std::ranges::all_of(hess, [](float h) { return h == 0.25F; }));
}

TEST_CASE("LogLossObjective: compute hess stays in [0, 0.25] over reasonable scores",
          "[log_loss_objective][compute][range]")
{
    // hess = p * (1-p); max at p=0.5 (score=0), -> 0 at saturation.
    // Doc claims (0, 0.25]; in float at extreme scores p saturates to exactly
    // 0/1 and hess becomes exactly 0. min_child_hess catches it downstream.
    std::vector<float> scores;
    std::vector<float> labels;
    for (int i = -50; i <= 50; ++i)
    {
        scores.push_back(static_cast<float>(i) * 0.1F);
        labels.push_back(0.0F);
    }
    std::vector<float> grad(scores.size());
    std::vector<float> hess(scores.size());

    LogLossObjective::compute(scores, labels, grad, hess);

    CHECK(std::ranges::all_of(hess, [](float h) { return h >= 0.0F && h <= 0.25F; }));
    CHECK(hess[50] == 0.25F); // score = 0 -> hess = 0.25 exactly
}

TEST_CASE("LogLossObjective: eval returns log(2) when all scores are 0",
          "[log_loss_objective][eval][basic]")
{
    // score=0 -> BCE = -log(0.5) = log(2), regardless of y. Mean = log(2).
    std::vector<float> const scores = {0.0F, 0.0F, 0.0F, 0.0F};
    std::vector<float> const labels = {0.0F, 1.0F, 0.0F, 1.0F};

    CHECK(LogLossObjective::eval(scores, labels) ==
          Catch::Approx(std::log(2.0F)).margin(1e-6));
}

TEST_CASE("LogLossObjective: eval matches reference cross-entropy",
          "[log_loss_objective][eval][basic]")
{
    std::vector<float> const scores = {0.0F, 1.0F, -1.0F, 2.0F, -2.0F};
    std::vector<float> const labels = {1.0F, 0.0F, 1.0F, 1.0F, 0.0F};

    // Reference: per-row BCE-from-logits via stable softplus form.
    double ref_sum = 0.0;
    for (size_t i = 0; i < scores.size(); ++i)
    {
        double const x        = scores[i];
        double const softplus = std::max(0.0, x) + std::log1p(std::exp(-std::abs(x)));
        ref_sum += softplus - (labels[i] * x);
    }
    auto const expected =
        static_cast<float>(ref_sum / static_cast<double>(scores.size()));

    CHECK(LogLossObjective::eval(scores, labels) ==
          Catch::Approx(expected).margin(1e-6));
}

TEST_CASE("LogLossObjective: eval is finite for extreme scores",
          "[log_loss_objective][eval][edge]")
{
    // The stable softplus form must not overflow at large |score|.
    std::vector<float> const scores = {+100.0F, -100.0F, +1000.0F, -1000.0F};
    std::vector<float> const labels = {1.0F, 0.0F, 1.0F, 0.0F};

    float const loss = LogLossObjective::eval(scores, labels);

    CHECK(std::isfinite(loss));
    // Correctly-classified extreme scores -> near-zero loss.
    CHECK(loss == Catch::Approx(0.0F).margin(1e-3));
}

TEST_CASE("LogLossObjective: init_score stays finite on a single-class label set",
          "[objective][logloss][edge]")
{
    SECTION("every label 0")
    {
        std::vector<float> const y(8, 0.0F);
        float const              init = LogLossObjective::init_score(y);
        CHECK(std::isfinite(init));
        CHECK(init < 0.0F);
    }
    SECTION("every label 1")
    {
        std::vector<float> const y(8, 1.0F);
        float const              init = LogLossObjective::init_score(y);
        CHECK(std::isfinite(init));
        CHECK(init > 0.0F);
    }
}

TEST_CASE("LogLossObjective: init_score is the log-odds of a two-class label set",
          "[objective][logloss]")
{
    std::vector<float> const y{0.0F, 0.0F, 0.0F, 1.0F};
    CHECK(LogLossObjective::init_score(y) == Catch::Approx(std::log(1.0F / 3.0F)));
}

TEST_CASE("MAEObjective: sign gradients, unit hessians, median init",
          "[objective][mae]")
{
    std::vector<float> preds{2.0F, 1.0F, 3.0F};
    std::vector<float> targets{1.0F, 1.0F, 5.0F};
    std::vector<float> grad(3);
    std::vector<float> hess(3);
    MAEObjective::compute(preds, targets, grad, hess);
    CHECK(grad[0] == 1.0F);  // over-prediction
    CHECK(grad[1] == 0.0F);  // exact
    CHECK(grad[2] == -1.0F); // under-prediction
    CHECK(hess[0] == 1.0F);

    // eval: mean |r| = (1 + 0 + 2) / 3.
    CHECK(MAEObjective::eval(preds, targets) == Catch::Approx(1.0F));

    std::vector<float> labels{5.0F, 1.0F, 3.0F, 9.0F, 7.0F};
    CHECK(MAEObjective::init_score(labels) == 5.0F); // median
}

TEST_CASE("HuberObjective: gradient clamps at delta", "[objective][huber]")
{
    Config cfg{};
    cfg.objective.huber_delta = 2.0F;
    HuberObjective const obj{cfg};

    std::vector<float> preds{0.5F, 5.0F, -5.0F};
    std::vector<float> targets{0.0F, 0.0F, 0.0F};
    std::vector<float> grad(3);
    std::vector<float> hess(3);
    obj.compute(preds, targets, grad, hess);
    CHECK(grad[0] == 0.5F);  // inside the L2 zone
    CHECK(grad[1] == 2.0F);  // clamped
    CHECK(grad[2] == -2.0F); // clamped

    // eval: 0.5*0.25 + 2*(5-1) + 2*(5-1) over 3.
    CHECK(obj.eval(preds, targets) == Catch::Approx((0.125F + 8.0F + 8.0F) / 3.0F));
}

TEST_CASE("QuantileObjective: pinball gradients and alpha-quantile init",
          "[objective][quantile]")
{
    Config cfg{};
    cfg.objective.quantile_alpha = 0.9F;
    QuantileObjective const obj{cfg};

    std::vector<float> preds{2.0F, 0.0F};
    std::vector<float> targets{1.0F, 1.0F};
    std::vector<float> grad(2);
    std::vector<float> hess(2);
    obj.compute(preds, targets, grad, hess);
    CHECK(grad[0] == Catch::Approx(0.1F));  // over-prediction: 1 - alpha
    CHECK(grad[1] == Catch::Approx(-0.9F)); // under-prediction: -alpha

    // eval: over by 1 -> (1-a)*1 = 0.1; under by 1 -> a*1 = 0.9.
    CHECK(obj.eval(preds, targets) == Catch::Approx(0.5F));

    std::vector<float> labels{1.0F, 2.0F, 3.0F, 4.0F, 5.0F,
                              6.0F, 7.0F, 8.0F, 9.0F, 10.0F};
    // Nearest-rank on alpha * (n-1): round(0.9 * 9) = 8 -> value 9.
    CHECK(obj.init_score(labels) == 9.0F);
}

TEST_CASE("Objective: eval is the float-ordered mean of the per-pair loss",
          "[objective][eval]")
{
    // Pinned to the values the left-to-right float reduction returns on
    // non-dyadic inputs, so a refactor of the reduction holds them bit for bit.
    std::vector<float> const preds{0.3F, -1.7F, 2.2F, 4.9F, -0.6F, 1.1F, 3.8F};
    std::vector<float> const targets{0.0F, -1.0F, 2.5F, 1.0F, -0.9F, 1.4F, 0.7F};

    SECTION("mse")
    {
        CHECK(MSEObjective::eval(preds, targets) == 3.66714287F);
    }
    SECTION("mae")
    {
        CHECK(MAEObjective::eval(preds, targets) == 1.27142847F);
    }
    SECTION("huber")
    {
        Config cfg{};
        cfg.objective.huber_delta = 2.0F;
        CHECK(HuberObjective{cfg}.eval(preds, targets) == 1.48928571F);
    }
    SECTION("quantile")
    {
        Config cfg{};
        cfg.objective.quantile_alpha = 0.9F;
        CHECK(QuantileObjective{cfg}.eval(preds, targets) == 0.275714308F);
    }
    SECTION("logloss")
    {
        std::vector<float> const labels{0.0F, 1.0F, 1.0F, 0.0F, 1.0F, 0.0F, 1.0F};
        CHECK(LogLossObjective::eval(preds, labels) == 1.45451319F);
    }
}

TEST_CASE("PoissonObjective: gradients, init, and the label guard",
          "[objective][poisson]")
{
    // grad = exp(F) - y, hess = exp(F) at raw log-rates F.
    std::vector<float> const scores{0.0F, std::log(4.0F)};
    std::vector<float> const y{1.0F, 4.0F};
    std::vector<float>       grad(2);
    std::vector<float>       hess(2);
    PoissonObjective::compute(scores, y, grad, hess);
    CHECK(grad[0] == Catch::Approx(1.0F - 1.0F)); // exp(0) - 1
    CHECK(hess[0] == Catch::Approx(1.0F));
    CHECK(grad[1] == Catch::Approx(4.0F - 4.0F));
    CHECK(hess[1] == Catch::Approx(4.0F));

    // init = log(mean(y)); a perfectly-initialized model has zero mean grad.
    float const init = PoissonObjective::init_score(y);
    CHECK(init == Catch::Approx(std::log(2.5F)));

    // Negative labels are a modeling error, loudly.
    std::vector<float> const bad{1.0F, -0.5F};
    CHECK_THROWS_AS(PoissonObjective::init_score(bad), std::invalid_argument);

    // The clamp keeps a runaway raw score finite instead of inf-poisoning.
    std::vector<float> const wild{100.0F};
    std::vector<float> const y1{1.0F};
    std::vector<float>       g1(1);
    std::vector<float>       h1(1);
    PoissonObjective::compute(wild, y1, g1, h1);
    CHECK(std::isfinite(g1[0]));
    CHECK(std::isfinite(h1[0]));
}

TEST_CASE("Poisson end to end: recovers rates through the log link",
          "[objective][poisson][booster]")
{
    // Two regimes: rate ~1 below the split, rate ~8 above. After training,
    // exp(raw predictions) should sit near the regime means.
    detail::ColumnBatch batch;
    batch.features.resize(1);
    batch.feature_names = {"x"};
    std::mt19937                   rng(3);
    std::poisson_distribution<int> low(1.0);
    std::poisson_distribution<int> high(8.0);
    for (int i = 0; i < 400; ++i)
    {
        float const x = static_cast<float>(i % 2);
        batch.features[0].push_back(x + 0.001F * static_cast<float>(i));
        batch.labels.push_back(static_cast<float>(x < 0.5F ? low(rng) : high(rng)));
    }
    BinMappers const mappers = BinMappers::fit(batch, {});
    Dataset const    train   = Dataset::bin(batch, mappers, {});

    Config cfg;
    cfg.booster_config.n_iters       = 60;
    cfg.booster_config.learning_rate = 0.2F;
    cfg.tree_config.max_depth        = 3;
    Booster<PoissonObjective, DepthwiseGrower<>, AllRowsSampler> b{cfg};
    for (int i = 0; i < 60; ++i)
    {
        b.update_one_iter(train);
    }

    std::vector<float> raw(batch.features[0]);
    features_view      X{raw.data(), raw.size(), 1};
    std::vector<float> pred(raw.size());
    b.predict(X, pred);
    apply_link_inverse_by_name("poisson", pred);
    double lo_sum = 0.0, hi_sum = 0.0;
    for (size_t i = 0; i < pred.size(); ++i)
    {
        (raw[i] < 0.5F ? lo_sum : hi_sum) += pred[i];
    }
    CHECK(lo_sum / 200.0 == Catch::Approx(1.0).margin(0.35));
    CHECK(hi_sum / 200.0 == Catch::Approx(8.0).margin(1.5));
}

namespace
{
std::vector<WeightedValue> paired(std::vector<float> const &values,
                                  std::vector<float> const &weights)
{
    std::vector<WeightedValue> out(values.size());
    for (size_t i = 0; i < values.size(); ++i)
    {
        out[i] = {.value = values[i], .weight = weights[i]};
    }
    return out;
}

template <typename Obj>
void check_equal_weights_change_nothing(Obj const                &obj,
                                        std::vector<float> const &values)
{
    std::vector<float> reordered = values;
    float const        renewed   = obj.renew_leaf(std::span<float>{reordered});
    for (float const weight : {1.0F, 0.3F, 2.5F})
    {
        std::vector<float> const weights(values.size(), weight);
        CHECK(obj.init_score(values, weights) == obj.init_score(values));
        std::vector<WeightedValue> residuals = paired(values, weights);
        CHECK(obj.renew_leaf(std::span<WeightedValue>{residuals}) == renewed);
    }
}

template <typename Obj> void check_weightless_rows_are_absent(Obj const &obj)
{
    std::vector<float> const kept{0.0F, 1.0F, 1.0F, 0.0F, 1.0F};
    std::vector<float> const values{0.0F, 0.0F, 1.0F, 0.0F, 1.0F,
                                    0.0F, 0.0F, 0.0F, 1.0F};
    std::vector<float> const weights{1.0F, 0.0F, 1.0F, -1.0F, 1.0F,
                                     0.0F, 1.0F, 0.0F, 1.0F};
    CHECK(obj.init_score(values, weights) == obj.init_score(kept));
}

// The weighted quantile rule read off a full sort: the first value, in value
// order over the rows of positive weight, whose running weight in units of the
// mean weight, less one half, passes alpha * (n - 1).
float sorted_scan_quantile(std::vector<WeightedValue> values, float alpha)
{
    std::erase_if(values, [](WeightedValue const &v) { return !(v.weight > 0.0F); });
    std::ranges::sort(values, {}, &WeightedValue::value);
    double total = 0.0;
    for (WeightedValue const &v : values)
    {
        total += v.weight;
    }
    auto const   n       = static_cast<double>(values.size());
    double const rank    = static_cast<double>(alpha) * (n - 1.0);
    double       running = 0.0;
    for (WeightedValue const &v : values)
    {
        running += v.weight;
        if (running / (total / n) - 0.5 > rank)
        {
            return v.value;
        }
    }
    return values.back().value;
}

template <typename Obj> void check_no_positive_weight_is_refused(Obj const &obj)
{
    std::vector<float> const values{0.0F, 1.0F, 1.0F};
    std::vector<float> const weights{0.0F, 0.0F, -2.0F};
    CHECK_THROWS_AS(obj.init_score(values, weights), std::invalid_argument);
}
} // namespace

// INVARIANT: equal-weights-select-the-unweighted-quantile
// Under one constant weight on every row the weighted quantile is the
// element the unweighted rule picks, for the base score and for leaf
// renewal, at every size and every alpha, with ties or without.
TEST_CASE("Objective: equal weights select the unweighted quantile",
          "[objective][weights]")
{
    std::mt19937 rng{7};
    for (size_t const n : {1U, 2U, 3U, 4U, 7U, 10U, 64U, 101U})
    {
        for (int const distinct : {4, 1000})
        {
            std::uniform_int_distribution<int> draw{-distinct, distinct};
            std::vector<float>                 values(n);
            std::ranges::generate(values,
                                  [&] { return static_cast<float>(draw(rng)); });
            INFO("n = " << n << ", values in +-" << distinct);
            check_equal_weights_change_nothing(MAEObjective{}, values);
            for (float const alpha : {0.05F, 0.1F, 0.25F, 0.5F, 0.75F, 0.9F, 0.95F})
            {
                Config cfg;
                cfg.objective.quantile_alpha = alpha;
                INFO("alpha = " << alpha);
                check_equal_weights_change_nothing(QuantileObjective{cfg}, values);
            }
        }
    }
}

TEST_CASE("Objective: a weighted base score is the weighted statistic",
          "[objective][weights]")
{
    SECTION("mse reads the weighted mean")
    {
        std::vector<float> const y{1.0F, 2.0F, 10.0F};
        std::vector<float> const w{1.0F, 1.0F, 2.0F};
        CHECK(MSEObjective::init_score(y, w) == 5.75F);
    }
    SECTION("logloss reads the log-odds of the weighted base rate")
    {
        std::vector<float> const y{1.0F, 0.0F, 0.0F, 0.0F};
        CHECK(LogLossObjective::init_score(y, std::vector{3.0F, 1.0F, 1.0F, 1.0F}) ==
              0.0F);
        CHECK(LogLossObjective::init_score(y, std::vector{6.0F, 1.0F, 1.0F, 0.0F}) ==
              std::log(3.0F));
    }
    SECTION("poisson reads the log of the weighted mean")
    {
        std::vector<float> const y{0.0F, 4.0F};
        std::vector<float> const w{1.0F, 3.0F};
        CHECK(PoissonObjective::init_score(y, w) == static_cast<float>(std::log(3.0)));
    }
    SECTION("mae and huber read the weighted median")
    {
        std::vector<float> const y{10.0F, 20.0F, 30.0F};
        std::vector<float> const w{4.0F, 1.0F, 1.0F};
        CHECK(MAEObjective::init_score(y) == 20.0F);
        CHECK(MAEObjective::init_score(y, w) == 10.0F);
        CHECK(HuberObjective::init_score(y, w) == 10.0F);
    }
    SECTION("quantile reads the weighted alpha-quantile")
    {
        Config cfg;
        cfg.objective.quantile_alpha = 0.25F;
        QuantileObjective const  obj{cfg};
        std::vector<float> const y{1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
        std::vector<float> const w{1.0F, 1.0F, 1.0F, 1.0F, 8.0F};
        CHECK(obj.init_score(y) == 2.0F);
        CHECK(obj.init_score(y, w) == 4.0F);
    }
}

TEST_CASE("Objective: a row without positive weight is absent from the base score",
          "[objective][weights]")
{
    Config const cfg;
    check_weightless_rows_are_absent(MSEObjective{cfg});
    check_weightless_rows_are_absent(LogLossObjective{cfg});
    check_weightless_rows_are_absent(PoissonObjective{cfg});
    check_weightless_rows_are_absent(MAEObjective{cfg});
    check_weightless_rows_are_absent(HuberObjective{cfg});
    check_weightless_rows_are_absent(QuantileObjective{cfg});

    // Away from the median the rank depends on how many rows are counted, so
    // a weightless row that stayed in the count would move the answer to 1.
    Config low;
    low.objective.quantile_alpha = 0.2F;
    QuantileObjective const  quantile{low};
    std::vector<float> const kept{1.0F, 2.0F, 3.0F, 4.0F};
    std::vector<float> const values{1.0F, 9.0F, 2.0F, 9.0F, 3.0F, 0.0F, 4.0F, 0.0F};
    std::vector<float> const weights{1.0F, 0.0F, 1.0F, 0.0F, 1.0F, 0.0F, 1.0F, 0.0F};
    CHECK(quantile.init_score(kept) == 2.0F);
    CHECK(quantile.init_score(values, weights) == 2.0F);
    std::vector<WeightedValue> residuals = paired(values, weights);
    CHECK(quantile.renew_leaf(std::span<WeightedValue>{residuals}) == 2.0F);
}

TEST_CASE("Objective: weighted leaf renewal is the weighted statistic",
          "[objective][weights][renewal]")
{
    SECTION("mae renews to the weighted median")
    {
        std::vector<WeightedValue> residuals{
            {-10.0F, 1.0F}, {0.0F, 4.0F}, {-10.0F, 1.0F}, {50.0F, 0.0F}};
        CHECK(MAEObjective::renew_leaf(std::span<WeightedValue>{residuals}) == 0.0F);
    }
    SECTION("quantile renews to the weighted alpha-quantile")
    {
        Config cfg;
        cfg.objective.quantile_alpha = 0.25F;
        std::vector<WeightedValue> residuals{
            {5.0F, 8.0F}, {1.0F, 1.0F}, {2.0F, 1.0F}, {3.0F, 1.0F}, {4.0F, 1.0F}};
        CHECK(QuantileObjective{cfg}.renew_leaf(std::span<WeightedValue>{residuals}) ==
              4.0F);
    }
    SECTION("huber renews to the weighted median plus the weighted clamped mean")
    {
        Config const               cfg;
        std::vector<WeightedValue> residuals{
            {0.0F, 1.0F}, {4.0F, 2.0F}, {100.0F, 0.0F}, {0.0F, 1.0F}};
        CHECK(HuberObjective{cfg}.renew_leaf(std::span<WeightedValue>{residuals}) ==
              3.5F);
    }
}

TEST_CASE("Objective: the weighted quantile is the sorted scan's value",
          "[objective][weights][renewal]")
{
    std::mt19937               rng{11};
    std::array<float, 6> const weight_values{0.0F, 0.25F, 0.5F, 1.0F, 4.0F, 32.0F};
    std::uniform_int_distribution<size_t> draw_weight{0, weight_values.size() - 1};
    for (size_t const n : {1U, 2U, 3U, 5U, 16U, 257U, 4096U})
    {
        for (int const distinct : {3, 50, 100000})
        {
            std::uniform_int_distribution<int> draw_value{-distinct, distinct};
            std::vector<WeightedValue>         residuals(n);
            std::ranges::generate(residuals,
                                  [&]
                                  {
                                      return WeightedValue{
                                          .value  = static_cast<float>(draw_value(rng)),
                                          .weight = weight_values[draw_weight(rng)]};
                                  });
            residuals.front().weight = 1.0F;
            for (float const alpha : {0.05F, 0.2F, 0.5F, 0.8F, 0.95F})
            {
                Config cfg;
                cfg.objective.quantile_alpha = alpha;
                INFO("n = " << n << ", values in +-" << distinct
                            << ", alpha = " << alpha);
                std::vector<WeightedValue> scratch = residuals;
                CHECK(QuantileObjective{cfg}.renew_leaf(std::span<WeightedValue>{
                          scratch}) == sorted_scan_quantile(residuals, alpha));
            }
        }
    }
}

TEST_CASE("Objective: a fit with no positive weight is refused",
          "[objective][weights][edge]")
{
    Config const cfg;
    check_no_positive_weight_is_refused(MSEObjective{cfg});
    check_no_positive_weight_is_refused(LogLossObjective{cfg});
    check_no_positive_weight_is_refused(PoissonObjective{cfg});
    check_no_positive_weight_is_refused(MAEObjective{cfg});
    check_no_positive_weight_is_refused(HuberObjective{cfg});
    check_no_positive_weight_is_refused(QuantileObjective{cfg});

    CHECK(require_positive_weight(2.5) == 2.5);
    CHECK_THROWS_AS(require_positive_weight(0.0), std::invalid_argument);
    CHECK_THROWS_AS(require_positive_weight(std::nan("")), std::invalid_argument);
}
