#include <algorithm>
#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstddef>
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
    CHECK(obj.eval(preds, targets) ==
          Catch::Approx((0.125F + 8.0F + 8.0F) / 3.0F));
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

    std::vector<float> labels{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F,
                              9.0F, 10.0F};
    // Nearest-rank on alpha * (n-1): round(0.9 * 9) = 8 -> value 9.
    CHECK(obj.init_score(labels) == 9.0F);
}
