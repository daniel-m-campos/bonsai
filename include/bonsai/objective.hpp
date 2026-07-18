#pragma once

#include "bonsai/config/config.hpp"
#include "bonsai/types.hpp"
#include <concepts>
#include <span>

namespace bonsai
{

// Objectives are instances constructed from Config so parameterized losses
// (huber_delta, quantile_alpha) can carry state. Parameter-free objectives
// keep static methods — statics satisfy instance-call syntax.
template <typename T>
concept Objective = std::constructible_from<T, Config const &> &&
                    requires(T const &o, floats_view preds, floats_view targets,
                             floats_out grad, floats_out hess) {
                        { o.compute(preds, targets, grad, hess) } -> std::same_as<void>;
                        { o.eval(preds, targets) } -> std::same_as<float>;
                        { o.init_score(targets) } -> std::same_as<float>;
                    };

struct MSEObjective
{
    MSEObjective() = default;
    explicit MSEObjective(Config const &) {}
    static void  compute(floats_view preds, floats_view targets, floats_out grad,
                         floats_out hess);
    static float eval(floats_view preds, floats_view targets);
    static float init_score(floats_view targets);
};

struct LogLossObjective
{
    LogLossObjective() = default;
    explicit LogLossObjective(Config const &) {}
    static void  compute(floats_view scores, floats_view labels, floats_out grad,
                         floats_out hess);
    static float eval(floats_view scores, floats_view labels);
    static float init_score(floats_view labels);
};

// Poisson raw-score clamp: log-rates clamp to +-this before exp so a runaway
// leaf cannot overflow to inf and poison every later gradient. exp(30) ~ 1e13,
// far past any sane rate. Named here so the host objective and the device
// gradient kernel clamp with one identical constant.
inline constexpr float k_poisson_max_log = 30.0F;

// Poisson negative log-likelihood with a log link: raw scores are
// log-rates, grad = exp(F) - y, hess = exp(F). Labels must be >= 0
// (init_score throws otherwise). Raw scores are clamped to +-k_poisson_max_log
// inside compute/eval so the exp never overflows, the same guard role
// xgboost's max_delta_step plays for count:poisson.
struct PoissonObjective
{
    PoissonObjective() = default;
    explicit PoissonObjective(Config const &) {}
    static void  compute(floats_view scores, floats_view targets, floats_out grad,
                         floats_out hess);
    static float eval(floats_view scores, floats_view targets);
    static float init_score(floats_view targets);
};

// L1 loss: grad = sign(residual), hess = 1 (constant-hessian objective, so
// min_child_hess acts as a row count). Leaf values are gradient means, not
// residual medians — no leaf-renewal pass yet (LightGBM/XGBoost renew;
// expect an accuracy gap on MAE-scored comparisons).
struct MAEObjective
{
    MAEObjective() = default;
    explicit MAEObjective(Config const &) {}
    static void  compute(floats_view preds, floats_view targets, floats_out grad,
                         floats_out hess);
    static float eval(floats_view preds, floats_view targets);
    static float init_score(floats_view targets); // median
    // Leaf renewal: the L1-optimal leaf value is the residual median, not
    // the Newton step (which is a mean of +-1 gradients). Reorders in place.
    static float renew_leaf(std::span<float> residuals);
};

// Huber loss: L2 within |residual| <= delta, L1 outside.
struct HuberObjective
{
    HuberObjective() = default;
    explicit HuberObjective(Config const &cfg) : delta_(cfg.objective.huber_delta) {}
    void         compute(floats_view preds, floats_view targets, floats_out grad,
                         floats_out hess) const;
    float        eval(floats_view preds, floats_view targets) const;
    static float init_score(floats_view targets); // median
    // LightGBM-style huber renewal: residual median plus the mean of the
    // delta-clamped deviations from it. Reorders in place.
    float renew_leaf(std::span<float> residuals) const;

  private:
    float delta_ = 1.0F;
};

// Pinball loss for the alpha-quantile.
struct QuantileObjective
{
    QuantileObjective() = default;
    explicit QuantileObjective(Config const &cfg) : alpha_(cfg.objective.quantile_alpha)
    {
    }
    void  compute(floats_view preds, floats_view targets, floats_out grad,
                  floats_out hess) const;
    float eval(floats_view preds, floats_view targets) const;
    float init_score(floats_view targets) const; // alpha-quantile
    // Pinball-optimal leaf value: the alpha-quantile of the residuals.
    // Reorders in place.
    float renew_leaf(std::span<float> residuals) const;

  private:
    float alpha_ = 0.5F;
};

// Multiclass softmax. A dispatch tag more than an Objective: the
// K-output shape doesn't fit the 1-D Objective concept, so BoosterFor
// routes {softmax, G, Sa} to MulticlassBooster<G, Sa>, which owns the
// softmax math internally. The members below only satisfy the registry
// thunks (eval table, link table); the 1-D eval cannot express K columns
// and throws if reached.
// A dispatch TAG wearing the Objective interface (design review 2026-07-12,
// L finding — documented rather than re-typed): the K-output shape can't
// satisfy the 1-D concept, so BoosterFor routes {softmax, G, Sa} to
// MulticlassBooster and these methods are never called on the training
// path. They must still exist and throw because the generic per-objective
// trait tables (eval_table etc.) instantiate them for every typelist
// member; a separate tag type would need the same stubs under a different
// name.
struct SoftmaxObjective
{
    SoftmaxObjective() = default;
    explicit SoftmaxObjective(Config const &) {}
    static void  compute(floats_view preds, floats_view targets, floats_out grad,
                         floats_out hess);
    static float eval(floats_view preds, floats_view targets);
    static float init_score(floats_view targets);
};

} // namespace bonsai
