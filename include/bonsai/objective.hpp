#pragma once

#include "bonsai/config/config.hpp"
#include "bonsai/types.hpp"
#include <concepts>

namespace bonsai
{

// Objectives are instances constructed from Config so parameterized losses
// (huber_delta, quantile_alpha) can carry state. Parameter-free objectives
// keep static methods — statics satisfy instance-call syntax.
template <typename T>
concept Objective =
    std::constructible_from<T, Config const &> &&
    requires(T const &o, floats_view preds, floats_view targets, floats_out grad,
             floats_out hess) {
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
};

// Huber loss: L2 within |residual| <= delta, L1 outside.
struct HuberObjective
{
    HuberObjective() = default;
    explicit HuberObjective(Config const &cfg) : delta_(cfg.objective.huber_delta) {}
    void  compute(floats_view preds, floats_view targets, floats_out grad,
                  floats_out hess) const;
    float eval(floats_view preds, floats_view targets) const;
    static float init_score(floats_view targets); // median

  private:
    float delta_ = 1.0F;
};

// Pinball loss for the alpha-quantile.
struct QuantileObjective
{
    QuantileObjective() = default;
    explicit QuantileObjective(Config const &cfg)
        : alpha_(cfg.objective.quantile_alpha)
    {
    }
    void  compute(floats_view preds, floats_view targets, floats_out grad,
                  floats_out hess) const;
    float eval(floats_view preds, floats_view targets) const;
    float init_score(floats_view targets) const; // alpha-quantile

  private:
    float alpha_ = 0.5F;
};

} // namespace bonsai
