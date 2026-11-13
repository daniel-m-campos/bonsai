#pragma once

#include "bonsai/config/config.hpp"
#include "bonsai/types.hpp"
#include <concepts>
#include <span>

namespace bonsai
{

struct DeviceHuber;
struct DeviceQuantile;

// A label or residual paired with its row's weight. The weighted
// alpha-quantile drops every element whose weight is not positive, orders the
// n that remain by value, scales their weights to average one, and returns
// the first whose running weight less one half exceeds alpha * (n - 1). Under
// equal weights that is index llround(alpha * (n - 1)), the unweighted rule's
// element. Pinned by "Objective: equal weights select the unweighted
// quantile" and "Objective: the weighted quantile is the sorted scan's value".
struct WeightedValue
{
    float value;
    float weight;
};

struct WeightedQuantile
{
    double mean_weight;
    double rank;

    constexpr bool bounded_by(double running_weight) const
    {
        return (running_weight / mean_weight) - 0.5 > rank;
    }
};

// Returns total_weight, or throws std::invalid_argument when it is not
// positive: rows that all weigh nothing have no base score. Pinned by
// "Objective: a fit with no positive weight is refused".
double require_positive_weight(double total_weight);

// Objectives are instances constructed from Config so parameterized losses
// (huber_delta, quantile_alpha) can carry state. Parameter-free objectives
// keep static methods: statics satisfy instance-call syntax.
//
// init_score(targets, weights) counts every row by its weight, as a weighted
// mean or a weighted quantile: a row whose weight is not positive is absent,
// and weights that are all one return the unweighted score.
template <typename T>
concept Objective = std::constructible_from<T, Config const &> &&
                    requires(T const &o, floats_view preds, floats_view targets,
                             floats_view weights, floats_out grad, floats_out hess) {
                        { o.compute(preds, targets, grad, hess) } -> std::same_as<void>;
                        { o.eval(preds, targets) } -> std::same_as<float>;
                        { o.init_score(targets) } -> std::same_as<float>;
                        { o.init_score(targets, weights) } -> std::same_as<float>;
                    };

struct MSEObjective
{
    MSEObjective() = default;
    explicit MSEObjective(Config const &) {}
    static void  compute(floats_view preds, floats_view targets, floats_out grad,
                         floats_out hess);
    static float eval(floats_view preds, floats_view targets);
    static float init_score(floats_view targets);
    static float init_score(floats_view targets, floats_view weights);
};

struct LogLossObjective
{
    LogLossObjective() = default;
    explicit LogLossObjective(Config const &) {}
    static void  compute(floats_view scores, floats_view labels, floats_out grad,
                         floats_out hess);
    static float eval(floats_view scores, floats_view labels);
    static float init_score(floats_view labels);
    static float init_score(floats_view labels, floats_view weights);
};

// Poisson raw-score clamp: log-rates clamp to +-this before exp so a runaway
// leaf cannot overflow to inf and poison every later gradient. exp(30) ~ 1e13,
// far past any sane rate. One named constant so the host objective and the
// device gradient kernel cannot drift apart (invariants:
// device-objective-formula-matches-host).
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
    static float init_score(floats_view targets, floats_view weights);
};

// L1 loss: grad = sign(residual), hess = 1, so min_child_hess acts as a
// row count. renew_leaf replaces each leaf with its residual median.
struct MAEObjective
{
    MAEObjective() = default;
    explicit MAEObjective(Config const &) {}
    static void  compute(floats_view preds, floats_view targets, floats_out grad,
                         floats_out hess);
    static float eval(floats_view preds, floats_view targets);
    static float init_score(floats_view targets); // median
    static float init_score(floats_view targets, floats_view weights);
    // Leaf renewal: the L1-optimal leaf value is the residual median, not
    // the Newton step (which is a mean of +-1 gradients). Reorders in place.
    static float renew_leaf(std::span<float> residuals);
    static float renew_leaf(std::span<WeightedValue> residuals);
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
    static float init_score(floats_view targets, floats_view weights);
    // LightGBM-style huber renewal: residual median plus the mean of the
    // delta-clamped deviations from it. Reorders in place.
    float              renew_leaf(std::span<float> residuals) const;
    float              renew_leaf(std::span<WeightedValue> residuals) const;
    friend DeviceHuber device_objective_of(HuberObjective const &objective);

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
    float init_score(floats_view targets, floats_view weights) const;
    // Pinball-optimal leaf value: the alpha-quantile of the residuals.
    // Reorders in place.
    float                 renew_leaf(std::span<float> residuals) const;
    float                 renew_leaf(std::span<WeightedValue> residuals) const;
    friend DeviceQuantile device_objective_of(QuantileObjective const &objective);

  private:
    float alpha_ = 0.5F;
};

// A dispatch TAG wearing the Objective interface, documented rather than
// re-typed: the K-output shape can't satisfy the 1-D concept, so BoosterFor
// routes {softmax, G, Sa} to MulticlassBooster and these methods are never
// called on the training path. They must still exist and throw because the
// generic per-objective trait tables (link_inverse_of, default_metrics_of)
// instantiate them for every typelist member; a separate tag type would need
// the same stubs under a different name.
struct SoftmaxObjective
{
    SoftmaxObjective() = default;
    explicit SoftmaxObjective(Config const &) {}
    static void  compute(floats_view preds, floats_view targets, floats_out grad,
                         floats_out hess);
    static float eval(floats_view preds, floats_view targets);
    static float init_score(floats_view targets, floats_view weights = {});
};

} // namespace bonsai
