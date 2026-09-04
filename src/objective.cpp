#include "bonsai/objective.hpp"
#include "bonsai/parallel.hpp"
#include "bonsai/types.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace bonsai
{

namespace
{

float quantile_in_place(std::span<float> v, float alpha)
{
    assert(!v.empty());
    auto const k = std::min<size_t>(
        v.size() - 1,
        static_cast<size_t>(std::llround(static_cast<double>(alpha) *
                                         static_cast<double>(v.size() - 1))));
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(k), v.end());
    return v[k];
}

float quantile_of(floats_view values, float alpha)
{
    std::vector<float> v(values.begin(), values.end());
    return quantile_in_place(v, alpha);
}

template <typename PairLoss>
float mean_pair_loss(floats_view preds, floats_view targets, PairLoss loss)
{
    assert(preds.size() == targets.size());
    assert(!preds.empty());
    float const sum = std::transform_reduce(preds.begin(), preds.end(), targets.begin(),
                                            0.0F, std::plus<>(), loss);
    return sum / static_cast<float>(preds.size());
}

} // namespace

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void MSEObjective::compute(floats_view preds, floats_view targets, floats_out grad,
                           floats_out hess)
{
    assert(preds.size() == targets.size());
    assert(targets.size() == grad.size());
    assert(grad.size() == hess.size());

    parallel::for_each_index(preds.size(),
                             [&](size_t i)
                             {
                                 grad[i] = preds[i] - targets[i];
                                 hess[i] = 1.0F;
                             });
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
auto MSEObjective::eval(floats_view preds, floats_view targets)
    -> floats_view::value_type
{
    return mean_pair_loss(preds, targets,
                          [](auto const p, auto const t)
                          {
                              double const diff = p - t;
                              return diff * diff;
                          });
}

auto MSEObjective::init_score(floats_view targets) -> floats_view::value_type
{
    assert(!targets.empty());
    auto const  n   = static_cast<float>(targets.size());
    float const sum = std::accumulate(targets.begin(), targets.end(), 0.0F);
    return sum / n;
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void LogLossObjective::compute(floats_view scores, floats_view labels, floats_out grad,
                               floats_out hess)
{
    assert(scores.size() == labels.size());
    assert(labels.size() == grad.size());
    assert(grad.size() == hess.size());

    parallel::for_each_index(scores.size(),
                             [&](size_t i)
                             {
                                 float const score = scores[i];
                                 float const p     = 1.0F / (1.0F + std::exp(-score));
                                 grad[i]           = p - labels[i];
                                 hess[i]           = p * (1.0F - p);
                             });
}

float LogLossObjective::eval(floats_view scores, floats_view labels)
{
    return mean_pair_loss(scores, labels,
                          [](auto const score, auto const y)
                          {
                              float const ax = std::abs(score);
                              return std::max(0.0F, score) + std::log1p(std::exp(-ax)) -
                                     (y * score);
                          });
}

auto LogLossObjective::init_score(floats_view labels) -> floats_view::value_type
{
    assert(!labels.empty());
    auto const  n   = static_cast<float>(labels.size());
    float const sum = std::accumulate(labels.begin(), labels.end(), 0.0F);
    float const p   = sum / n;
    return std::log(p / (1.0F - p));
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void MAEObjective::compute(floats_view preds, floats_view targets, floats_out grad,
                           floats_out hess)
{
    assert(preds.size() == targets.size());
    parallel::for_each_index(preds.size(),
                             [&](size_t i)
                             {
                                 float const r = preds[i] - targets[i];
                                 grad[i] = r > 0.0F ? 1.0F : (r < 0.0F ? -1.0F : 0.0F);
                                 hess[i] = 1.0F;
                             });
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
auto MAEObjective::eval(floats_view preds, floats_view targets)
    -> floats_view::value_type
{
    return mean_pair_loss(preds, targets,
                          [](auto const p, auto const t) { return std::abs(p - t); });
}

auto MAEObjective::init_score(floats_view targets) -> floats_view::value_type
{
    return quantile_of(targets, 0.5F);
}

float MAEObjective::renew_leaf(std::span<float> residuals)
{
    return quantile_in_place(residuals, 0.5F);
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void HuberObjective::compute(floats_view preds, floats_view targets, floats_out grad,
                             floats_out hess) const
{
    assert(preds.size() == targets.size());
    float const d = delta_;
    parallel::for_each_index(preds.size(),
                             [&](size_t i)
                             {
                                 float const r = preds[i] - targets[i];
                                 grad[i]       = std::clamp(r, -d, d);
                                 hess[i]       = 1.0F;
                             });
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
auto HuberObjective::eval(floats_view preds, floats_view targets) const
    -> floats_view::value_type
{
    float const d = delta_;
    return mean_pair_loss(preds, targets,
                          [d](auto const p, auto const t)
                          {
                              float const a = std::abs(p - t);
                              return a <= d ? 0.5F * a * a : d * (a - (0.5F * d));
                          });
}

auto HuberObjective::init_score(floats_view targets) -> floats_view::value_type
{
    return quantile_of(targets, 0.5F);
}

float HuberObjective::renew_leaf(std::span<float> residuals) const
{
    float const med = quantile_in_place(residuals, 0.5F);
    double      sum = 0.0;
    for (float const r : residuals)
    {
        sum += std::clamp(r - med, -delta_, delta_);
    }
    return med + static_cast<float>(sum / static_cast<double>(residuals.size()));
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void QuantileObjective::compute(floats_view preds, floats_view targets, floats_out grad,
                                floats_out hess) const
{
    assert(preds.size() == targets.size());
    float const a = alpha_;
    parallel::for_each_index(preds.size(),
                             [&](size_t i)
                             {
                                 grad[i] = preds[i] > targets[i] ? (1.0F - a) : -a;
                                 hess[i] = 1.0F;
                             });
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
auto QuantileObjective::eval(floats_view preds, floats_view targets) const
    -> floats_view::value_type
{
    float const a = alpha_;
    return mean_pair_loss(preds, targets, [a](auto const p, auto const t)
                          { return t >= p ? a * (t - p) : (1.0F - a) * (p - t); });
}

auto QuantileObjective::init_score(floats_view targets) const -> floats_view::value_type
{
    return quantile_of(targets, alpha_);
}

float QuantileObjective::renew_leaf(std::span<float> residuals) const
{
    return quantile_in_place(residuals, alpha_);
}

namespace
{
float clamped_exp(float raw)
{
    return std::exp(std::clamp(raw, -k_poisson_max_log, k_poisson_max_log));
}
} // namespace

void PoissonObjective::compute(floats_view scores, floats_view targets, floats_out grad,
                               floats_out hess)
{
    assert(scores.size() == targets.size());
    parallel::for_each_index(scores.size(),
                             [&](size_t i)
                             {
                                 float const rate = clamped_exp(scores[i]);
                                 grad[i]          = rate - targets[i];
                                 hess[i]          = rate;
                             });
}

float PoissonObjective::eval(floats_view scores, floats_view targets)
{
    assert(scores.size() == targets.size());
    double total = 0.0;
    for (size_t i = 0; i < scores.size(); ++i)
    {
        float const f = std::clamp(scores[i], -k_poisson_max_log, k_poisson_max_log);
        total += static_cast<double>(clamped_exp(f)) -
                 (static_cast<double>(targets[i]) * static_cast<double>(f));
    }
    return static_cast<float>(total / static_cast<double>(scores.size()));
}

float PoissonObjective::init_score(floats_view targets)
{
    double sum = 0.0;
    for (float const y : targets)
    {
        if (y < 0.0F || std::isnan(y))
        {
            throw std::invalid_argument(
                "poisson objective requires non-negative labels");
        }
        sum += y;
    }
    double const mean = sum / static_cast<double>(std::max<size_t>(targets.size(), 1));
    return static_cast<float>(std::log(std::max(mean, 1e-9)));
}

void SoftmaxObjective::compute(floats_view /*preds*/, floats_view /*targets*/,
                               floats_out /*grad*/, floats_out /*hess*/)
{
    throw std::logic_error("softmax objective is handled by MulticlassBooster");
}

float SoftmaxObjective::eval(floats_view /*preds*/, floats_view /*targets*/)
{
    throw std::logic_error("softmax eval needs K columns; use MulticlassBooster::eval");
}

float SoftmaxObjective::init_score(floats_view /*targets*/)
{
    throw std::logic_error("softmax init is handled by MulticlassBooster");
}

} // namespace bonsai
