#include "bonsai/objective.hpp"
#include "bonsai/parallel.hpp"
#include "bonsai/types.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <numeric>
#include <vector>

namespace bonsai
{

namespace
{

// Value at the given quantile of `values` (0.5 = median), nearest-rank on
// alpha * (n - 1) so float noise in alpha can't shift the index. Copies;
// callers are one-shot init_score paths.
float quantile_of(floats_view values, float alpha)
{
    assert(!values.empty());
    std::vector<float> v(values.begin(), values.end());
    auto const         k = std::min<size_t>(
        v.size() - 1, static_cast<size_t>(std::llround(
                          static_cast<double>(alpha) *
                          static_cast<double>(v.size() - 1))));
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(k), v.end());
    return v[k];
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
    assert(preds.size() == targets.size());
    assert(!preds.empty());
    auto const  n                 = static_cast<float>(preds.size());
    float const sum_squared_error = std::transform_reduce(
        preds.begin(), preds.end(), targets.begin(), 0.0F, std::plus<>(),
        [](auto const p, auto const t)
        {
            double const diff = p - t;
            return diff * diff;
        });

    return sum_squared_error / n;
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
                                 float const p = 1.0F / (1.0F + std::exp(-score));
                                 grad[i]       = p - labels[i];
                                 hess[i]       = p * (1.0F - p);
                             });
}

float LogLossObjective::eval(floats_view scores, floats_view labels)
{
    assert(scores.size() == labels.size());
    assert(!scores.empty());

    auto const  n            = static_cast<float>(scores.size());
    float const sum_log_loss = std::transform_reduce(
        scores.begin(), scores.end(), labels.begin(), 0.0F, std::plus<>(),
        [](auto const score, auto const y)
        {
            // BCE from raw score: softplus(score) - y*score.
            // Stable form: max(0, score) + log1p(exp(-|score|)).
            float const ax = std::abs(score);
            return std::max(0.0F, score) + std::log1p(std::exp(-ax)) - (y * score);
        });

    return sum_log_loss / n;
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
    assert(preds.size() == targets.size());
    assert(!preds.empty());
    float const sum = std::transform_reduce(preds.begin(), preds.end(),
                                            targets.begin(), 0.0F, std::plus<>(),
                                            [](auto const p, auto const t)
                                            { return std::abs(p - t); });
    return sum / static_cast<float>(preds.size());
}

auto MAEObjective::init_score(floats_view targets) -> floats_view::value_type
{
    return quantile_of(targets, 0.5F);
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
    assert(preds.size() == targets.size());
    assert(!preds.empty());
    float const d   = delta_;
    float const sum = std::transform_reduce(
        preds.begin(), preds.end(), targets.begin(), 0.0F, std::plus<>(),
        [d](auto const p, auto const t)
        {
            float const a = std::abs(p - t);
            return a <= d ? 0.5F * a * a : d * (a - (0.5F * d));
        });
    return sum / static_cast<float>(preds.size());
}

auto HuberObjective::init_score(floats_view targets) -> floats_view::value_type
{
    return quantile_of(targets, 0.5F);
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void QuantileObjective::compute(floats_view preds, floats_view targets,
                                floats_out grad, floats_out hess) const
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
    assert(preds.size() == targets.size());
    assert(!preds.empty());
    float const a   = alpha_;
    float const sum = std::transform_reduce(
        preds.begin(), preds.end(), targets.begin(), 0.0F, std::plus<>(),
        [a](auto const p, auto const t)
        { return t >= p ? a * (t - p) : (1.0F - a) * (p - t); });
    return sum / static_cast<float>(preds.size());
}

auto QuantileObjective::init_score(floats_view targets) const
    -> floats_view::value_type
{
    return quantile_of(targets, alpha_);
}

} // namespace bonsai
