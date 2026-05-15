#include "bonsai/objective.hpp"
#include "bonsai/types.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <numeric>

namespace bonsai
{

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void MSEObjective::compute(floats_view preds, floats_view targets, floats_out grad,
                           floats_out hess)
{
    assert(preds.size() == targets.size());
    assert(targets.size() == grad.size());
    assert(grad.size() == hess.size());

    for (size_t i = 0; i < preds.size(); ++i)
    {
        grad[i] = preds[i] - targets[i];
        hess[i] = 1.0F;
    }
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
auto MSEObjective::eval(floats_view preds, floats_view targets)
    -> floats_view::value_type
{
    assert(preds.size() == targets.size());
    assert(!preds.empty());
    auto const n                  = static_cast<float>(preds.size());
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
    auto const n    = static_cast<float>(targets.size());
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

    for (size_t i = 0; i < scores.size(); ++i)
    {
        float const score = scores[i];
        float const p     = 1.0F / (1.0F + std::exp(-score));
        grad[i]           = p - labels[i];
        hess[i]           = p * (1.0F - p);
    }
}

float LogLossObjective::eval(floats_view scores, floats_view labels)
{
    assert(scores.size() == labels.size());
    assert(!scores.empty());

    auto const n             = static_cast<float>(scores.size());
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
    auto const n    = static_cast<float>(labels.size());
    float const sum = std::accumulate(labels.begin(), labels.end(), 0.0F);
    float const p   = sum / n;
    return std::log(p / (1.0F - p));
}

} // namespace bonsai
