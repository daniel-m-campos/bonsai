#include "bonsai/metric.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <format>
#include <functional>
#include <limits>
#include <numeric>
#include <string_view>
#include <vector>

#include "bonsai/task.hpp"
#include "bonsai/types.hpp"

namespace bonsai
{

float compute_rmse(floats_view preds, floats_view labels)
{
    assert(preds.size() == labels.size());
    assert(!preds.empty());
    auto const  n  = static_cast<float>(preds.size());
    float const ss = std::transform_reduce(preds.begin(), preds.end(), labels.begin(),
                                           0.0F, std::plus<>(),
                                           [](float const p, float const t)
                                           {
                                               double const d = p - t;
                                               return d * d;
                                           });
    return std::sqrt(ss / n);
}

float compute_mae(floats_view preds, floats_view labels)
{
    assert(preds.size() == labels.size());
    assert(!preds.empty());
    auto const  n = static_cast<float>(preds.size());
    float const s = std::transform_reduce(
        preds.begin(), preds.end(), labels.begin(), 0.0F, std::plus<>(),
        [](float const p, float const t) { return std::abs(p - t); });
    return s / n;
}

float compute_r2(floats_view preds, floats_view labels)
{
    assert(preds.size() == labels.size());
    assert(!preds.empty());
    auto const n      = static_cast<double>(labels.size());
    double     mean_y = 0.0;
    for (float const y : labels)
    {
        mean_y += static_cast<double>(y);
    }
    mean_y /= n;

    double ss_res = 0.0;
    double ss_tot = 0.0;
    for (size_t i = 0; i < labels.size(); ++i)
    {
        double const d_res =
            static_cast<double>(preds[i]) - static_cast<double>(labels[i]);
        double const d_tot = static_cast<double>(labels[i]) - mean_y;
        ss_res += d_res * d_res;
        ss_tot += d_tot * d_tot;
    }
    if (ss_tot == 0.0)
    {
        // Constant labels: r2 undefined. Return NaN (honest signal that the
        // input is degenerate); sklearn's 0.0 default is more forgiving but
        // hides data issues.
        return std::numeric_limits<float>::quiet_NaN();
    }
    return static_cast<float>(1.0 - (ss_res / ss_tot));
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
float compute_logloss(floats_view probs, floats_view labels)
{
    assert(probs.size() == labels.size());
    assert(!probs.empty());
    constexpr double eps = 1e-7;
    double           ll  = 0.0;
    for (size_t i = 0; i < probs.size(); ++i)
    {
        double const p = probs[i];
        double const t = labels[i] > 0.5F ? 1.0 : 0.0;
        ll -= (t * std::log(std::max(p, eps))) +
              ((1.0 - t) * std::log(std::max(1.0 - p, eps)));
    }
    return static_cast<float>(ll / static_cast<double>(probs.size()));
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
float compute_accuracy(floats_view probs, floats_view labels)
{
    assert(probs.size() == labels.size());
    assert(!probs.empty());
    size_t correct = 0;
    for (size_t i = 0; i < probs.size(); ++i)
    {
        bool const pred_pos  = probs[i] >= 0.5F;
        bool const label_pos = labels[i] >= 0.5F;
        if (pred_pos == label_pos)
        {
            ++correct;
        }
    }
    return static_cast<float>(static_cast<double>(correct) /
                              static_cast<double>(probs.size()));
}

// AUC via the rank-sum (Mann-Whitney) identity: sort by score ascending,
// sum the positives' ranks, tie groups get their average rank.
float compute_auc(floats_view probs, floats_view labels)
{
    assert(probs.size() == labels.size());
    assert(!probs.empty());
    size_t const n = probs.size();

    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), size_t{0});
    std::ranges::sort(order, [&](size_t a, size_t b) { return probs[a] < probs[b]; });

    double rank_sum_pos = 0.0;
    size_t n_pos        = 0;
    size_t i            = 0;
    while (i < n)
    {
        size_t j = i;
        while (j < n && probs[order[j]] == probs[order[i]])
        {
            ++j;
        }
        // average 1-based rank of the tie group [i, j)
        double const avg_rank = 0.5 * (static_cast<double>(i + 1 + j));
        for (size_t k = i; k < j; ++k)
        {
            if (labels[order[k]] >= 0.5F)
            {
                rank_sum_pos += avg_rank;
                ++n_pos;
            }
        }
        i = j;
    }
    size_t const n_neg = n - n_pos;
    if (n_pos == 0 || n_neg == 0)
    {
        return 1.0F; // degenerate: one class only
    }
    double const u = rank_sum_pos - (static_cast<double>(n_pos) * (n_pos + 1) / 2.0);
    return static_cast<float>(u / (static_cast<double>(n_pos) * n_neg));
}

float compute_mc_accuracy(floats_view class_ids, floats_view labels)
{
    assert(class_ids.size() == labels.size());
    assert(!class_ids.empty());
    size_t correct = 0;
    for (size_t i = 0; i < class_ids.size(); ++i)
    {
        correct += std::lround(class_ids[i]) == std::lround(labels[i]) ? 1 : 0;
    }
    return static_cast<float>(correct) / static_cast<float>(class_ids.size());
}

namespace
{

// Tiny hand-written registry. No for_each_type machinery -- five entries.
// Adding a metric is one line below + matching free function above.
inline constexpr auto all_metrics = std::array<Metric, 7>{{
    metric_rmse,
    metric_mae,
    metric_r2,
    metric_logloss,
    metric_accuracy,
    metric_auc,
    metric_mc_accuracy,
}};

} // namespace

Metric find_metric(std::string_view name)
{
    for (auto const &m : all_metrics)
    {
        if (m.name == name)
        {
            return m;
        }
    }
    throw MetricNotFoundError(
        std::format("metric '{}' is not a built-in metric", name));
}

Metric resolve_metric_for_task(std::string_view name, TaskKind task)
{
    auto const m = find_metric(name);
    if (m.task != task)
    {
        throw MetricTaskMismatchError(
            std::format("metric '{}' is a {} metric but the model is for {}", name,
                        task_kind_name(m.task), task_kind_name(task)));
    }
    return m;
}

} // namespace bonsai
