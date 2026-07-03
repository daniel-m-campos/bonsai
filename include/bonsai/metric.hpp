#pragma once

#include <stdexcept>
#include <string_view>

#include "bonsai/task.hpp"
#include "bonsai/types.hpp"

namespace bonsai
{

// Type-erased metric callable. Function pointer (not std::function or virtual)
// so Metric values can be `inline constexpr`. CRTP / deducing-this don't fit
// here because the registry needs a homogeneous std::array<Metric, N>; static
// polymorphism would produce N different types. The "avoid raw pointers" rule
// is about ownership/lifetime -- function pointers point at code, don't own,
// don't have a lifetime to manage. Same idiom as LinkFn in objective_dispatch.
using MetricFn = float (*)(floats_view preds, floats_view labels);

// A metric is a named eval-time scalar function for a specific TaskKind.
// Metrics see link-applied predictions (probabilities for binary, raw scores
// for regression -- the link inverse is identity there).
struct Metric
{
    std::string_view name;
    TaskKind         task;
    MetricFn         compute;
};

// Compute functions for the five built-ins. Free functions so Metric values
// can refer to them at compile time via &compute_*.
float compute_rmse(floats_view preds, floats_view labels);
float compute_mae(floats_view preds, floats_view labels);
float compute_r2(floats_view preds, floats_view labels);
float compute_logloss(floats_view probs, floats_view labels);
float compute_accuracy(floats_view probs, floats_view labels);
float compute_auc(floats_view probs, floats_view labels);

inline constexpr Metric metric_rmse{
    .name = "rmse", .task = TaskKind::regression, .compute = &compute_rmse};
inline constexpr Metric metric_mae{
    .name = "mae", .task = TaskKind::regression, .compute = &compute_mae};
inline constexpr Metric metric_r2{
    .name = "r2", .task = TaskKind::regression, .compute = &compute_r2};
inline constexpr Metric metric_logloss{.name    = "logloss",
                                       .task    = TaskKind::binary_classification,
                                       .compute = &compute_logloss};
inline constexpr Metric metric_accuracy{.name    = "accuracy",
                                        .task    = TaskKind::binary_classification,
                                        .compute = &compute_accuracy};
inline constexpr Metric metric_auc{.name    = "auc",
                                   .task    = TaskKind::binary_classification,
                                   .compute = &compute_auc};

class MetricNotFoundError : public std::runtime_error
{
  public:
    using std::runtime_error::runtime_error;
};

class MetricTaskMismatchError : public std::runtime_error
{
  public:
    using std::runtime_error::runtime_error;
};

// Look up a metric by name across all built-ins. Throws MetricNotFoundError
// if name is unknown.
Metric find_metric(std::string_view name);

// Look up a metric by name and verify it serves the given task. Throws
// MetricNotFoundError if name unknown, MetricTaskMismatchError on task drift.
Metric resolve_metric_for_task(std::string_view name, TaskKind task);

} // namespace bonsai
