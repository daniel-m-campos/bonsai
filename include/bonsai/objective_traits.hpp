#pragma once

#include <concepts>
#include <cstdint>
#include <span>
#include <string_view>

#include "bonsai/objective.hpp"
#include "bonsai/task.hpp"
#include "bonsai/types.hpp"

namespace bonsai
{

// Objectives whose gradient and hessian a device backend can derive from
// resident scores and labels, with no host objective pass and no per-tree
// gradient upload. Squared error is the trivial case: g = score - label,
// h = 1. LogLoss and Poisson add a transcendental per row (sigmoid, exp) but
// stay two-line kernels. The booster and the CUDA engine share this tag so the
// resident seam is decided once, and this core header carries no CUDA include.
enum class DeviceObjectiveKind : uint8_t
{
    none,
    mse,
    logloss,
    poisson,
};

template <typename Objective> struct device_objective_kind_of
{
    static constexpr DeviceObjectiveKind value = DeviceObjectiveKind::none;
};

template <> struct device_objective_kind_of<MSEObjective>
{
    static constexpr DeviceObjectiveKind value = DeviceObjectiveKind::mse;
};

template <> struct device_objective_kind_of<LogLossObjective>
{
    static constexpr DeviceObjectiveKind value = DeviceObjectiveKind::logloss;
};

template <> struct device_objective_kind_of<PoissonObjective>
{
    static constexpr DeviceObjectiveKind value = DeviceObjectiveKind::poisson;
};

template <typename Objective>
inline constexpr DeviceObjectiveKind device_objective_kind =
    device_objective_kind_of<Objective>::value;

// Inverse link function for objective T, applied in place. Identity for
// regression objectives; sigmoid for binary classification. CLI-only concern
// (the training spine talks to raw scores), kept as an external trait so the
// Objective concept does not have to grow a member it would not use.
//
// Specialize per impl, alongside the entries in registry/names.hpp.
template <typename T> struct link_inverse_of;

template <typename T>
concept HasLinkInverse = requires(floats_out scores) {
    { link_inverse_of<T>::apply(scores) } -> std::same_as<void>;
};

template <> struct link_inverse_of<MSEObjective>
{
    static void apply(floats_out /*scores*/) {}
};

template <> struct link_inverse_of<LogLossObjective>
{
    static void apply(floats_out scores);
};

template <> struct link_inverse_of<MAEObjective>
{
    static void apply(floats_out /*scores*/) {}
};
template <> struct link_inverse_of<HuberObjective>
{
    static void apply(floats_out /*scores*/) {}
};
template <> struct link_inverse_of<QuantileObjective>
{
    static void apply(floats_out /*scores*/) {}
};
template <> struct link_inverse_of<SoftmaxObjective>
{
    // Multiclass predict emits argmax class ids; nothing to invert.
    static void apply(floats_out /*scores*/) {}
};

template <> struct link_inverse_of<PoissonObjective>
{
    static void apply(floats_out scores); // exp: raw log-rates -> rates
};

// Task this objective serves. Determines which metrics are compatible.
template <> struct task_of<MSEObjective>
{
    static constexpr TaskKind value = TaskKind::regression;
};
template <> struct task_of<LogLossObjective>
{
    static constexpr TaskKind value = TaskKind::binary_classification;
};
template <> struct task_of<MAEObjective>
{
    static constexpr TaskKind value = TaskKind::regression;
};
template <> struct task_of<HuberObjective>
{
    static constexpr TaskKind value = TaskKind::regression;
};
template <> struct task_of<QuantileObjective>
{
    static constexpr TaskKind value = TaskKind::regression;
};
template <> struct task_of<SoftmaxObjective>
{
    static constexpr TaskKind value = TaskKind::multiclass_classification;
};
template <> struct task_of<PoissonObjective>
{
    static constexpr TaskKind value = TaskKind::regression;
};

// Default metric names to report when the user does not set `metrics.fit` /
// `metrics.eval`. Resolved against the Metric registry at call time. Mirrors
// LightGBM/XGBoost's "objective declares its default eval metric" idea.
template <typename T> struct default_metrics_of;

template <typename T>
concept HasDefaultMetricNames = requires {
    {
        default_metrics_of<T>::value()
    } -> std::convertible_to<std::span<std::string_view const>>;
};

template <> struct default_metrics_of<MSEObjective>
{
    static std::span<std::string_view const> value();
};
template <> struct default_metrics_of<LogLossObjective>
{
    static std::span<std::string_view const> value();
};
template <> struct default_metrics_of<MAEObjective>
{
    static std::span<std::string_view const> value();
};
template <> struct default_metrics_of<HuberObjective>
{
    static std::span<std::string_view const> value();
};
template <> struct default_metrics_of<QuantileObjective>
{
    static std::span<std::string_view const> value();
};
template <> struct default_metrics_of<PoissonObjective>
{
    static std::span<std::string_view const> value();
};
template <> struct default_metrics_of<SoftmaxObjective>
{
    static std::span<std::string_view const> value();
};

} // namespace bonsai
