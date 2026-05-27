#pragma once

#include <concepts>
#include <span>
#include <string_view>

#include "bonsai/objective.hpp"
#include "bonsai/task.hpp"
#include "bonsai/types.hpp"

namespace bonsai
{

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

// Task this objective serves. Determines which metrics are compatible.
template <> struct task_of<MSEObjective>
{
    static constexpr TaskKind value = TaskKind::regression;
};
template <> struct task_of<LogLossObjective>
{
    static constexpr TaskKind value = TaskKind::binary_classification;
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

} // namespace bonsai
