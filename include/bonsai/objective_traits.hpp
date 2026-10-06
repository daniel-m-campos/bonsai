#pragma once

#include <array>
#include <concepts>
#include <cstdint>
#include <span>
#include <string_view>
#include <variant>

#include "bonsai/objective.hpp"
#include "bonsai/task.hpp"
#include "bonsai/types.hpp"

namespace bonsai
{

// Objectives whose gradient and hessian a device backend can derive from
// resident scores and labels, with no host objective pass and no per-tree
// gradient upload. Squared error is the trivial case: g = score - label,
// h = 1. LogLoss and Poisson add a transcendental per row (sigmoid, exp) but
// stay two-line kernels; MAE, Huber and Quantile renew their leaves from the
// resident residuals after the tree is built. The booster and the CUDA
// engine share this tag (invariants: resident-objective-eligibility), and
// this core header carries no CUDA include.
enum class DeviceObjectiveKind : uint8_t
{
    none,
    mse,
    logloss,
    poisson,
    mae,
    huber,
    quantile,
};

// One alternative per kind, each carrying only the parameters its kernels
// read; an alternative's index in the variant is its kind.
struct DeviceMse
{
};
struct DeviceLogLoss
{
};
struct DevicePoisson
{
};
struct DeviceMae
{
};
struct DeviceHuber
{
    float delta = 1.0F;
};
struct DeviceQuantile
{
    float alpha = 0.5F;
};
using DeviceObjective =
    std::variant<std::monostate, DeviceMse, DeviceLogLoss, DevicePoisson, DeviceMae,
                 DeviceHuber, DeviceQuantile>;

constexpr DeviceObjectiveKind device_kind(DeviceObjective const &objective)
{
    return static_cast<DeviceObjectiveKind>(objective.index());
}

template <typename Form>
inline constexpr DeviceObjectiveKind device_kind_of_form =
    device_kind(DeviceObjective{Form{}});

static_assert(device_kind_of_form<std::monostate> == DeviceObjectiveKind::none);
static_assert(device_kind_of_form<DeviceMse> == DeviceObjectiveKind::mse);
static_assert(device_kind_of_form<DeviceLogLoss> == DeviceObjectiveKind::logloss);
static_assert(device_kind_of_form<DevicePoisson> == DeviceObjectiveKind::poisson);
static_assert(device_kind_of_form<DeviceMae> == DeviceObjectiveKind::mae);
static_assert(device_kind_of_form<DeviceHuber> == DeviceObjectiveKind::huber);
static_assert(device_kind_of_form<DeviceQuantile> == DeviceObjectiveKind::quantile);

constexpr bool renew_leaf_on_device(DeviceObjectiveKind kind)
{
    return kind == DeviceObjectiveKind::mae || kind == DeviceObjectiveKind::huber ||
           kind == DeviceObjectiveKind::quantile;
}

template <typename Objective> struct device_form_of
{
    using type = std::monostate;
};

template <> struct device_form_of<MSEObjective>
{
    using type = DeviceMse;
};

template <> struct device_form_of<LogLossObjective>
{
    using type = DeviceLogLoss;
};

template <> struct device_form_of<PoissonObjective>
{
    using type = DevicePoisson;
};

template <> struct device_form_of<MAEObjective>
{
    using type = DeviceMae;
};

template <> struct device_form_of<HuberObjective>
{
    using type = DeviceHuber;
};

template <> struct device_form_of<QuantileObjective>
{
    using type = DeviceQuantile;
};

template <typename Objective>
using device_form_t = typename device_form_of<Objective>::type;

template <typename Objective>
inline constexpr DeviceObjectiveKind device_objective_kind =
    device_kind_of_form<device_form_t<Objective>>;

template <typename Objective>
device_form_t<Objective> device_objective_of(Objective const &)
{
    return {};
}

inline DeviceHuber device_objective_of(HuberObjective const &objective)
{
    return {objective.delta_};
}

inline DeviceQuantile device_objective_of(QuantileObjective const &objective)
{
    return {objective.alpha_};
}

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
        default_metrics_of<T>::names
    } -> std::convertible_to<std::span<std::string_view const>>;
};

template <> struct default_metrics_of<MSEObjective>
{
    static constexpr std::array<std::string_view, 1> names{"rmse"};
};
template <> struct default_metrics_of<LogLossObjective>
{
    static constexpr std::array<std::string_view, 2> names{"logloss", "accuracy"};
};
template <> struct default_metrics_of<MAEObjective>
{
    static constexpr std::array<std::string_view, 2> names{"mae", "rmse"};
};
template <> struct default_metrics_of<HuberObjective>
{
    static constexpr std::array<std::string_view, 2> names{"mae", "rmse"};
};
template <> struct default_metrics_of<QuantileObjective>
{
    static constexpr std::array<std::string_view, 2> names{"mae", "rmse"};
};
template <> struct default_metrics_of<PoissonObjective>
{
    static constexpr std::array<std::string_view, 2> names{"rmse", "mae"};
};
template <> struct default_metrics_of<SoftmaxObjective>
{
    static constexpr std::array<std::string_view, 1> names{"mc_accuracy"};
};

} // namespace bonsai
