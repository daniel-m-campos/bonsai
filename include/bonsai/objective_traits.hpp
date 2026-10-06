#pragma once

#include <array>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <span>
#include <string_view>
#include <type_traits>
#include <variant>

#include "bonsai/objective.hpp"
#include "bonsai/task.hpp"
#include "bonsai/types.hpp"

namespace bonsai
{

// Objectives a device backend derives from resident scores and labels with
// no host objective pass and no per-tree gradient upload, one alternative per
// objective. Each carries only its own parameters and states its gradient,
// hessian and loss as constexpr members, which the CUDA compiler treats as
// callable from device code, so a kernel instantiated on the alternative
// reads the formula from the type; a renewing alternative names the quantile
// its leaves take (invariants: resident-objective-eligibility). This core
// header carries no CUDA include.
struct UnitHessian
{
    static constexpr bool unit_hessian = true;
    constexpr float       hessian(float /*score*/, float /*label*/) const
    {
        return 1.0F;
    }
};

struct NewtonLeaf
{
    static constexpr bool renew_leaf = false;
};

struct MedianLeaf
{
    static constexpr bool renew_leaf = true;
    constexpr float       leaf_quantile() const
    {
        return 0.5F;
    }
};

struct DeviceMse : UnitHessian, NewtonLeaf
{
    constexpr float gradient(float score, float label) const
    {
        return score - label;
    }
    constexpr double loss(float score, float label) const
    {
        double const d = static_cast<double>(score) - static_cast<double>(label);
        return d * d;
    }
};

struct DeviceLogLoss : NewtonLeaf
{
    static constexpr bool unit_hessian = false;
    constexpr float       gradient(float score, float label) const
    {
        return (1.0F / (1.0F + expf(-score))) - label;
    }
    constexpr float hessian(float score, float /*label*/) const
    {
        float const p = 1.0F / (1.0F + expf(-score));
        return p * (1.0F - p);
    }
    constexpr double loss(float score, float label) const
    {
        return fmaxf(0.0F, score) + log1pf(expf(-fabsf(score))) - (label * score);
    }
};

struct DevicePoisson : NewtonLeaf
{
    static constexpr bool unit_hessian = false;
    constexpr float       clamped_score(float score) const
    {
        return fminf(fmaxf(score, -k_poisson_max_log), k_poisson_max_log);
    }
    constexpr float gradient(float score, float label) const
    {
        return expf(clamped_score(score)) - label;
    }
    constexpr float hessian(float score, float /*label*/) const
    {
        return expf(clamped_score(score));
    }
    constexpr double loss(float score, float label) const
    {
        float const f = clamped_score(score);
        return static_cast<double>(expf(f)) -
               (static_cast<double>(label) * static_cast<double>(f));
    }
};

struct DeviceMae : UnitHessian, MedianLeaf
{
    constexpr float gradient(float score, float label) const
    {
        float const r = score - label;
        return r > 0.0F ? 1.0F : (r < 0.0F ? -1.0F : 0.0F);
    }
    constexpr double loss(float score, float label) const
    {
        return fabsf(score - label);
    }
};

struct DeviceHuber : UnitHessian, MedianLeaf
{
    float delta = 1.0F;

    constexpr DeviceHuber() = default;
    constexpr explicit DeviceHuber(float delta_in) : delta(delta_in) {}
    constexpr float gradient(float score, float label) const
    {
        return fminf(fmaxf(score - label, -delta), delta);
    }
    constexpr double loss(float score, float label) const
    {
        float const a = fabsf(score - label);
        return a <= delta ? 0.5F * a * a : delta * (a - (0.5F * delta));
    }
};

struct DeviceQuantile : UnitHessian
{
    float                 alpha      = 0.5F;
    static constexpr bool renew_leaf = true;

    constexpr DeviceQuantile() = default;
    constexpr explicit DeviceQuantile(float alpha_in) : alpha(alpha_in) {}
    constexpr float leaf_quantile() const
    {
        return alpha;
    }
    constexpr float gradient(float score, float label) const
    {
        return score > label ? (1.0F - alpha) : -alpha;
    }
    constexpr double loss(float score, float label) const
    {
        return label >= score ? alpha * (label - score)
                              : (1.0F - alpha) * (score - label);
    }
};

using DeviceObjective =
    std::variant<std::monostate, DeviceMse, DeviceLogLoss, DevicePoisson, DeviceMae,
                 DeviceHuber, DeviceQuantile>;

template <typename Form>
inline constexpr bool is_device_form = !std::same_as<Form, std::monostate>;

constexpr bool has_device_form(DeviceObjective const &objective)
{
    return !std::holds_alternative<std::monostate>(objective);
}

template <typename Fn>
constexpr auto with_device_form(DeviceObjective const &objective, Fn &&fn, auto none)
{
    return std::visit(
        [&](auto const &form)
        {
            if constexpr (is_device_form<std::decay_t<decltype(form)>>)
            {
                return fn(form);
            }
            else
            {
                return none;
            }
        },
        objective);
}

template <typename Fn>
constexpr void for_device_form(DeviceObjective const &objective, Fn &&fn)
{
    with_device_form(
        objective,
        [&](auto const &form)
        {
            fn(form);
            return true;
        },
        false);
}

constexpr bool renew_leaf_on_device(DeviceObjective const &objective)
{
    return with_device_form(
        objective, [](auto const &form) { return form.renew_leaf; }, false);
}

constexpr bool unit_hessian_on_device(DeviceObjective const &objective)
{
    return with_device_form(
        objective, [](auto const &form) { return form.unit_hessian; }, false);
}

constexpr float leaf_quantile_on_device(DeviceObjective const &objective)
{
    return with_device_form(
        objective,
        [](auto const &form)
        {
            if constexpr (form.renew_leaf)
            {
                return form.leaf_quantile();
            }
            else
            {
                return 0.5F;
            }
        },
        0.5F);
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
inline constexpr bool has_device_objective = is_device_form<device_form_t<Objective>>;

template <typename Objective>
device_form_t<Objective> device_objective_of(Objective const &)
{
    return {};
}

inline DeviceHuber device_objective_of(HuberObjective const &objective)
{
    return DeviceHuber{objective.delta_};
}

inline DeviceQuantile device_objective_of(QuantileObjective const &objective)
{
    return DeviceQuantile{objective.alpha_};
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
