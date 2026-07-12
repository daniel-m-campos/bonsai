#include "bonsai/objective_traits.hpp"

#include <array>
#include <cmath>
#include <span>
#include <string_view>

#include "bonsai/objective.hpp"
#include "bonsai/task.hpp"
#include "bonsai/types.hpp"

namespace bonsai
{

namespace
{

inline float sigmoid(float x)
{
    return 1.0F / (1.0F + std::exp(-x));
}

} // namespace

void link_inverse_of<LogLossObjective>::apply(floats_out scores)
{
    for (float &v : scores)
    {
        v = sigmoid(v);
    }
}

void link_inverse_of<PoissonObjective>::apply(floats_out scores)
{
    for (float &v : scores)
    {
        v = std::exp(v);
    }
}

std::string_view task_kind_name(TaskKind kind)
{
    switch (kind)
    {
    case TaskKind::regression:
        return "regression";
    case TaskKind::binary_classification:
        return "binary_classification";
    case TaskKind::multiclass_classification:
        return "multiclass_classification";
    }
    return "unknown";
}

std::span<std::string_view const> default_metrics_of<MSEObjective>::value()
{
    static constexpr auto names = std::array<std::string_view, 1>{"rmse"};
    return names;
}

std::span<std::string_view const> default_metrics_of<LogLossObjective>::value()
{
    static constexpr auto names =
        std::array<std::string_view, 2>{"logloss", "accuracy"};
    return names;
}

std::span<std::string_view const> default_metrics_of<MAEObjective>::value()
{
    static constexpr auto names = std::array<std::string_view, 2>{"mae", "rmse"};
    return names;
}

std::span<std::string_view const> default_metrics_of<HuberObjective>::value()
{
    static constexpr auto names = std::array<std::string_view, 2>{"mae", "rmse"};
    return names;
}

std::span<std::string_view const> default_metrics_of<QuantileObjective>::value()
{
    static constexpr auto names = std::array<std::string_view, 2>{"mae", "rmse"};
    return names;
}

std::span<std::string_view const> default_metrics_of<PoissonObjective>::value()
{
    static constexpr auto names = std::array<std::string_view, 2>{"rmse", "mae"};
    return names;
}

std::span<std::string_view const> default_metrics_of<SoftmaxObjective>::value()
{
    static constexpr auto names = std::array<std::string_view, 1>{"mc_accuracy"};
    return names;
}

} // namespace bonsai
