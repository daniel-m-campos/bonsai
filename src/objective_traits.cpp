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

std::string_view task_kind_name(TaskKind kind)
{
    switch (kind)
    {
    case TaskKind::regression:
        return "regression";
    case TaskKind::binary_classification:
        return "binary_classification";
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

} // namespace bonsai
