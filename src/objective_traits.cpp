#include "bonsai/objective_traits.hpp"

#include <cmath>
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

} // namespace bonsai
