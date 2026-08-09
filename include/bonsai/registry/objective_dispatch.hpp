#pragma once

#include <span>
#include <string_view>

#include "bonsai/task.hpp"
#include "bonsai/types.hpp"

namespace bonsai
{

// Look up an objective by name in the Objectives typelist and apply its
// inverse link function in place. Throws UnknownImplError if no match.
void apply_link_inverse_by_name(std::string_view objective_name, floats_out scores);

// Look up an objective by name and return its TaskKind. Lets the CLI know
// which metrics are compatible without naming the objective type.
// Throws UnknownImplError if no match.
TaskKind task_kind_by_name(std::string_view objective_name);

// Look up an objective by name and return the default metric names to report
// when the user has not configured `metrics.fit` / `metrics.eval`. Throws
// UnknownImplError if no match.
std::span<std::string_view const>
default_metric_names_by_name(std::string_view objective_name);

} // namespace bonsai
