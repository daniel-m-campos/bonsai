#pragma once

#include <span>
#include <string_view>

#include "bonsai/config/config.hpp"
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

// Look up an objective by name and evaluate its loss on raw scores (no link
// applied — objectives eval in raw-score space). The Config carries loss
// parameters (huber_delta, quantile_alpha). Used by early stopping.
// Throws UnknownImplError if no match.
float eval_objective_by_name(std::string_view objective_name, Config const &cfg,
                             floats_view scores, floats_view labels);

// Look up an objective by name and return the default metric names to report
// when the user has not configured `metrics.fit` / `metrics.eval`. Throws
// UnknownImplError if no match.
std::span<std::string_view const>
default_metric_names_by_name(std::string_view objective_name);

} // namespace bonsai
