#pragma once

#include <string>
#include <vector>

namespace bonsai
{

// Names of metrics to report during fit / eval. Empty -> use the model's
// objective's default metric names (see registry's default_metric_names_by_name).
struct MetricsConfig
{
    std::vector<std::string> fit;
    std::vector<std::string> eval;

    bool operator==(MetricsConfig const &) const = default;
};

} // namespace bonsai
