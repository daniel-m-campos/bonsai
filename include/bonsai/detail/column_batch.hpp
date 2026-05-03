#pragma once

#include <string>
#include <vector>

namespace bonsai::detail
{

struct ColumnBatch
{
    std::vector<std::vector<float>> features; // [n_features][n_rows]
    std::vector<float> labels;                // [n_rows]
    std::vector<float> weights;               // empty if uniform
    std::vector<std::string> feature_names;   // [n_features]
};

} // namespace bonsai::detail
