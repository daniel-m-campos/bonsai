#pragma once

#include <optional>
#include <string>
#include <vector>

namespace bonsai
{

struct DataConfig
{
    std::string              train;
    std::vector<std::string> valid;
    std::string              test;

    std::string      format        = "csv";
    // libsvm only: force the feature count (0 = infer from max index).
    // Needed when a test split's highest feature index is below train's.
    int libsvm_n_features = 0;
    bool             header        = true;
    int              label_column  = 0;
    int              weight_column = -1;
    std::vector<int> ignore_columns;

    bool                 missing_nan      = true;
    std::optional<float> missing_sentinel = std::nullopt;

    bool operator==(DataConfig const &) const = default;
};

} // namespace bonsai
