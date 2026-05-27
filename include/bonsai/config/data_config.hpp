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
    bool             header        = true;
    int              label_column  = 0;
    int              weight_column = -1;
    std::vector<int> ignore_columns;

    bool                 missing_nan      = true;
    std::optional<float> missing_sentinel = std::nullopt;

    bool operator==(DataConfig const &) const = default;
};

} // namespace bonsai
