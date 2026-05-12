#pragma once

#include "bonsai/config/tree_config.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/types.hpp"
#include <concepts>
#include <vector>

namespace bonsai
{

struct SplitNode
{
    std::vector<Histogram> hists;
    std::vector<row_id_t> rows;
    double grad  = 0.0;
    double hess  = 0.0;
    node_id_t id = 0;
};

struct Split
{
    double gain             = 0.0;
    feature_id_t feature_id = 0;
    bin_id_t bin_id         = 0;
    bool default_left       = true;
    bool valid              = false;
};

template <typename T>
concept SplitFinder = requires(SplitNode const &node, TreeConfig const &config) {
    { T::find(node, config) } -> std::same_as<Split>;
};

struct HistogramSplitFinder
{
    static Split find(SplitNode const &node, TreeConfig const &config);
};

inline double score(double g, double h, double lambda)
{
    return (g * g) / (h + lambda);
}

} // namespace bonsai
