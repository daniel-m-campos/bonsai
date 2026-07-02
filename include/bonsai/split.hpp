#pragma once

#include "bonsai/config/tree_config.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/types.hpp"
#include <concepts>
#include <span>
#include <vector>

namespace bonsai
{

struct SplitInput
{
    std::vector<Histogram> hists;
    std::vector<row_id_t>  rows;
    node_id_t              id = 0;

    // Node-level totals from the first populated histogram (every populated
    // feature sums the same rows). Unselected features under
    // feature_fraction < 1 are zero-binned placeholders and are skipped.
    HistCell totals() const
    {
        for (auto const &h : hists)
        {
            if (h.size() != 0)
            {
                return h.totals();
            }
        }
        return {};
    }
    double total_grad() const
    {
        return totals().sum_grad;
    }
    double total_hess() const
    {
        return totals().sum_hess;
    }
};

struct SplitOutput
{
    double       gain         = 0.0;
    feature_id_t feature_id   = 0;
    bin_id_t     bin_id       = 0;
    bool         default_left = true;
    bool         valid        = false;
};

template <typename T>
concept NodeSplitFinder = requires(SplitInput const &input, TreeConfig const &config) {
    { T::find(input, config) } -> std::same_as<SplitOutput>;
};

struct HistogramNodeSplitFinder
{
    static SplitOutput find(SplitInput const &input, TreeConfig const &config);
};

using FrontierInput = std::span<SplitInput const>;

template <typename T>
concept LevelSplitFinder = requires(FrontierInput frontier, TreeConfig const &config) {
    { T::find(frontier, config) } -> std::same_as<SplitOutput>;
};

struct HistogramLevelSplitFinder
{
    static SplitOutput find(FrontierInput frontier, TreeConfig const &config);
};

inline double score(double g, double h, double lambda)
{
    return (g * g) / (h + lambda);
}

} // namespace bonsai
