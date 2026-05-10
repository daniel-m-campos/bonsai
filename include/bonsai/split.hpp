#pragma once

#include "bonsai/histogram.hpp"
#include "bonsai/types.hpp"
#include <concepts>

namespace bonsai
{

struct Split
{
    double gain             = 0.0;
    feature_id_t feature_id = 0;
    // Cut is inclusive on the left: left = bins [0, bin_id], right = bins (bin_id, n_real).
    bin_id_t bin_id   = 0;
    bool default_left = true;
    bool valid        = false;
};

template <typename T>
concept SplitFinder = requires(histogram_view_t hists, typename T::Params params) {
    typename T::Params;
    { T::find(hists, params) } -> std::same_as<Split>;
};

struct HistogramSplitFinder
{
    struct Params
    {
        double parent_score;
        double lambda_l2      = 1.0;
        double min_child_hess = 1.0;
    };

    static Split find(histogram_view_t hists, Params params);
};

} // namespace bonsai
