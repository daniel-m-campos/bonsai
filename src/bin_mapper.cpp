#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <ranges>
#include <utility>
#include <vector>

#include "bonsai/bin_mapper.hpp"
#include "bonsai/config/bin_mapper_config.hpp"
#include "bonsai/types.hpp"

namespace bonsai
{

namespace
{

bool is_not_nan(float x)
{
    return !std::isnan(x);
}

// Build a NaN-free working set (unsorted). Either copies the column whole
// or draws a seeded reservoir sample.
std::vector<float> create_subsample(floats_view column, BinMapperConfig const &cfg)
{
    std::vector<float> subsample;
    if (column.size() <= cfg.n_samples)
    {
        std::ranges::copy_if(column, std::back_inserter(subsample), is_not_nan);
    }
    else
    {
        subsample.reserve(cfg.n_samples);
        std::ranges::sample(
            column | std::views::filter(is_not_nan), std::back_inserter(subsample),
            static_cast<std::ptrdiff_t>(cfg.n_samples), std::mt19937(cfg.seed));
    }
    return subsample;
}

// Count-weighted greedy allocation for columns where some value is heavier
// than a mean-sized bin (issue #63, decision 57; lightgbm's GreedyFindBin):
// heavy values get a bin to themselves, the rest fill toward the running
// mean, and cuts land at the midpoint BETWEEN adjacent distinct values so
// no value straddles a boundary — the equal-frequency stride let a heavy
// value's run swallow its neighbours' resolution.
std::vector<float> greedy_weighted_cuts(std::vector<float> const  &vals,
                                        std::vector<size_t> const &counts,
                                        size_t n_samples, size_t cut_budget,
                                        double mean_bin)
{
    std::vector<float> cuts;
    size_t const       n_groups = cut_budget + 1;
    std::vector<bool>  is_big(vals.size());
    size_t             n_big   = 0;
    size_t             big_sum = 0;
    for (size_t i = 0; i < vals.size(); ++i)
    {
        if (static_cast<double>(counts[i]) >= mean_bin)
        {
            is_big[i] = true;
            ++n_big;
            big_sum += counts[i];
        }
    }
    size_t rest_sum    = n_samples - big_sum;
    size_t rest_groups = n_groups - n_big;
    double bin_size    = rest_groups != 0U ? static_cast<double>(rest_sum) /
                                              static_cast<double>(rest_groups)
                                           : mean_bin;
    size_t in_bin      = 0;
    for (size_t i = 0; i + 1 < vals.size() && cuts.size() < cut_budget; ++i)
    {
        if (!is_big[i])
        {
            rest_sum -= counts[i];
        }
        in_bin += counts[i];
        // Close the bin when it is a heavy value's own bin, has reached
        // the running mean, or a heavy value comes next and this bin is
        // at least half full (avoids slivers just before heavy values).
        if (is_big[i] || static_cast<double>(in_bin) >= bin_size ||
            (is_big[i + 1] && static_cast<double>(in_bin) >= bin_size / 2.0))
        {
            float const mid = std::midpoint(vals[i], vals[i + 1]);
            if (cuts.empty() || cuts.back() < mid)
            {
                cuts.push_back(mid);
            }
            in_bin = 0;
            if (!is_big[i] && rest_groups > 1)
            {
                --rest_groups;
                bin_size =
                    static_cast<double>(rest_sum) / static_cast<double>(rest_groups);
            }
        }
    }
    return cuts;
}

// Sort once, run-length encode into (distinct value, count) pairs, then
// place at most `cut_budget` cuts by the lightest rule that is exact for
// the column's shape. Distinct values within the budget: one right-inclusive
// cut per value — binning is exact and the +inf sentinel bin stays
// missing-only (issue #61). Above the budget with every value lighter than
// a mean bin: the quantile stride, which IS the count-weighted allocation
// when no value can overflow a bin (ceiling stride per decision 51). Only
// heavy-value columns pay the greedy walk (issue #63) — the rule is per
// column, so all-continuous datasets are untouched by it.
std::vector<float> create_cuts(std::vector<float> &subsample, size_t cut_budget)
{
    std::sort(subsample.begin(), subsample.end());
    std::vector<float>  vals;
    std::vector<size_t> counts;
    for (float const v : subsample)
    {
        if (vals.empty() || vals.back() < v)
        {
            vals.push_back(v);
            counts.push_back(0);
        }
        ++counts.back();
    }

    std::vector<float> cuts;
    double const       mean_bin =
        static_cast<double>(subsample.size()) / static_cast<double>(cut_budget + 1);
    if (vals.size() <= cut_budget)
    {
        cuts = std::move(vals);
    }
    else if (static_cast<double>(std::ranges::max(counts)) >= mean_bin)
    {
        cuts =
            greedy_weighted_cuts(vals, counts, subsample.size(), cut_budget, mean_bin);
    }
    else
    {
        size_t const step =
            std::max((subsample.size() + cut_budget) / (cut_budget + 1), 1UL);
        for (size_t k = step; k < subsample.size(); k += step)
        {
            float const v = subsample[k];
            if (cuts.empty() || cuts.back() < v)
            {
                cuts.push_back(v);
            }
        }
    }
    cuts.push_back(std::numeric_limits<float>::infinity());
    return cuts;
}

} // namespace

BinMapper BinMapper::fit(floats_view column, BinMapperConfig const &cfg)
{
    assert(cfg.max_bin > 2);
    // 1 bin for the +inf sentinel, another for the missing slot.
    size_t const cut_budget = cfg.max_bin - 2;
    auto         subsample  = create_subsample(column, cfg);
    auto         cuts       = create_cuts(subsample, cut_budget);
    return BinMapper{std::move(cuts)};
}

bin_id_t BinMapper::transform(float x) const
{
    if (std::isnan(x))
    {
        return n_bins() - 1;
    }
    return std::ranges::lower_bound(cuts_, x) - cuts_.begin();
}

} // namespace bonsai
