#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
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

// Quantile-step over `subsample.size()` items yielding at most `cut_budget`
// cuts: the ceiling stride guarantees floor((n-1)/step) <= budget, where the
// old floored stride overshot for subsamples not comfortably above the
// budget — 400 distinct values at max_bin=255 produced 401 bins, silently
// disqualifying small datasets from u8 storage (issue #17, decision 51).
size_t quantile_step(size_t subsample_size, size_t cut_budget)
{
    return std::max((subsample_size + cut_budget) / (cut_budget + 1), 1UL);
}

// Sort once and read the quantile value at each stride: identical to the
// previous shrinking-suffix nth_element per stride (both yield the exact
// k-th order statistic) at a fraction of the element visits. Drops
// duplicates and appends a +inf sentinel.
std::vector<float> create_cuts(std::vector<float> &subsample, size_t step)
{
    std::sort(subsample.begin(), subsample.end());
    std::vector<float> cuts;
    // Duplicate-heavy columns first (issue #61): the quantile stride walks
    // duplicate runs and the dedup below then discards most of the budget —
    // house_sales' 13-distinct-value bedrooms column got SEVEN cuts at
    // max_bin 255. When the subsample's distinct values fit the budget,
    // every distinct value becomes a cut and binning is exact, matching
    // what lightgbm/xgboost do; the stride only kicks in above the budget.
    size_t const budget     = (subsample.size() + step - 1) / step;
    size_t       n_distinct = 0;
    for (size_t k = 0; k < subsample.size(); ++k)
    {
        if (k == 0 || subsample[k] > subsample[k - 1])
        {
            ++n_distinct;
        }
    }
    if (n_distinct <= budget)
    {
        // Every distinct value gets its own right-inclusive cut, so binning
        // is exact and the +inf sentinel bin stays missing-only.
        for (float const v : subsample)
        {
            if (cuts.empty() || cuts.back() < v)
            {
                cuts.push_back(v);
            }
        }
    }
    else
    {
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
    auto cuts = create_cuts(subsample, quantile_step(subsample.size(), cut_budget));
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
