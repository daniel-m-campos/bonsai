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

// Quantile-step over `subsample.size()` items targeting `cut_budget` cuts,
// floored at 1 so dense subsamples still produce strictly-increasing cuts.
size_t quantile_step(size_t subsample_size, size_t cut_budget)
{
    return std::max(subsample_size / cut_budget, 1UL);
}

// Pull out quantile values at stride `step` using nth_element on a
// shrinking suffix. Drops duplicates and appends a +inf sentinel.
std::vector<float> create_cuts(std::vector<float> &subsample, size_t step)
{
    std::vector<float> cuts;
    auto               lo = subsample.begin();
    for (size_t k = step; k < subsample.size(); k += step)
    {
        auto target = subsample.begin() + static_cast<std::ptrdiff_t>(k);
        std::nth_element(lo, target, subsample.end());
        if (cuts.empty() || cuts.back() < *target)
        {
            cuts.push_back(*target);
        }
        lo = target + 1;
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
