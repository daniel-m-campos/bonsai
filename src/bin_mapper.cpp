#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <random>
#include <ranges>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "bonsai/bin_mapper.hpp"
#include "bonsai/config/bin_mapper_config.hpp"

namespace bonsai
{

namespace
{

bool is_not_nan(float x)
{
    return !std::isnan(x);
}

// Build a sorted, NaN-free working set. Either copies the column whole
// (when it already fits in n_samples) or draws a seeded reservoir sample.
std::vector<float> create_sorted_subsample(std::span<float const> column,
                                           BinMapperConfig const &cfg)
{
    std::vector<float> subsample;
    if (column.size() <= cfg.n_samples)
    {
        std::ranges::copy_if(column, std::back_inserter(subsample), is_not_nan);
    }
    else
    {
        subsample.reserve(cfg.n_samples);
        std::ranges::sample(column | std::views::filter(is_not_nan),
                            std::back_inserter(subsample), cfg.n_samples,
                            std::mt19937(cfg.seed));
    }
    std::ranges::sort(subsample);
    return subsample;
}

// Quantile-step over `subsample.size()` items targeting `cut_budget` cuts,
// floored at 1 so dense subsamples still produce strictly-increasing cuts.
size_t quantile_step(size_t subsample_size, size_t cut_budget)
{
    return std::max(subsample_size / cut_budget, 1UL);
}

// Walk the sorted subsample with stride `step`, dropping duplicates,
// and append a +inf sentinel as the right-edge of the final bin.
std::vector<float> create_cuts(std::span<float const> subsample, size_t step)
{
    std::vector<float> cuts;
    if (!subsample.empty())
    {
        size_t const seed_idx = std::min(step, subsample.size() - 1);
        cuts.push_back(subsample[seed_idx]);
        for (size_t i = seed_idx + step; i < subsample.size(); i += step)
        {
            if (cuts.back() < subsample[i])
            {
                cuts.push_back(subsample[i]);
            }
        }
    }
    cuts.push_back(std::numeric_limits<float>::infinity());
    return cuts;
}

} // namespace

BinMapper BinMapper::fit(std::span<float const> column, BinMapperConfig const &cfg)
{
    assert(cfg.max_bin > 2);
    // 1 bin for the +inf sentinel, another for the missing slot.
    size_t const cut_budget = cfg.max_bin - 2;
    auto const subsample    = create_sorted_subsample(column, cfg);
    auto cuts = create_cuts(subsample, quantile_step(subsample.size(), cut_budget));
    return {std::move(cuts)};
}

uint16_t BinMapper::transform(float x) const
{
    if (std::isnan(x))
    {
        return n_buckets() - 1;
    }
    return std::ranges::lower_bound(cuts_, x) - cuts_.begin();
}

} // namespace bonsai
