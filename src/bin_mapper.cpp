#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <ranges>
#include <utility>
#include <vector>

#include "bonsai/bin_mapper.hpp"
#include "bonsai/config/bin_mapper_config.hpp"
#include "bonsai/config/errors.hpp"
#include "bonsai/types.hpp"

namespace bonsai
{

namespace
{

bool is_not_nan(float x)
{
    return !std::isnan(x);
}

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

// perf: LSD byte-radix sort for the NaN-free subsample: the standard order-
// preserving key transform (flip all bits of negatives, flip the sign bit
// of non-negatives) makes unsigned byte passes order floats like operator<.
// The output equals std::sort's up to reordering within equal-comparing
// values (only -0.0 vs +0.0: the key transform puts -0.0 first, unstable
// std::sort may not), so a mixed-zero run's RLE representative below can
// differ in SIGN across the two paths. Binning and predictions are
// unaffected: lower_bound under operator< treats the zeros as equal, and
// std::midpoint agrees on either. Small inputs keep std::sort: four
// counting passes only pay past ~2k elements.
// Motivation: the mapper fit is sort-bound at wide shapes (11.5s of a
// 54.9s GPU fit at 131k x 16384).
void sort_floats(std::vector<float> &v)
{
    constexpr size_t k_radix_min = 2048;
    if (v.size() < k_radix_min)
    {
        std::sort(v.begin(), v.end());
        return;
    }
    size_t const                              n = v.size();
    static thread_local std::vector<uint32_t> keys;
    static thread_local std::vector<uint32_t> scratch;
    keys.resize(n);
    scratch.resize(n);
    for (size_t i = 0; i < n; ++i)
    {
        auto b  = std::bit_cast<uint32_t>(v[i]);
        keys[i] = b ^ ((b >> 31U) != 0U ? 0xFFFFFFFFU : 0x80000000U);
    }
    uint32_t *src = keys.data();
    uint32_t *dst = scratch.data();
    for (unsigned shift = 0; shift < 32; shift += 8)
    {
        std::array<size_t, 257> count{};
        for (size_t i = 0; i < n; ++i)
        {
            ++count[((src[i] >> shift) & 0xFFU) + 1];
        }
        for (size_t b = 1; b < 257; ++b)
        {
            count[b] += count[b - 1];
        }
        for (size_t i = 0; i < n; ++i)
        {
            dst[count[(src[i] >> shift) & 0xFFU]++] = src[i];
        }
        std::swap(src, dst);
    }
    for (size_t i = 0; i < n; ++i)
    {
        uint32_t const k = src[i];
        v[i] = std::bit_cast<float>((k >> 31U) != 0U ? k ^ 0x80000000U : ~k);
    }
}

std::vector<float> create_cuts(std::vector<float> &subsample, size_t cut_budget)
{
    sort_floats(subsample);
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
    if (cuts.empty() || cuts.back() < std::numeric_limits<float>::max())
    {
        cuts.push_back(std::numeric_limits<float>::max());
    }
    cuts.push_back(std::numeric_limits<float>::infinity());
    return cuts;
}

} // namespace

BinMapper BinMapper::fit(floats_view column, BinMapperConfig const &cfg)
{
    assert(cfg.max_bin > 2);
    return from_sample(create_subsample(column, cfg), cfg);
}

BinMapper BinMapper::from_sample(std::vector<float> sample, BinMapperConfig const &cfg)
{
    assert(cfg.max_bin > 2);
    assert(std::ranges::none_of(sample, [](float x) { return std::isnan(x); }));
    size_t const cut_budget = cfg.max_bin - 2;
    auto         cuts       = create_cuts(sample, cut_budget);
    return BinMapper{std::move(cuts)};
}

BinMapper BinMapper::from_edges(std::vector<float> edges)
{
    if (edges.empty())
    {
        throw ConfigError("bin_edges: a column's edge list must not be empty");
    }
    for (size_t i = 0; i < edges.size(); ++i)
    {
        if (!std::isfinite(edges[i]) || edges[i] >= std::numeric_limits<float>::max())
        {
            throw ConfigError("bin_edges: edges must be finite (the top band "
                              "and the missing bin are implicit)");
        }
        if (i > 0 && edges[i] <= edges[i - 1])
        {
            throw ConfigError("bin_edges: edges must be strictly increasing");
        }
    }
    edges.push_back(std::numeric_limits<float>::max());
    edges.push_back(std::numeric_limits<float>::infinity());
    return BinMapper{std::move(edges)};
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
