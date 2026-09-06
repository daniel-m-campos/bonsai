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
#include <span>
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

void push_if_above_last(std::vector<float> &out, float v)
{
    if (out.empty() || out.back() < v)
    {
        out.push_back(v);
    }
}

struct HeavyMarks
{
    std::vector<bool> is_heavy;
    size_t            n_heavy   = 0;
    size_t            heavy_sum = 0;
};

HeavyMarks mark_heavy_values(std::span<size_t const> counts, double mean_bin)
{
    HeavyMarks marks{std::vector<bool>(counts.size())};
    for (size_t i = 0; i < counts.size(); ++i)
    {
        if (static_cast<double>(counts[i]) < mean_bin)
        {
            continue;
        }
        marks.is_heavy[i] = true;
        ++marks.n_heavy;
        marks.heavy_sum += counts[i];
    }
    return marks;
}

bool closes_a_bin(bool heavy, bool next_heavy, size_t in_bin, double bin_size)
{
    auto const filled = static_cast<double>(in_bin);
    return heavy || filled >= bin_size || (next_heavy && filled >= bin_size / 2.0);
}

std::vector<float> greedy_weighted_cuts(std::span<float const>  vals,
                                        std::span<size_t const> counts,
                                        size_t n_samples, size_t cut_budget,
                                        double mean_bin)
{
    std::vector<float> cuts;
    auto const [is_heavy, n_heavy, heavy_sum] = mark_heavy_values(counts, mean_bin);
    size_t rest_sum                           = n_samples - heavy_sum;
    size_t rest_groups                        = cut_budget + 1 - n_heavy;
    double bin_size = rest_groups != 0U ? static_cast<double>(rest_sum) /
                                              static_cast<double>(rest_groups)
                                        : mean_bin;
    size_t in_bin   = 0;
    for (size_t i = 0; i + 1 < vals.size() && cuts.size() < cut_budget; ++i)
    {
        if (!is_heavy[i])
        {
            rest_sum -= counts[i];
        }
        in_bin += counts[i];
        if (!closes_a_bin(is_heavy[i], is_heavy[i + 1], in_bin, bin_size))
        {
            continue;
        }
        push_if_above_last(cuts, std::midpoint(vals[i], vals[i + 1]));
        in_bin = 0;
        if (!is_heavy[i] && rest_groups > 1)
        {
            --rest_groups;
            bin_size = static_cast<double>(rest_sum) / static_cast<double>(rest_groups);
        }
    }
    return cuts;
}

using ByteHistograms = std::array<std::array<size_t, 256>, 4>;

uint32_t sortable_key(float f)
{
    auto const b = std::bit_cast<uint32_t>(f);
    return b ^ ((b >> 31U) != 0U ? 0xFFFFFFFFU : 0x80000000U);
}

float key_to_float(uint32_t k)
{
    return std::bit_cast<float>((k >> 31U) != 0U ? k ^ 0x80000000U : ~k);
}

uint32_t key_byte(uint32_t k, unsigned pass)
{
    return (k >> (8U * pass)) & 0xFFU;
}

ByteHistograms transform_keys(std::span<float const> v, uint32_t *keys)
{
    ByteHistograms hist{};
    for (size_t i = 0; i < v.size(); ++i)
    {
        uint32_t const k = sortable_key(v[i]);
        keys[i]          = k;
        for (unsigned pass = 0; pass < 4; ++pass)
        {
            ++hist[pass][key_byte(k, pass)];
        }
    }
    return hist;
}

void exclusive_prefix(std::array<size_t, 256> &counts)
{
    size_t sum = 0;
    for (auto &c : counts)
    {
        size_t const bucket = c;
        c                   = sum;
        sum += bucket;
    }
}

// perf: LSD byte-radix sort for the NaN-free subsample: the standard order-
// preserving key transform (flip all bits of negatives, flip the sign bit
// of non-negatives) makes unsigned byte passes order floats like operator<.
// The output equals std::sort's up to reordering within equal-comparing
// values (only -0.0 vs +0.0: the key transform puts -0.0 first, unstable
// std::sort may not), so a mixed-zero run's representative in run_lengths
// can differ in SIGN across the two paths. Binning and predictions are
// unaffected: lower_bound under operator< treats the zeros as equal, and
// std::midpoint agrees on either. All four byte histograms come from the
// key transform pass, and a byte every key shares skips its scatter (the
// low bytes of small integers, the high byte of a one-signed column).
// Small inputs keep std::sort: the histogram pass only pays past ~2k
// elements. Single thread, 256 columns of 131072 rows on an M2, sort plus
// run lengths: 0.247 s with a count pass per byte and a push_back
// run-length loop, 0.133 s this way (0.478 s to 0.157 s on integer-valued
// columns, where two of the four scatters skip).
void sort_floats(std::span<float> v)
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
    ByteHistograms hist = transform_keys(v, keys.data());
    uint32_t      *src  = keys.data();
    uint32_t      *dst  = scratch.data();
    for (unsigned pass = 0; pass < 4; ++pass)
    {
        auto &offsets = hist[pass];
        if (offsets[key_byte(src[0], pass)] == n)
        {
            continue;
        }
        exclusive_prefix(offsets);
        for (size_t i = 0; i < n; ++i)
        {
            dst[offsets[key_byte(src[i], pass)]++] = src[i];
        }
        std::swap(src, dst);
    }
    for (size_t i = 0; i < n; ++i)
    {
        v[i] = key_to_float(src[i]);
    }
}

struct RunLengths
{
    std::span<float const>  vals;
    std::span<size_t const> counts;
};

RunLengths run_lengths(std::span<float const> sorted)
{
    static thread_local std::vector<float>  vals;
    static thread_local std::vector<size_t> ends;
    size_t const                            n = sorted.size();
    if (n == 0)
    {
        return {};
    }
    vals.resize(n);
    ends.resize(n);
    float  *run_val = vals.data();
    size_t *run_end = ends.data();
    float   prev    = sorted[0];
    size_t  m       = 0;
    run_val[0]      = prev;
    for (size_t i = 0; i < n; ++i)
    {
        float const f = sorted[i];
        m += static_cast<size_t>(prev < f);
        run_val[m] = f;
        run_end[m] = i + 1;
        prev       = f;
    }
    for (size_t r = m; r > 0; --r)
    {
        run_end[r] -= run_end[r - 1];
    }
    return {{run_val, m + 1}, {run_end, m + 1}};
}

std::vector<float> create_cuts(std::span<float> subsample, size_t cut_budget)
{
    sort_floats(subsample);
    auto const [vals, counts] = run_lengths(subsample);

    std::vector<float> cuts;
    double const       mean_bin =
        static_cast<double>(subsample.size()) / static_cast<double>(cut_budget + 1);
    if (vals.size() <= cut_budget)
    {
        cuts.assign(vals.begin(), vals.end());
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
            push_if_above_last(cuts, subsample[k]);
        }
    }
    push_if_above_last(cuts, std::numeric_limits<float>::max());
    cuts.push_back(std::numeric_limits<float>::infinity());
    return cuts;
}

} // namespace

BinMapper BinMapper::fit(floats_view column, BinMapperConfig const &cfg)
{
    assert(cfg.max_bin > 2);
    auto sample = create_subsample(column, cfg);
    return from_sample(sample, cfg);
}

BinMapper BinMapper::from_sample(std::span<float> sample, BinMapperConfig const &cfg)
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
    auto const missing_bin = static_cast<bin_id_t>(n_bins() - 1);
    if (std::isnan(x))
    {
        return missing_bin;
    }
    auto const top_band_bin = static_cast<bin_id_t>(missing_bin - 1);
    auto const bin =
        static_cast<bin_id_t>(std::ranges::lower_bound(cuts_, x) - cuts_.begin());
    return std::min(bin, top_band_bin);
}

} // namespace bonsai
