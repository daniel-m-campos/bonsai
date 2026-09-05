#include "bonsai/bin_mappers.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <optional>
#include <random>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "bonsai/bin_mapper.hpp"
#include "bonsai/config/bin_mapper_config.hpp"
#include "bonsai/config/errors.hpp"
#include "bonsai/detail/column_batch.hpp"
#include "bonsai/detail/perf.hpp"
#include "bonsai/parallel.hpp"

namespace bonsai
{

// perf: One shared row sample for the whole matrix: every feature's
// cuts come from the same rows, so the O(n) selection pass runs once instead of
// once per feature (mapper-fit was ~5-8s of a 16M fit). Empty result means
// "n_rows <= n_samples, use every row": the whole-column path, unchanged and
// bit-identical for datasets that fit the sample.
std::vector<uint32_t> bin_sample_rows(size_t n_rows, BinMapperConfig const &cfg)
{
    if (n_rows <= cfg.n_samples)
    {
        return {};
    }
    std::vector<uint32_t> picked;
    picked.reserve(cfg.n_samples);
    std::ranges::sample(std::views::iota(uint32_t{0}, static_cast<uint32_t>(n_rows)),
                        std::back_inserter(picked),
                        static_cast<std::ptrdiff_t>(cfg.n_samples),
                        std::mt19937(cfg.seed));
    return picked;
}

namespace
{

std::vector<uint32_t> sample_or_all_rows(size_t n_rows, BinMapperConfig const &cfg)
{
    auto rows = bin_sample_rows(n_rows, cfg);
    if (rows.empty())
    {
        rows.resize(n_rows);
        std::iota(rows.begin(), rows.end(), uint32_t{0});
    }
    return rows;
}

void push_present(std::vector<float> &out, float v)
{
    if (!std::isnan(v))
    {
        out.push_back(v);
    }
}

template <typename ColumnFn>
std::vector<float> gather(std::span<uint32_t const> rows, ColumnFn value)
{
    std::vector<float> out;
    out.reserve(rows.size());
    for (uint32_t const r : rows)
    {
        push_present(out, value(r));
    }
    return out;
}

// perf: A row-major matrix is gathered a block of adjacent columns per row
// pass, with the row 16 ahead prefetched, so a row's cache line serves every
// column of the block instead of one float per line at a 64 KiB stride. On
// an M2 at 32768 x 16384 the gather reads 1.16 s one column at a time,
// 0.46 s in blocks of 8 without the prefetch, 0.29 s with it; blocks of 16
// and 32 read 0.38 s and 0.46 s, and 16 rows ahead sits on the plateau
// between 8 (0.29 s) and 32 (0.38 s).
constexpr size_t k_column_block      = 8;
constexpr size_t k_prefetch_ahead    = 16;
constexpr size_t k_blocks_per_worker = 4;

size_t column_block_width(size_t n_features)
{
    size_t const per_worker =
        n_features / (static_cast<size_t>(parallel::n_threads()) * k_blocks_per_worker);
    return std::clamp<size_t>(per_worker, 1, k_column_block);
}

std::vector<std::vector<float>> gather_columns(features_view             X,
                                               std::span<uint32_t const> rows,
                                               size_t first, size_t width)
{
    std::vector<std::vector<float>> out(width);
    for (auto &col : out)
    {
        col.reserve(rows.size());
    }
    for (size_t i = 0; i < rows.size(); ++i)
    {
        if (i + k_prefetch_ahead < rows.size())
        {
            __builtin_prefetch(&X[rows[i + k_prefetch_ahead], first]);
        }
        for (size_t j = 0; j < width; ++j)
        {
            push_present(out[j], X[rows[i], first + j]);
        }
    }
    return out;
}

// sync: seeded before the parallel::for_each_index in fit, because
// from_edges validates and throws ConfigError, and a throw must not cross
// the region.
void seed_edge_slots(BinEdges const &bin_edges, size_t n_features,
                     std::vector<std::optional<BinMapper>> &slots)
{
    for (auto const &[col, edges] : bin_edges)
    {
        if (col >= n_features)
        {
            throw ConfigError("bin_edges: column " + std::to_string(col) +
                              " is out of range for " + std::to_string(n_features) +
                              " features");
        }
        if (slots[col])
        {
            throw ConfigError("bin_edges: column " + std::to_string(col) +
                              " listed twice");
        }
        slots[col] = BinMapper::from_edges(edges);
    }
}

} // namespace

BinMappers BinMappers::fit(detail::ColumnBatch const &batch, BinMapperConfig const &cfg,
                           BinEdges const &bin_edges)
{
    detail::IngestProfiler::Lap lap;
    size_t const n_rows = batch.features.empty() ? 0 : batch.features[0].size();
    auto const   rows   = sample_or_all_rows(n_rows, cfg);
    std::vector<std::optional<BinMapper>> slots(batch.features.size());
    seed_edge_slots(bin_edges, batch.features.size(), slots);
    parallel::for_each_index(batch.features.size(),
                             [&](size_t f)
                             {
                                 if (slots[f])
                                 {
                                     return;
                                 }
                                 auto const &col = batch.features[f];
                                 slots[f]        = BinMapper::from_sample(
                                     gather(rows, [&](size_t r) { return col[r]; }),
                                     cfg);
                             });
    lap(detail::IngestProfiler::instance().fit_s);

    BinMappers out;
    out.mappers_.reserve(slots.size());
    for (auto &s : slots)
    {
        out.mappers_.push_back(
            std::move(*s)); // NOLINT(bugprone-unchecked-optional-access)
    }
    out.feature_names_ = batch.feature_names;
    return out;
}

BinMappers BinMappers::fit(features_view X, std::vector<std::string> feature_names,
                           BinMapperConfig const &cfg, BinEdges const &bin_edges)
{
    detail::IngestProfiler::Lap           lap;
    size_t const                          n    = X.extent(0);
    size_t const                          f    = X.extent(1);
    auto const                            rows = sample_or_all_rows(n, cfg);
    std::vector<std::optional<BinMapper>> slots(f);
    seed_edge_slots(bin_edges, f, slots);
    size_t const width    = column_block_width(f);
    size_t const n_blocks = (f + width - 1) / width;
    parallel::for_each_index(n_blocks,
                             [&](size_t b)
                             {
                                 size_t const first = b * width;
                                 size_t const count = std::min(width, f - first);
                                 auto samples = gather_columns(X, rows, first, count);
                                 for (size_t j = 0; j < count; ++j)
                                 {
                                     if (slots[first + j])
                                     {
                                         continue;
                                     }
                                     slots[first + j] = BinMapper::from_sample(
                                         std::move(samples[j]), cfg);
                                 }
                             });
    lap(detail::IngestProfiler::instance().fit_s);

    std::vector<BinMapper> mappers;
    mappers.reserve(f);
    for (auto &s : slots)
    {
        mappers.push_back(std::move(*s)); // NOLINT(bugprone-unchecked-optional-access)
    }
    return from_mappers(std::move(mappers), std::move(feature_names));
}

BinMapper const &BinMappers::operator[](size_t fid) const
{
    return mappers_[fid];
}

size_t BinMappers::size() const
{
    return mappers_.size();
}

std::span<std::string const> BinMappers::feature_names() const
{
    return feature_names_;
}

bool BinMappers::same_cuts(BinMappers const &other) const
{
    if (mappers_.size() != other.mappers_.size())
    {
        return false;
    }
    for (size_t f = 0; f < mappers_.size(); ++f)
    {
        if (!std::ranges::equal(mappers_[f].cuts(), other.mappers_[f].cuts()))
        {
            return false;
        }
    }
    return true;
}

} // namespace bonsai
