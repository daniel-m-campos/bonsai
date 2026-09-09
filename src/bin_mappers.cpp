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
// column of the block instead of one float per line at a 64 KiB stride. The
// block's column streams sit 32 floats past a multiple of the row count
// apart: at a power-of-two row count an unpadded stride lands every stream
// in one cache set, which is why the first sweep read blocks of 16 and 32
// slower than 8 (0.38 s and 0.46 s against 0.29 s at 32768 x 16384 on an
// M2, 8 threads). Padded and with the NaN drop as an unconditional store
// plus a conditional advance, that sweep reads 0.121 s at 16 and 0.101 s
// at 32 (0.116 s at 64), against 1.16 s one column at a time. The sort
// runs in place on the block, so no column is copied out.
constexpr size_t k_column_block      = 32;
constexpr size_t k_stream_pad        = 32;
constexpr size_t k_floats_per_line   = 16;
constexpr size_t k_prefetch_ahead    = 16;
constexpr size_t k_blocks_per_worker = 4;

size_t column_block_width(size_t n_features)
{
    size_t const per_worker =
        n_features / (static_cast<size_t>(parallel::n_threads()) * k_blocks_per_worker);
    return std::clamp<size_t>(per_worker, 1, k_column_block);
}

struct ColumnBlock
{
    size_t              stride;
    std::vector<float>  storage;
    std::vector<size_t> fill;

    auto cells()
    {
        return std::mdspan(storage.data(), fill.size(), stride);
    }

    std::span<float> column(size_t j)
    {
        return {&cells()[j, 0], fill[j]};
    }
};

ColumnBlock gather_columns(features_view X, std::span<uint32_t const> rows,
                           size_t first, size_t width)
{
    size_t const stride = rows.size() + k_stream_pad;
    ColumnBlock  block{stride, std::vector<float>(width * stride),
                      std::vector<size_t>(width, 0)};
    auto const   cells = block.cells();
    size_t      *fill  = block.fill.data();
    for (size_t i = 0; i < rows.size(); ++i)
    {
        if (i + k_prefetch_ahead < rows.size())
        {
            float const *ahead = &X[rows[i + k_prefetch_ahead], first];
            for (size_t j = 0; j < width; j += k_floats_per_line)
            {
                __builtin_prefetch(ahead + j);
            }
        }
        float const *row = &X[rows[i], first];
        for (size_t j = 0; j < width; ++j)
        {
            float const v     = row[j];
            cells[j, fill[j]] = v;
            fill[j] += static_cast<size_t>(!std::isnan(v));
        }
    }
    return block;
}

} // namespace

std::vector<std::optional<BinMapper>> mappers_from_edges(BinEdges const &bin_edges,
                                                         size_t          n_features)
{
    std::vector<std::optional<BinMapper>> slots(n_features);
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
    return slots;
}

std::vector<BinMapper> resolve_mappers(std::vector<std::optional<BinMapper>> &slots)
{
    std::vector<BinMapper> mappers;
    mappers.reserve(slots.size());
    for (auto &s : slots)
    {
        mappers.push_back(std::move(*s)); // NOLINT(bugprone-unchecked-optional-access)
    }
    return mappers;
}

BinMappers BinMappers::fit(detail::ColumnBatch const &batch, BinMapperConfig const &cfg,
                           BinEdges const &bin_edges)
{
    detail::IngestProfiler::Lap lap;
    size_t const n_rows = batch.features.empty() ? 0 : batch.features[0].size();
    auto const   rows   = sample_or_all_rows(n_rows, cfg);
    auto         slots  = mappers_from_edges(bin_edges, batch.features.size());
    parallel::for_each_index(batch.features.size(),
                             [&](size_t f)
                             {
                                 if (slots[f])
                                 {
                                     return;
                                 }
                                 auto const &col = batch.features[f];
                                 auto        sample =
                                     gather(rows, [&](size_t r) { return col[r]; });
                                 slots[f] = BinMapper::from_sample(sample, cfg);
                             });
    lap(detail::IngestProfiler::instance().fit_s);

    BinMappers out;
    out.mappers_       = resolve_mappers(slots);
    out.feature_names_ = batch.feature_names;
    return out;
}

BinMappers BinMappers::fit(features_view X, std::vector<std::string> feature_names,
                           BinMapperConfig const &cfg, BinEdges const &bin_edges)
{
    detail::IngestProfiler::Lap lap;
    size_t const                n        = X.extent(0);
    size_t const                f        = X.extent(1);
    auto const                  rows     = sample_or_all_rows(n, cfg);
    auto                        slots    = mappers_from_edges(bin_edges, f);
    size_t const                width    = column_block_width(f);
    size_t const                n_blocks = (f + width - 1) / width;
    parallel::for_each_index(n_blocks,
                             [&](size_t b)
                             {
                                 size_t const first = b * width;
                                 size_t const count = std::min(width, f - first);
                                 auto block = gather_columns(X, rows, first, count);
                                 for (size_t j = 0; j < count; ++j)
                                 {
                                     if (slots[first + j])
                                     {
                                         continue;
                                     }
                                     slots[first + j] =
                                         BinMapper::from_sample(block.column(j), cfg);
                                 }
                             });
    lap(detail::IngestProfiler::instance().fit_s);

    return from_mappers(resolve_mappers(slots), std::move(feature_names));
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
