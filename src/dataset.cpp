#include "bonsai/dataset.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "bonsai/bin_mappers.hpp"
#include "bonsai/config/data_config.hpp"
#include "bonsai/detail/column_batch.hpp"
#include "bonsai/parallel.hpp"
#include "bonsai/types.hpp"
#include "ingest_profiler.hpp"

namespace bonsai
{

namespace
{

bool all_fit_u8(BinMappers const &mappers)
{
    for (size_t f = 0; f < mappers.size(); ++f)
    {
        if (mappers[f].n_bins() > 256)
        {
            return false;
        }
    }
    return true;
}

// Shared bin loop: `read(f, r)` yields the raw float for (row, feature);
// values are identical either width, so models stay byte-identical. Workers
// own row tiles and visit every feature within the tile, so a row-major
// source is pulled into cache once per tile instead of once per feature (a
// straight column pass reads X[r, f] at n_features x 4B stride — ~25x line
// amplification at 100 features). The tile size only reorders independent
// writes; 64 u8 rows = one cache line per column, so tiles never share one.
template <typename Read>
void fill_binned(std::vector<std::vector<uint8_t>>  &u8,
                 std::vector<std::vector<uint16_t>> &u16, bool &u8_mode,
                 size_t n_features, size_t n_rows, BinMappers const &mappers, Read read)
{
    u8_mode = all_fit_u8(mappers);
    if (u8_mode)
    {
        u8.resize(n_features);
    }
    else
    {
        u16.resize(n_features);
    }
    parallel::for_each_index(n_features,
                             [&](size_t f)
                             {
                                 if (u8_mode)
                                 {
                                     u8[f].resize(n_rows);
                                 }
                                 else
                                 {
                                     u16[f].resize(n_rows);
                                 }
                             });
    constexpr size_t tile = 64;
    parallel::for_each_index((n_rows + tile - 1) / tile,
                             [&](size_t block)
                             {
                                 size_t const r0 = block * tile;
                                 size_t const r1 = std::min(r0 + tile, n_rows);
                                 for (size_t f = 0; f < n_features; ++f)
                                 {
                                     auto const &mapper = mappers[f];
                                     if (u8_mode)
                                     {
                                         uint8_t *const out = u8[f].data();
                                         for (size_t r = r0; r < r1; ++r)
                                         {
                                             out[r] = static_cast<uint8_t>(
                                                 mapper.transform(read(f, r)));
                                         }
                                     }
                                     else
                                     {
                                         uint16_t *const out = u16[f].data();
                                         for (size_t r = r0; r < r1; ++r)
                                         {
                                             out[r] = mapper.transform(read(f, r));
                                         }
                                     }
                                 }
                             });
}

} // namespace

// DataConfig is reserved for future sentinel-aware binning (cfg.missing_sentinel,
// cfg.missing_nan). Today ColumnBatch is assumed to carry NaNs for missing, so
// the reader layer is responsible for any sentinel-to-NaN conversion upstream.
Dataset Dataset::bin(detail::ColumnBatch const &batch, BinMappers const &mappers,
                     DataConfig const & /*cfg*/)
{
    assert(batch.features.size() == mappers.size());
    detail::IngestProfiler::Lap lap;
    Dataset                     ds;
    ds.n_rows_  = batch.labels.size();
    ds.mappers_ = mappers;
    ds.labels_  = batch.labels;
    ds.weights_ = batch.weights;
    ds.is_categorical_.assign(batch.features.size(), false);
    fill_binned(ds.features_u8_, ds.features_u16_, ds.bins_are_u8_,
                batch.features.size(), ds.n_rows_, mappers,
                [&](size_t f, size_t r) { return batch.features[f][r]; });
    lap(detail::IngestProfiler::instance().bin_s);
    return ds;
}

Dataset Dataset::bin(features_view X, floats_view labels, BinMappers const &mappers,
                     DataConfig const & /*cfg*/)
{
    assert(X.extent(1) == mappers.size());
    detail::IngestProfiler::Lap lap;
    Dataset                     ds;
    ds.n_rows_  = labels.size();
    ds.mappers_ = mappers;
    ds.labels_.assign(labels.begin(), labels.end());
    ds.is_categorical_.assign(X.extent(1), false);
    fill_binned(ds.features_u8_, ds.features_u16_, ds.bins_are_u8_, X.extent(1),
                ds.n_rows_, mappers, [&](size_t f, size_t r) { return X[r, f]; });
    lap(detail::IngestProfiler::instance().bin_s);
    return ds;
}

size_t Dataset::n_rows() const
{
    return n_rows_;
}

size_t Dataset::n_features() const
{
    return bins_are_u8_ ? features_u8_.size() : features_u16_.size();
}

floats_view Dataset::labels() const
{
    return labels_;
}

floats_view Dataset::weights() const
{
    return weights_;
}

BinMappers const &Dataset::mappers() const
{
    return mappers_;
}

size_t Dataset::n_bins(size_t fid) const
{
    return mappers_[fid].n_bins();
}

bool Dataset::is_categorical(size_t fid) const
{
    return is_categorical_[fid];
}

std::span<uint8_t const> Dataset::row_major_bins() const
{
    if (!bins_are_u8_)
    {
        return {};
    }
    if (!row_major_)
    {
        size_t const f   = features_u8_.size();
        auto         rm  = std::make_shared<std::vector<uint8_t>>(n_rows_ * f);
        uint8_t     *out = rm->data();
        // Tiled column-to-row transpose: each worker owns a row block, so
        // writes never overlap and the mirror is byte-identical at any
        // thread count.
        constexpr size_t tile = 64;
        parallel::for_each_index((n_rows_ + tile - 1) / tile,
                                 [&](size_t block)
                                 {
                                     size_t const r0 = block * tile;
                                     size_t const r1 = std::min(r0 + tile, n_rows_);
                                     for (size_t c0 = 0; c0 < f; c0 += tile)
                                     {
                                         size_t const c1 = std::min(c0 + tile, f);
                                         for (size_t c = c0; c < c1; ++c)
                                         {
                                             uint8_t const *col =
                                                 features_u8_[c].data();
                                             for (size_t r = r0; r < r1; ++r)
                                             {
                                                 out[(r * f) + c] = col[r];
                                             }
                                         }
                                     }
                                 });
        row_major_ = std::move(rm);
    }
    return *row_major_;
}

} // namespace bonsai
