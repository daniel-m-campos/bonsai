#include "bonsai/dataset.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bonsai/bin_mappers.hpp"
#include "bonsai/config/data_config.hpp"
#include "bonsai/detail/column_batch.hpp"
#include "bonsai/detail/perf.hpp"
#include "bonsai/parallel.hpp"
#include "bonsai/types.hpp"

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

// DataConfig is unused here: NaN is the missing marker on every path, and the
// columns already carry it whether they came from a reader or from a caller's
// matrix.
Dataset Dataset::bin(detail::ColumnBatch const &batch, BinMappers const &mappers,
                     DataConfig const & /*cfg*/,
                     std::shared_ptr<IngestPlane const> plane)
{
    assert(batch.features.size() == mappers.size());
    detail::Phase<&detail::IngestProfiler::bin_s> phase;
    Dataset                                       ds;
    ds.n_rows_     = batch.labels.size();
    ds.n_features_ = batch.features.size();
    ds.rows_       = RowView::all(ds.n_rows_);
    ds.mappers_    = mappers;
    ds.labels_     = batch.labels;
    ds.weights_    = batch.weights;
    if (plane)
    {
        ds.plane_       = std::move(plane);
        ds.lazy_        = std::make_shared<HostBins>();
        ds.bins_are_u8_ = all_fit_u8(mappers);
    }
    else
    {
        fill_binned(ds.cols_->u8, ds.cols_->u16, ds.bins_are_u8_, batch.features.size(),
                    ds.n_rows_, mappers,
                    [&](size_t f, size_t r) { return batch.features[f][r]; });
    }
    return ds;
}

Dataset Dataset::bin(features_view X, floats_view labels, BinMappers const &mappers,
                     DataConfig const &cfg, std::shared_ptr<IngestPlane const> plane,
                     floats_view weights)
{
    assert(X.extent(1) == mappers.size());
    if (plane)
    {
        return bin(labels.size(), X.extent(1), labels, mappers, cfg, std::move(plane),
                   weights);
    }
    detail::Phase<&detail::IngestProfiler::bin_s> phase;
    Dataset                                       ds;
    ds.n_rows_     = labels.size();
    ds.n_features_ = X.extent(1);
    ds.rows_       = RowView::all(ds.n_rows_);
    ds.mappers_    = mappers;
    ds.labels_.assign(labels.begin(), labels.end());
    ds.weights_.assign(weights.begin(), weights.end());
    fill_binned(ds.cols_->u8, ds.cols_->u16, ds.bins_are_u8_, X.extent(1), ds.n_rows_,
                mappers, [&](size_t f, size_t r) { return X[r, f]; });
    return ds;
}

Dataset Dataset::bin(size_t n_rows, size_t n_features, floats_view labels,
                     BinMappers const &mappers, DataConfig const & /*cfg*/,
                     std::shared_ptr<IngestPlane const> plane, floats_view weights)
{
    assert(plane != nullptr);
    assert(n_features == mappers.size());
    detail::Phase<&detail::IngestProfiler::bin_s> phase;
    Dataset                                       ds;
    ds.n_rows_     = n_rows;
    ds.n_features_ = n_features;
    ds.rows_       = RowView::all(ds.n_rows_);
    ds.mappers_    = mappers;
    ds.labels_.assign(labels.begin(), labels.end());
    ds.weights_.assign(weights.begin(), weights.end());
    ds.plane_       = std::move(plane);
    ds.lazy_        = std::make_shared<HostBins>();
    ds.bins_are_u8_ = all_fit_u8(mappers);
    return ds;
}

Dataset Dataset::from_bins(std::vector<std::vector<uint8_t>>  u8,
                           std::vector<std::vector<uint16_t>> u16, bool bins_are_u8,
                           BinMappers mappers, floats_view labels, floats_view weights)
{
    size_t const n_features = bins_are_u8 ? u8.size() : u16.size();
    if (n_features == 0)
    {
        throw std::invalid_argument(
            "Dataset::from_bins: no binned columns of the declared width");
    }
    if (mappers.size() != n_features)
    {
        throw std::invalid_argument(
            "Dataset::from_bins: " + std::to_string(mappers.size()) + " bin mapper" +
            (mappers.size() == 1 ? "" : "s") + " for " + std::to_string(n_features) +
            " binned columns");
    }
    size_t const n_rows = labels.size();
    for (size_t f = 0; f < n_features; ++f)
    {
        size_t const held = bins_are_u8 ? u8[f].size() : u16[f].size();
        if (held != n_rows)
        {
            throw std::invalid_argument(
                "Dataset::from_bins: column " + std::to_string(f) + " holds " +
                std::to_string(held) + " bins and this dataset holds " +
                std::to_string(n_rows) + " rows");
        }
    }
    // Default-constructed, never copied from the dataset the bins came from:
    // cols_ and row_major_ are shared across Dataset copies by design, so a
    // copy here would serve that dataset's bins and its row-major mirror.
    Dataset ds;
    ds.n_rows_     = n_rows;
    ds.n_features_ = n_features;
    ds.rows_       = RowView::all(n_rows);
    ds.mappers_    = std::move(mappers);
    ds.labels_.assign(labels.begin(), labels.end());
    ds.weights_.assign(weights.begin(), weights.end());
    ds.bins_are_u8_ = bins_are_u8;
    ds.cols_->u8    = std::move(u8);
    ds.cols_->u16   = std::move(u16);
    return ds;
}

Dataset Dataset::select_features(std::span<feature_id_t const> keep) const
{
    if (keep.empty())
    {
        throw std::invalid_argument("Dataset::select_features: no features kept; a "
                                    "dataset with no columns has nothing to split on");
    }
    std::vector<BinMapper>   kept;
    std::vector<std::string> names;
    kept.reserve(keep.size());
    names.reserve(keep.size());
    auto const parent_names = mappers_.feature_names();
    // Names are optional on a core Dataset (the binding always supplies
    // f0..fN, a direct BinMappers::fit need not), and an unnamed parent has
    // an EMPTY span rather than blanks. Carrying that through keeps the child
    // in the same state as its parent instead of reading past the end.
    bool const     named = parent_names.size() == n_features_;
    RowIndex const rows{rows_};
    for (feature_id_t const f : keep)
    {
        if (f >= n_features_)
        {
            throw std::invalid_argument("Dataset::select_features: feature " +
                                        std::to_string(f) +
                                        " is past the last of this dataset's " +
                                        std::to_string(n_features_) + " features");
        }
        kept.push_back(mappers_[f]);
        if (named)
        {
            names.emplace_back(parent_names[f]);
        }
    }
    // Labels and weights follow the rows the columns were gathered through,
    // so the result is an ordinary dataset whose row ids number from zero.
    std::vector<float> const lab = gather_rows(rows_, labels_);
    std::vector<float> const w =
        weights_.empty() ? std::vector<float>{} : gather_rows(rows_, weights_);
    BinMappers kept_mappers =
        BinMappers::from_mappers(std::move(kept), std::move(names));

    // The backend's own gather first: a device-resident dataset rewrites its
    // columns without the plane ever coming home, which is the difference
    // between a feature-selection loop that stays on the card and one that
    // pays a round trip per round. A backend without one says so and the host
    // path below runs unchanged.
    if (plane_)
    {
        std::vector<row_id_t> const ids =
            rows_.is_identity() ? std::vector<row_id_t>{} : rows_.materialize();
        if (auto sub = plane_->select_columns(keep, ids))
        {
            return bin(rows.size(), keep.size(), floats_view{lab}, kept_mappers,
                       DataConfig{}, std::move(sub), floats_view{w});
        }
    }

    std::vector<std::vector<uint8_t>>  u8(bins_are_u8_ ? keep.size() : 0);
    std::vector<std::vector<uint16_t>> u16(bins_are_u8_ ? 0 : keep.size());
    for (size_t k = 0; k < keep.size(); ++k)
    {
        // Reads through the plane's lazy host materialization when this
        // dataset is device-resident and the backend declined above.
        visit_bins(keep[k],
                   [&](auto col)
                   {
                       using T =
                           std::remove_const_t<typename decltype(col)::element_type>;
                       auto &dst = [&]() -> auto &
                       {
                           if constexpr (std::is_same_v<T, uint8_t>)
                           {
                               return u8[k];
                           }
                           else
                           {
                               return u16[k];
                           }
                       }();
                       dst.resize(rows.size());
                       for (size_t i = 0; i < rows.size(); ++i)
                       {
                           dst[i] = col[rows[i]];
                       }
                   });
    }
    return from_bins(std::move(u8), std::move(u16), bins_are_u8_,
                     std::move(kept_mappers), lab, w);
}

Dataset Dataset::materialize() const
{
    std::vector<feature_id_t> all(n_features_);
    std::iota(all.begin(), all.end(), feature_id_t{0});
    return select_features(all);
}

Dataset Dataset::with_rows(RowView rows) const
{
    if (rows.parent_rows() != n_rows_)
    {
        throw std::invalid_argument("Dataset::with_rows: the row view describes " +
                                    std::to_string(rows.parent_rows()) +
                                    " rows and this dataset holds " +
                                    std::to_string(n_rows_));
    }
    // The fill reads a run as a subspan of a column, so a run past the end is
    // refused here rather than left to land somewhere inside the allocation.
    if (!rows.fits(n_rows_))
    {
        throw std::invalid_argument(
            "Dataset::with_rows: the row view names a row past the last of this "
            "dataset's " +
            std::to_string(n_rows_));
    }
    Dataset view = *this;
    view.rows_   = std::move(rows);
    return view;
}

size_t Dataset::n_rows() const
{
    return n_rows_;
}

size_t Dataset::n_features() const
{
    return n_features_;
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

void Dataset::mint_row_major() const
{
    auto const  &cols  = plane_ ? host_bins().u8 : cols_->u8;
    size_t const f     = cols.size();
    size_t const width = mirror_tile_width();
    row_major_->bins.resize(n_rows_ * f);
    uint8_t *out = row_major_->bins.data();
    // Tiled column-to-row transpose into the block layout: each worker
    // owns a row block, so writes never overlap and the mirror is
    // byte-identical at any thread count. Feature c lands in mirror
    // block c/width at column position c%width; one block reproduces
    // the classic layout exactly.
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
                                         size_t const mb = c / width;
                                         size_t const width_b =
                                             std::min(width, f - (mb * width));
                                         size_t const   base = n_rows_ * mb * width;
                                         size_t const   in_b = c - (mb * width);
                                         uint8_t       *dst  = out + base + in_b;
                                         uint8_t const *col  = cols[c].data();
                                         for (size_t r = r0; r < r1; ++r)
                                         {
                                             dst[r * width_b] = col[r];
                                         }
                                     }
                                 }
                             });
}

std::span<uint8_t const> Dataset::row_major_bins() const
{
    if (!bins_are_u8_)
    {
        return {};
    }
    std::call_once(row_major_->once, [this] { mint_row_major(); });
    return row_major_->bins;
}

} // namespace bonsai
