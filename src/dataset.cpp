#include "bonsai/dataset.hpp"

#include <algorithm>
#include <atomic>
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

// One counter serves both id spaces: the types keep them incomparable, and
// a shared counter cannot hand the same value to two different mints.
namespace
{
std::atomic<uint64_t> &identity_counter()
{
    static std::atomic<uint64_t> counter{1};
    return counter;
}
} // namespace

LabelsId Dataset::mint_labels_id()
{
    return LabelsId{identity_counter().fetch_add(1, std::memory_order_relaxed)};
}

FitId Dataset::mint_fit_id()
{
    return FitId{identity_counter().fetch_add(1, std::memory_order_relaxed)};
}

namespace
{

// Shared bin loop: `read(f, r)` yields the raw float for (row, feature);
// values are identical either width, so models stay byte-identical. Workers
// own row tiles and visit every feature within the tile, so a row-major
// source is pulled into cache once per tile instead of once per feature (a
// straight column pass reads X[r, f] at n_features x 4B stride — ~25x line
// amplification at 100 features). The tile size only reorders independent
// writes; 64 u8 rows = one cache line per column, so tiles never share one.
template <typename Read>
void fill_binned(std::vector<std::vector<uint8_t>>  &u8,
                 std::vector<std::vector<uint16_t>> &u16, bool u8_mode,
                 size_t n_features, size_t n_rows, BinMappers const &mappers, Read read)
{
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
    size_t const                                  n = batch.labels.size();
    std::shared_ptr<BinStore const>               store;
    if (plane)
    {
        store =
            std::make_shared<BinStore const>(BinStore{n, mappers, std::move(plane)});
    }
    else
    {
        BinStore::HostColumns cols;
        bool const            u8 = mappers.all_fit_u8();
        fill_binned(cols.u8, cols.u16, u8, batch.features.size(), n, mappers,
                    [&](size_t f, size_t r) { return batch.features[f][r]; });
        store =
            std::make_shared<BinStore const>(BinStore{n, mappers, std::move(cols), u8});
    }
    Dataset ds;
    ds.rows_  = RowView::all(n);
    ds.store_ = std::move(store);
    ds.id_    = mint_fit_id();
    ds.meta_  = std::make_shared<Meta const>(
        Meta{.labels = batch.labels, .weights = batch.weights, .id = mint_labels_id()});
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
    size_t const                                  n = labels.size();
    BinStore::HostColumns                         cols;
    bool const                                    u8 = mappers.all_fit_u8();
    fill_binned(cols.u8, cols.u16, u8, X.extent(1), n, mappers,
                [&](size_t f, size_t r) { return X[r, f]; });
    Dataset ds;
    ds.rows_ = RowView::all(n);
    ds.store_ =
        std::make_shared<BinStore const>(BinStore{n, mappers, std::move(cols), u8});
    ds.id_   = mint_fit_id();
    ds.meta_ = std::make_shared<Meta const>(
        Meta{.labels  = std::vector<float>(labels.begin(), labels.end()),
             .weights = std::vector<float>(weights.begin(), weights.end()),
             .id      = mint_labels_id()});
    return ds;
}

Dataset Dataset::bin(size_t n_rows, [[maybe_unused]] size_t n_features,
                     floats_view labels, BinMappers const &mappers,
                     DataConfig const & /*cfg*/,
                     std::shared_ptr<IngestPlane const> plane, floats_view weights)
{
    assert(plane != nullptr);
    assert(n_features == mappers.size());
    detail::Phase<&detail::IngestProfiler::bin_s> phase;
    Dataset                                       ds;
    ds.rows_ = RowView::all(n_rows);
    ds.store_ =
        std::make_shared<BinStore const>(BinStore{n_rows, mappers, std::move(plane)});
    ds.id_   = mint_fit_id();
    ds.meta_ = std::make_shared<Meta const>(
        Meta{.labels  = std::vector<float>(labels.begin(), labels.end()),
             .weights = std::vector<float>(weights.begin(), weights.end()),
             .id      = mint_labels_id()});
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
    // A fresh store, never the one the bins came from: adopting another
    // dataset's store would serve its caches and its mirror.
    Dataset ds;
    ds.rows_  = RowView::all(n_rows);
    ds.store_ = std::make_shared<BinStore const>(
        BinStore{n_rows, std::move(mappers),
                 BinStore::HostColumns{.u8 = std::move(u8), .u16 = std::move(u16)},
                 bins_are_u8});
    ds.id_   = mint_fit_id();
    ds.meta_ = std::make_shared<Meta const>(
        Meta{.labels  = std::vector<float>(labels.begin(), labels.end()),
             .weights = std::vector<float>(weights.begin(), weights.end()),
             .id      = mint_labels_id()});
    return ds;
}

Dataset Dataset::select_features(std::span<feature_id_t const> keep) const
{
    // The store gathers its own columns; what is Dataset's here is the fit:
    // spending the row view, gathering the labels and weights it names, and
    // minting the result's identities.
    std::vector<row_id_t> const ids =
        rows_.is_identity() ? std::vector<row_id_t>{} : rows_.materialize();
    Dataset ds;
    ds.store_ = store_->select_columns(keep, ids);
    ds.rows_  = RowView::all(ds.store_->n_rows());
    ds.id_    = mint_fit_id();
    ds.meta_  = std::make_shared<Meta const>(
        Meta{.labels  = gather_rows(rows_, meta_->labels),
              .weights = meta_->weights.empty() ? std::vector<float>{}
                                                : gather_rows(rows_, meta_->weights),
              .id      = mint_labels_id()});
    return ds;
}

Dataset Dataset::materialize() const
{
    std::vector<feature_id_t> all(store_->n_features());
    std::iota(all.begin(), all.end(), feature_id_t{0});
    return select_features(all);
}

Dataset Dataset::with_rows(RowView rows) const
{
    if (rows.parent_rows() != store_->n_rows())
    {
        throw std::invalid_argument("Dataset::with_rows: the row view describes " +
                                    std::to_string(rows.parent_rows()) +
                                    " rows and this dataset holds " +
                                    std::to_string(store_->n_rows()));
    }
    // The fill reads a run as a subspan of a column, so a run past the end is
    // refused here rather than left to land somewhere inside the allocation.
    if (!rows.fits(store_->n_rows()))
    {
        throw std::invalid_argument(
            "Dataset::with_rows: the row view names a row past the last of this "
            "dataset's " +
            std::to_string(store_->n_rows()));
    }
    Dataset view = *this;
    view.rows_   = std::move(rows);
    // A new fit: same labels, different rows. The copy above kept this
    // dataset's id, and a view must not be mistaken for its parent when the
    // resident state asks whether it is still armed for the right fit.
    view.id_ = mint_fit_id();
    return view;
}

size_t Dataset::n_rows() const
{
    return store_->n_rows();
}

size_t Dataset::n_features() const
{
    return store_->n_features();
}

floats_view Dataset::labels() const
{
    return meta_->labels;
}

floats_view Dataset::weights() const
{
    return meta_->weights;
}

BinMappers const &Dataset::mappers() const
{
    return store_->mappers();
}

size_t Dataset::n_bins(size_t fid) const
{
    return store_->n_bins(fid);
}

} // namespace bonsai
