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
#include <variant>
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

// perf: workers own row tiles and visit every feature within the tile, so a
// row-major source is pulled into cache once per tile instead of once per
// feature (a straight column pass reads X[r, f] at n_features x 4B stride,
// ~25x line amplification at 100 features). 64 u8 rows = one cache line per
// column, so tiles never share one.
template <typename BinT, typename Read>
void fill_columns(std::vector<std::vector<BinT>> &out, size_t n_features, size_t n_rows,
                  BinMappers const &mappers, Read read)
{
    out.resize(n_features);
    parallel::for_each_index(n_features, [&](size_t f) { out[f].resize(n_rows); });
    constexpr size_t tile = 64;
    parallel::for_each_index((n_rows + tile - 1) / tile,
                             [&](size_t block)
                             {
                                 size_t const r0 = block * tile;
                                 size_t const r1 = std::min(r0 + tile, n_rows);
                                 for (size_t f = 0; f < n_features; ++f)
                                 {
                                     auto const &mapper = mappers[f];
                                     BinT *const col    = out[f].data();
                                     for (size_t r = r0; r < r1; ++r)
                                     {
                                         col[r] = static_cast<BinT>(
                                             mapper.transform(read(f, r)));
                                     }
                                 }
                             });
}

template <typename Read>
BinColumns bin_columns(BinMappers const &mappers, size_t n_features, size_t n_rows,
                       Read read)
{
    BinColumns cols =
        mappers.all_fit_u8() ? BinColumns{U8Columns{}} : BinColumns{U16Columns{}};
    std::visit([&](auto &out) { fill_columns(out, n_features, n_rows, mappers, read); },
               cols);
    return cols;
}

} // namespace

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
        store = std::make_shared<BinStore const>(BinStore::Key{}, n, mappers,
                                                 std::move(plane));
    }
    else
    {
        store = std::make_shared<BinStore const>(
            BinStore::Key{}, n, mappers,
            bin_columns(mappers, batch.features.size(), n,
                        [&](size_t f, size_t r) { return batch.features[f][r]; }));
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
    Dataset                                       ds;
    ds.rows_  = RowView::all(n);
    ds.store_ = std::make_shared<BinStore const>(BinStore::Key{}, n, mappers,
                                                 bin_columns(mappers, X.extent(1), n,
                                                             [&](size_t f, size_t r)
                                                             { return X[r, f]; }));
    ds.id_    = mint_fit_id();
    ds.meta_  = std::make_shared<Meta const>(
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
    ds.rows_  = RowView::all(n_rows);
    ds.store_ = std::make_shared<BinStore const>(BinStore::Key{}, n_rows, mappers,
                                                 std::move(plane));
    ds.id_    = mint_fit_id();
    ds.meta_  = std::make_shared<Meta const>(
        Meta{.labels  = std::vector<float>(labels.begin(), labels.end()),
              .weights = std::vector<float>(weights.begin(), weights.end()),
              .id      = mint_labels_id()});
    return ds;
}

Dataset Dataset::from_bins(BinColumns cols, BinMappers mappers, floats_view labels,
                           floats_view weights)
{
    size_t const n_features = std::visit([](auto const &c) { return c.size(); }, cols);
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
    std::visit(
        [&](auto const &c)
        {
            for (size_t f = 0; f < n_features; ++f)
            {
                if (c[f].size() != n_rows)
                {
                    throw std::invalid_argument(
                        "Dataset::from_bins: column " + std::to_string(f) + " holds " +
                        std::to_string(c[f].size()) + " bins and this dataset holds " +
                        std::to_string(n_rows) + " rows");
                }
            }
        },
        cols);
    Dataset ds;
    ds.rows_  = RowView::all(n_rows);
    ds.store_ = std::make_shared<BinStore const>(BinStore::Key{}, n_rows,
                                                 std::move(mappers), std::move(cols));
    ds.id_    = mint_fit_id();
    ds.meta_  = std::make_shared<Meta const>(
        Meta{.labels  = std::vector<float>(labels.begin(), labels.end()),
              .weights = std::vector<float>(weights.begin(), weights.end()),
              .id      = mint_labels_id()});
    return ds;
}

Dataset Dataset::select_features(std::span<feature_id_t const> keep) const
{
    std::vector<row_id_t> const ids =
        rows_.is_identity() ? std::vector<row_id_t>{} : rows_.materialize();
    Dataset ds;
    ds.store_ = store_->select_columns(keep, ids);
    ds.rows_  = RowView::all(ds.store_->n_rows());
    ds.id_    = mint_fit_id();
    ds.meta_  = std::make_shared<Meta const>(
        Meta{.labels  = rows_.gather(meta_->labels),
              .weights = meta_->weights.empty() ? std::vector<float>{}
                                                : rows_.gather(meta_->weights),
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
    if (!rows.can_fit(store_->n_rows()))
    {
        throw std::invalid_argument(
            "Dataset::with_rows: the row view names a row past the last of this "
            "dataset's " +
            std::to_string(store_->n_rows()));
    }
    Dataset view = *this;
    view.rows_   = std::move(rows);
    view.id_     = mint_fit_id();
    return view;
}

size_t Dataset::plane_n_rows() const
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
