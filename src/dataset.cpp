#include "bonsai/dataset.hpp"

#include <cassert>
#include <cstddef>
#include <span>

#include "bonsai/bin_mappers.hpp"
#include "bonsai/config/data_config.hpp"
#include "bonsai/detail/column_batch.hpp"
#include "bonsai/parallel.hpp"
#include "bonsai/types.hpp"
#include "ingest_profiler.hpp"

namespace bonsai
{

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
    ds.features_.resize(batch.features.size());
    ds.is_categorical_.assign(batch.features.size(), false);
    parallel::for_each_index(batch.features.size(),
                             [&](size_t f)
                             {
                                 auto const &col = batch.features[f];
                                 assert(col.size() == ds.n_rows_);
                                 auto &binned = ds.features_[f];
                                 binned.resize(ds.n_rows_);
                                 auto const &mapper = mappers[f];
                                 for (size_t i = 0; i < ds.n_rows_; ++i)
                                 {
                                     binned[i] = mapper.transform(col[i]);
                                 }
                             });
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
    ds.features_.resize(X.extent(1));
    ds.is_categorical_.assign(X.extent(1), false);
    parallel::for_each_index(X.extent(1),
                             [&](size_t f)
                             {
                                 auto &binned = ds.features_[f];
                                 binned.resize(ds.n_rows_);
                                 auto const &mapper = mappers[f];
                                 for (size_t r = 0; r < ds.n_rows_; ++r)
                                 {
                                     binned[r] = mapper.transform(X[r, f]);
                                 }
                             });
    lap(detail::IngestProfiler::instance().bin_s);
    return ds;
}

size_t Dataset::n_rows() const
{
    return n_rows_;
}

size_t Dataset::n_features() const
{
    return features_.size();
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

std::span<bin_id_t const> Dataset::feature_bins(size_t fid) const
{
    return features_[fid];
}

} // namespace bonsai
