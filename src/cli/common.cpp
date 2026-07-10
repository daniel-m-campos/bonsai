#include "bonsai/cli/common.hpp"

#include <cstddef>

#include "bonsai/detail/column_batch.hpp"
#include "bonsai/detail/perf.hpp"
#include "bonsai/parallel.hpp"

namespace bonsai::cli
{

FeatureBuffer to_feature_buffer(detail::ColumnBatch const &batch)
{
    detail::IngestProfiler::Lap lap;
    FeatureBuffer               buf;
    buf.n_features = batch.features.size();
    buf.n_rows     = buf.n_features == 0 ? 0 : batch.features[0].size();
    buf.data.resize(buf.n_rows * buf.n_features);
    parallel::for_each_index(buf.n_rows,
                             [&](size_t r)
                             {
                                 for (size_t f = 0; f < buf.n_features; ++f)
                                 {
                                     buf.data[(r * buf.n_features) + f] =
                                         batch.features[f][r];
                                 }
                             });
    lap(detail::IngestProfiler::instance().buffer_s);
    return buf;
}

} // namespace bonsai::cli
