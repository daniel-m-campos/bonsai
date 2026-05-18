#include "bonsai/cli/common.hpp"

#include <cstddef>

#include "bonsai/detail/column_batch.hpp"

namespace bonsai::cli
{

FeatureBuffer to_feature_buffer(detail::ColumnBatch const &batch)
{
    FeatureBuffer buf;
    buf.n_features = batch.features.size();
    buf.n_rows     = buf.n_features == 0 ? 0 : batch.features[0].size();
    buf.data.resize(buf.n_rows * buf.n_features);
    for (size_t f = 0; f < buf.n_features; ++f)
    {
        for (size_t r = 0; r < buf.n_rows; ++r)
        {
            buf.data[(r * buf.n_features) + f] = batch.features[f][r];
        }
    }
    return buf;
}

} // namespace bonsai::cli
