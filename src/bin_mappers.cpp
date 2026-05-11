#include "bonsai/bin_mappers.hpp"

#include <cstddef>
#include <span>
#include <string>

#include "bonsai/bin_mapper.hpp"
#include "bonsai/config/bin_mapper_config.hpp"
#include "bonsai/detail/column_batch.hpp"

namespace bonsai
{

BinMappers BinMappers::fit(detail::ColumnBatch const &batch, BinMapperConfig const &cfg)
{
    BinMappers out;
    out.mappers_.reserve(batch.features.size());
    for (auto const &col : batch.features)
    {
        out.mappers_.push_back(BinMapper::fit(col, cfg));
    }
    out.feature_names_ = batch.feature_names;
    return out;
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

} // namespace bonsai
