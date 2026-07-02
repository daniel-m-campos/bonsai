#include "bonsai/bin_mappers.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "bonsai/bin_mapper.hpp"
#include "bonsai/config/bin_mapper_config.hpp"
#include "bonsai/detail/column_batch.hpp"
#include "bonsai/parallel.hpp"

namespace bonsai
{

BinMappers BinMappers::fit(detail::ColumnBatch const &batch, BinMapperConfig const &cfg)
{
    // Feature-parallel; each fit draws its own seeded rng, so results are
    // identical to a serial pass. Optional slots because BinMapper has no
    // default constructor.
    std::vector<std::optional<BinMapper>> slots(batch.features.size());
    parallel::for_each_index(batch.features.size(),
                             [&](size_t f)
                             { slots[f] = BinMapper::fit(batch.features[f], cfg); });

    BinMappers out;
    out.mappers_.reserve(slots.size());
    for (auto &s : slots)
    {
        out.mappers_.push_back(std::move(*s));
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
