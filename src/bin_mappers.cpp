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
#include "bonsai/detail/perf.hpp"
#include "bonsai/parallel.hpp"

namespace bonsai
{

BinMappers BinMappers::fit(detail::ColumnBatch const &batch, BinMapperConfig const &cfg)
{
    detail::IngestProfiler::Lap lap;
    // Feature-parallel; each fit draws its own seeded rng, so results are
    // identical to a serial pass. Optional slots because BinMapper has no
    // default constructor.
    std::vector<std::optional<BinMapper>> slots(batch.features.size());
    parallel::for_each_index(batch.features.size(), [&](size_t f)
                             { slots[f] = BinMapper::fit(batch.features[f], cfg); });
    lap(detail::IngestProfiler::instance().fit_s);

    BinMappers out;
    out.mappers_.reserve(slots.size());
    for (auto &s : slots)
    {
        // Every slot was filled by the loop above.
        out.mappers_.push_back(
            std::move(*s)); // NOLINT(bugprone-unchecked-optional-access)
    }
    out.feature_names_ = batch.feature_names;
    return out;
}

BinMappers BinMappers::fit(features_view X, std::vector<std::string> feature_names,
                           BinMapperConfig const &cfg)
{
    detail::IngestProfiler::Lap lap;
    size_t const                n = X.extent(0);
    size_t const                f = X.extent(1);
    // Gathering a column into contiguous scratch feeds BinMapper::fit the
    // exact sequence the ColumnBatch overload would — identical cuts.
    std::vector<std::optional<BinMapper>> slots(f);
    parallel::for_each_index(f,
                             [&](size_t c)
                             {
                                 std::vector<float> column(n);
                                 for (size_t r = 0; r < n; ++r)
                                 {
                                     column[r] = X[r, c];
                                 }
                                 slots[c] = BinMapper::fit(column, cfg);
                             });
    lap(detail::IngestProfiler::instance().fit_s);

    std::vector<BinMapper> mappers;
    mappers.reserve(f);
    for (auto &s : slots)
    {
        // Every slot was filled by the loop above.
        mappers.push_back(std::move(*s)); // NOLINT(bugprone-unchecked-optional-access)
    }
    return from_mappers(std::move(mappers), std::move(feature_names));
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
