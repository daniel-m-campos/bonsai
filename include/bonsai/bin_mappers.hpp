#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "bonsai/bin_mapper.hpp"
#include "bonsai/config/bin_mapper_config.hpp"
#include "bonsai/detail/column_batch.hpp"

namespace bonsai
{

class BinMappers
{
  public:
    static BinMappers fit(detail::ColumnBatch const &batch, BinMapperConfig const &cfg);
    static BinMappers from_mappers(std::vector<BinMapper> mappers,
                                   std::vector<std::string> feature_names)
    {
        BinMappers out;
        out.mappers_       = std::move(mappers);
        out.feature_names_ = std::move(feature_names);
        return out;
    }

    BinMapper const &operator[](size_t fid) const;
    size_t size() const;
    std::span<std::string const> feature_names() const;

  private:
    std::vector<BinMapper> mappers_;
    std::vector<std::string> feature_names_;
};

} // namespace bonsai
