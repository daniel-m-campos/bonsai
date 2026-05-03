#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include "bonsai/bin_mapper.hpp"
#include "bonsai/config/config.hpp"
#include "bonsai/detail/column_batch.hpp"

namespace bonsai
{

class BinMappers
{
  public:
    static BinMappers fit(detail::ColumnBatch const &batch, Config const &cfg);

    BinMapper const &operator[](size_t fid) const;
    size_t size() const;
    std::span<std::string const> feature_names() const;

  private:
    std::vector<BinMapper> mappers_;
    std::vector<std::string> feature_names_;
};

} // namespace bonsai
