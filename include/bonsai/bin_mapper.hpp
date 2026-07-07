#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "bonsai/config/bin_mapper_config.hpp"
#include "bonsai/types.hpp"

namespace bonsai
{

class BinMapper
{
  public:
    static BinMapper fit(floats_view column, BinMapperConfig const &cfg);
    static BinMapper from_cuts(std::vector<float> cuts)
    {
        return BinMapper{std::move(cuts)};
    }
    bin_id_t transform(float x) const;
    size_t   n_bins() const
    {
        return cuts_.size();
    }
    floats_view cuts() const
    {
        return {cuts_};
    }

  private:
    explicit BinMapper(std::vector<float> cuts) : cuts_{std::move(cuts)} {}

    std::vector<float> cuts_;
};

} // namespace bonsai
