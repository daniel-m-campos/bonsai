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
    // Cuts from an already-gathered, NaN-free working set. The row-sample-once
    // path (BinMappers::fit) draws one shared row sample and gathers each
    // feature's values at those rows, so the O(n) reservoir pass runs once for
    // the whole matrix instead of once per feature.
    static BinMapper from_sample(std::vector<float> sample, BinMapperConfig const &cfg);
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
