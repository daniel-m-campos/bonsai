#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "bonsai/config/bin_mapper_config.hpp"

namespace bonsai
{

class BinMapper
{
  public:
    static BinMapper fit(std::span<float const> column, BinMapperConfig const &cfg);

    uint16_t transform(float x) const;
    size_t n_buckets() const
    {
        return cuts_.size();
    }

    std::span<float const> cuts() const
    {
        return {cuts_};
    }

    bool has_missing_bin() const
    {
        return has_missing_bin_;
    }

  private:
    BinMapper(std::vector<float> cuts, bool has_missing_bin)
        : cuts_{std::move(cuts)}, has_missing_bin_{has_missing_bin}
    {
    }

    std::vector<float> cuts_;
    bool has_missing_bin_ = false;
};

} // namespace bonsai
