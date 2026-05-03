#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "bonsai/config/bin_mapper_config.hpp"

namespace bonsai
{

class BinMapper
{
  public:
    static BinMapper fit(std::span<float const> column, BinMapperConfig const &cfg);

    uint16_t transform(float x) const;

    size_t n_buckets() const;
    std::span<float const> cuts() const;
    float min() const;
    float max() const;

  private:
    std::vector<float> cuts_;
    uint16_t n_buckets_   = 0;
    bool has_missing_bin_ = false;
    float min_value_      = 0.0F;
    float max_value_      = 0.0F;
};

} // namespace bonsai
