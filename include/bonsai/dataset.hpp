#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "bonsai/bin_mappers.hpp"
#include "bonsai/config/data_config.hpp"
#include "bonsai/detail/column_batch.hpp"

namespace bonsai
{

class Dataset
{
  public:
    static Dataset bin(detail::ColumnBatch const &batch, BinMappers const &mappers,
                       DataConfig const &cfg);

    size_t num_rows() const;
    size_t num_features() const;

    std::span<float const> labels() const;
    std::span<float const> weights() const; // empty if uniform
    BinMappers const &mappers() const;
    size_t n_buckets(size_t fid) const;
    bool is_categorical(size_t fid) const;
    std::span<uint16_t const> column(size_t fid) const;

  private:
    std::vector<std::vector<uint16_t>> features_;
    std::vector<float> labels_;
    std::vector<float> weights_;
    BinMappers mappers_;
    std::vector<bool> is_categorical_;
    size_t n_rows_ = 0;
};

} // namespace bonsai
