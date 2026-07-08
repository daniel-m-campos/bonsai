#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "bonsai/bin_mappers.hpp"
#include "bonsai/config/data_config.hpp"
#include "bonsai/detail/column_batch.hpp"
#include "bonsai/types.hpp"

namespace bonsai
{

class Dataset
{
  public:
    static Dataset bin(detail::ColumnBatch const &batch, BinMappers const &mappers,
                       DataConfig const &cfg);
    // Row-major matrix path: transforms strided columns directly, no
    // column-major float materialization. Bin ids identical to the
    // ColumnBatch overload.
    static Dataset bin(features_view X, floats_view labels, BinMappers const &mappers,
                       DataConfig const &cfg);

    size_t n_rows() const;
    size_t n_features() const;

    floats_view               labels() const;
    floats_view               weights() const; // empty if uniform
    BinMappers const         &mappers() const;
    size_t                    n_bins(size_t fid) const;
    bool                      is_categorical(size_t fid) const;
    std::span<bin_id_t const> feature_bins(size_t fid) const;

  private:
    std::vector<std::vector<bin_id_t>> features_;
    std::vector<float>                 labels_;
    std::vector<float>                 weights_;
    BinMappers                         mappers_;
    std::vector<bool>                  is_categorical_;
    size_t                             n_rows_ = 0;
};

} // namespace bonsai
