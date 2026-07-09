#pragma once

#include <cstddef>
#include <cstdint>
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

    floats_view       labels() const;
    floats_view       weights() const; // empty if uniform
    BinMappers const &mappers() const;
    size_t            n_bins(size_t fid) const;
    bool              is_categorical(size_t fid) const;

    // Binned columns store 8-bit when every feature fits 256 bins (the
    // max_bin=255 default) — halving the memory traffic of the histogram
    // fill, the dominant fit stage — and 16-bit otherwise. Readers dispatch
    // once per column via visit_bins; the callable is monomorphized per
    // width, so the per-row loop never branches.
    bool bins_are_u8() const
    {
        return bins_are_u8_;
    }

    template <typename F> decltype(auto) visit_bins(size_t fid, F &&f) const
    {
        if (bins_are_u8_)
        {
            return f(std::span<uint8_t const>{features_u8_[fid]});
        }
        return f(std::span<uint16_t const>{features_u16_[fid]});
    }

    // Single-element read for tree-routing loops (feature varies per step, so
    // a per-column visitor buys nothing there); the branch predicts perfectly.
    bin_id_t bin_at(size_t fid, size_t row) const
    {
        return bins_are_u8_ ? features_u8_[fid][row] : features_u16_[fid][row];
    }

  private:
    std::vector<std::vector<uint8_t>>  features_u8_;
    std::vector<std::vector<uint16_t>> features_u16_;
    bool                               bins_are_u8_ = false;
    std::vector<float>                 labels_;
    std::vector<float>                 weights_;
    BinMappers                         mappers_;
    std::vector<bool>                  is_categorical_;
    size_t                             n_rows_ = 0;
};

} // namespace bonsai
