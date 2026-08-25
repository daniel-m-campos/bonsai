#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "bonsai/bin_mapper.hpp"
#include "bonsai/config/bin_mapper_config.hpp"
#include "bonsai/detail/column_batch.hpp"
#include "bonsai/types.hpp"

namespace bonsai
{

// Explicit interior cut points for select columns (doc 18): column index to
// strictly increasing finite edges. Listed columns skip sampling and fitting
// entirely (BinMapper::from_edges); unlisted columns fit as usual, so the
// default {} is bit-identical to no overrides.
using BinEdges = std::vector<std::pair<size_t, std::vector<float>>>;

// The shared row sample every fit overload draws before cutting, exposed so a
// caller whose matrix is not host-addressable (device-resident input) can
// gather exactly these rows and fit bit-identical cuts. Empty means "use every
// row"; the indices are ascending.
std::vector<uint32_t> bin_sample_rows(size_t n_rows, BinMapperConfig const &cfg);

class BinMappers
{
  public:
    static BinMappers fit(detail::ColumnBatch const &batch, BinMapperConfig const &cfg,
                          BinEdges const &bin_edges = {});
    // Row-major matrix path: each worker gathers one column into scratch and
    // fits it, so cuts are bit-identical to the ColumnBatch overload.
    static BinMappers fit(features_view X, std::vector<std::string> feature_names,
                          BinMapperConfig const &cfg, BinEdges const &bin_edges = {});
    static BinMappers from_mappers(std::vector<BinMapper>   mappers,
                                   std::vector<std::string> feature_names)
    {
        BinMappers out;
        out.mappers_       = std::move(mappers);
        out.feature_names_ = std::move(feature_names);
        return out;
    }

    // Whether every feature's bins fit a byte, which is what decides the
    // stored bin width everywhere a store is built.
    bool all_fit_u8() const
    {
        return std::ranges::all_of(mappers_, [](BinMapper const &m)
                                   { return m.n_bins() <= 256; });
    }

    BinMapper const             &operator[](size_t fid) const;
    size_t                       size() const;
    std::span<std::string const> feature_names() const;

    // Whether both sets cut every feature at the same points. Binning a
    // second matrix for a booster's use is only meaningful under the cuts
    // that booster's thresholds came from, so this is the precondition a
    // caller pairing two datasets owes.
    bool same_cuts(BinMappers const &other) const;

  private:
    std::vector<BinMapper>   mappers_;
    std::vector<std::string> feature_names_;
};

} // namespace bonsai
