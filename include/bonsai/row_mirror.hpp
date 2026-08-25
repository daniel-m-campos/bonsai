#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>
#include <vector>

namespace bonsai
{

// The row-major mirror of a binned plane: the bins laid out row-major in
// column blocks, plus the one rule for addressing them.
//
// Layout is column-block-tiled (issue #217): features are grouped into blocks
// of tile_width and each block is row-major on its own (block b starts at
// n_rows * b * tile_width; a row's bins inside it are width_b bytes). At
// n_features <= tile_width there is exactly one block and the layout is the
// classic n_rows x n_features mirror.
//
// Its own type because the layout is a pure function of the shape and one
// cache-sized constant. It changes when cache sizing changes, and for no
// reason a Dataset ever changes; keeping it here means the addressing rule
// has one home and can be exercised without binning anything.
class RowMirror
{
  public:
    // Features per block: sized so one block's full histogram footprint
    // (tile_width x 256 bins x 8B cells) stays cache-resident during a tiled
    // row-wise fill.
    static constexpr size_t tile_width = 2048;

    RowMirror() = default;
    RowMirror(size_t n_rows, size_t n_features)
        : n_rows_(n_rows), n_features_(n_features)
    {
    }

    // Where (row, fid) sits in bins(). Tree-routing loops call this per
    // element, so it stays a header-inline expression; one block falls out of
    // it unchanged.
    size_t index(size_t row, size_t fid) const
    {
        size_t const block = fid / tile_width;
        size_t const wide  = std::min(tile_width, n_features_ - (block * tile_width));
        return (n_rows_ * block * tile_width) + (row * wide) +
               (fid - (block * tile_width));
    }

    std::span<uint8_t const> bins() const
    {
        return bins_;
    }

    // Minted at most once across every Dataset sharing this mirror. The
    // caller owns knowing where bins come from; the mirror owns the layout,
    // the storage and the once. First use can be a parallel loop, so the
    // synchronization is not optional.
    //
    // The buffer goes out as a span, not a vector reference: this sizes it
    // from the shape it was constructed with while the filler bounds its
    // writes by the columns it reads, and those are two different facts. A
    // span carries the length to the place that would otherwise overrun it.
    template <typename Fill> void mint_once(Fill &&fill) const
    {
        std::call_once(once_,
                       [&]
                       {
                           bins_.resize(n_rows_ * n_features_);
                           fill(std::span<uint8_t>{bins_});
                       });
    }

  private:
    mutable std::vector<uint8_t> bins_;
    mutable std::once_flag       once_;
    size_t                       n_rows_     = 0;
    size_t                       n_features_ = 0;
};

} // namespace bonsai
