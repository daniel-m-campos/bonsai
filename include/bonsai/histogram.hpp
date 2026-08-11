#pragma once

#include "bonsai/detail/hist_pool.hpp"
#include "bonsai/parallel.hpp"
#include "bonsai/types.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <mdspan>
#include <new>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

namespace bonsai
{

// float cells: gradients/hessians arrive as float, per-cell sums are bounded
// by node size, and halving the cell halves the fill's memory traffic — the
// dominant fit stage (decision 50). Reductions ACROSS cells (totals, prefix
// scans, split running sums) accumulate in double and convert once at the
// store, so only per-cell storage carries float rounding.
struct HistCell
{
    float sum_grad = 0.0F;
    float sum_hess = 0.0F;

    HistCell &operator-=(HistCell const &other)
    {
        sum_grad -= other.sum_grad;
        sum_hess -= other.sum_hess;
        return *this;
    }
};

using cell_view_t = std::span<HistCell const>;

class Histogram
{
  public:
    // An unselected feature's slot: an empty view the split finders skip.
    Histogram() = default;

    explicit Histogram(size_t n_bins) : storage_(n_bins), cells_(storage_) {}

    // Non-owning view over cells a NodeHistograms arena owns.
    explicit Histogram(std::span<HistCell> view) : cells_(view) {}

    // Move-only: a copy would mean two things (deep for an owner, aliasing
    // for a view) and nothing copies a Histogram. Moves keep the span valid
    // (vector moves preserve the heap buffer) and empty the source's view,
    // so a moved-from owner reports size() == 0.
    Histogram(Histogram &&o) noexcept
        : storage_(std::move(o.storage_)), cells_(std::exchange(o.cells_, {}))
    {
    }
    Histogram &operator=(Histogram &&o) noexcept
    {
        storage_ = std::move(o.storage_);
        cells_   = std::exchange(o.cells_, {});
        return *this;
    }

    void add(bin_id_t bin, float grad, float hess)
    {
        cells_[bin].sum_grad += grad;
        cells_[bin].sum_hess += hess;
    }

    void clear()
    {
        std::ranges::fill(cells_, HistCell{});
    }

    size_t size() const
    {
        return cells_.size();
    }

    // Sum over all cells (missing included). Node-level totals are the
    // same for every feature, so callers compute this once per node
    // instead of the histogram carrying running totals in add().
    HistCell totals() const
    {
        double grad = 0.0;
        double hess = 0.0;
        for (auto const &cell : cells_)
        {
            grad += cell.sum_grad;
            hess += cell.sum_hess;
        }
        return {.sum_grad = static_cast<float>(grad),
                .sum_hess = static_cast<float>(hess)};
    }

    HistCell const &operator[](bin_id_t bin) const
    {
        return cells_[bin];
    }

    HistCell const &missing() const
    {
        return cells_.back();
    }

    cell_view_t sweep_cells() const
    {
        return cells_.first(cells_.size() - 1);
    }

    cell_view_t all_cells() const
    {
        return cells_;
    }

    // Mutable view for the row-wise fill, whose direct arm accumulates
    // straight into a node's own cells.
    std::span<HistCell> cells()
    {
        return cells_;
    }

    // Cut positions for binary splits: real bins minus the last one,
    // since "all real bins on the left, none on the right" is degenerate.
    cell_view_t cut_cells() const
    {
        return cells_.first(cells_.size() - 2);
    }

    // Buffer size required by fill_prefix; equals cut_cells().size().
    // Returns 0 when cells_.size() < 2 (degenerate hist with no cut
    // positions).
    size_t prefix_size() const
    {
        return cells_.size() >= 2 ? cells_.size() - 2 : 0;
    }

    // Fills `out[i] = sum of cut_cells()[0..i] inclusive` for
    // i in [0, prefix_size()). Used by the level finder to get
    // random-access left-side sums at every candidate bin.
    void fill_prefix(std::span<HistCell> out) const
    {
        assert(out.size() == prefix_size());
        if (out.empty())
        {
            return; // degenerate hist (cells_.size() < 2): no cuts to scan
        }
        double grad = 0.0;
        double hess = 0.0;
        size_t i    = 0;
        for (auto const &cell : cut_cells())
        {
            grad     = grad + cell.sum_grad;
            hess     = hess + cell.sum_hess;
            out[i++] = {.sum_grad = static_cast<float>(grad),
                        .sum_hess = static_cast<float>(hess)};
        }
    }

    // Accumulates one partial (the row-wise fill's per-thread scratch).
    // Callers merge partials in a fixed order, so sums depend on the thread
    // count but not on scheduling.
    void add_cells(cell_view_t src)
    {
        assert(src.size() == size());
        for (size_t i = 0; i < cells_.size(); ++i)
        {
            cells_[i].sum_grad += src[i].sum_grad;
            cells_[i].sum_hess += src[i].sum_hess;
        }
    }

    Histogram &operator-=(Histogram const &other)
    {
        assert(other.size() == size());
        for (size_t i = 0; i < cells_.size(); ++i)
        {
            cells_[i] -= other.cells_[i];
        }
        return *this;
    }

  private:
    std::vector<HistCell, detail::PoolAllocator<HistCell>> storage_;
    std::span<HistCell>                                    cells_;
};

using histogram_view_t = std::span<Histogram const>;

// One arena per node backs the selected histograms as contiguous slices; u8
// datasets pad every feature to this many cells so the dense row-wise fill can
// address any cell as arena[feature, bin] with no per-feature base load.
constexpr size_t k_feature_stride = 256;

// The padded arena as the grid it already is: one row per selected feature,
// the feature stride a static extent so the address folds to a shift.
using ArenaView =
    std::mdspan<HistCell, std::extents<size_t, std::dynamic_extent, k_feature_stride>>;

// The packing rule for a node's arena, built once from the selected features'
// bin counts: u8 data pads every feature to a k_feature_stride row, wider bins
// pack to their exact widths. It views the bin counts, so they outlive it.
class ArenaLayout
{
  public:
    ArenaLayout(std::span<size_t const> bin_counts, bool u8)
        : bins_(bin_counts), u8_(u8)
    {
        if (u8_)
        {
            // One feature stride holds the feature's whole histogram; a wider
            // one would overlap the next feature's view of the arena.
            assert(std::ranges::all_of(bins_,
                                       [](size_t n) { return n <= k_feature_stride; }));
            total_ = bins_.size() * k_feature_stride;
            return;
        }
        starts_.reserve(bins_.size());
        for (size_t const n : bins_)
        {
            starts_.push_back(total_);
            total_ += n;
        }
    }

    size_t total_cells() const
    {
        return total_;
    }

    // The j-th selected feature's cells within an arena of total_cells().
    std::span<HistCell> slice(std::span<HistCell> arena, size_t j) const
    {
        if (!u8_)
        {
            return arena.subspan(starts_[j], bins_[j]);
        }
        // A row extracts by hand: std::submdspan is C++26 and libc++ has not
        // shipped it.
        ArenaView const grid{arena.data(), bins_.size()};
        return {&grid[j, 0], bins_[j]};
    }

  private:
    std::span<size_t const> bins_;
    std::vector<size_t>     starts_; // packed starts; empty when padded
    size_t                  total_ = 0;
    bool                    u8_    = false;
};

// One node's per-feature histograms and the single arena their cells live
// in. The two are one type because they are one lifetime: the histograms
// view the arena, and both move together whenever a node's histograms do
// (docs/architecture/2-histogram.md).
class NodeHistograms
{
  public:
    // Sizes and zeroes this node's arena, then hands every feature its slot:
    // a view into the arena for the selected ones (in `selected` order, which
    // is ascending), an empty view for the rest. `alone` spreads the zero fill
    // across workers: a caller carving a whole level already runs one node per
    // worker, but a lone node has to parallelize inside or the fill is serial.
    void carve(ArenaLayout const &layout, std::span<feature_id_t const> selected,
               size_t n_features, bool alone = false)
    {
        std::span<HistCell> const arena = reset_arena(layout.total_cells(), alone);
        auto const                slot  = [&](feature_id_t fid)
        {
            auto const it = std::ranges::lower_bound(selected, fid);
            return it != selected.end() && *it == fid
                       ? Histogram{layout.slice(
                             arena, static_cast<size_t>(it - selected.begin()))}
                       : Histogram{};
        };
        hists_ =
            std::views::iota(feature_id_t{0}, static_cast<feature_id_t>(n_features)) |
            std::views::transform(slot) | std::ranges::to<std::vector>();
    }

    void push_back(Histogram h)
    {
        hists_.push_back(std::move(h));
    }

    Histogram &operator[](size_t fid)
    {
        return hists_[fid];
    }

    Histogram const &operator[](size_t fid) const
    {
        return hists_[fid];
    }

    size_t size() const
    {
        return hists_.size();
    }

    bool empty() const
    {
        return hists_.empty();
    }

    auto begin() const
    {
        return hists_.begin();
    }

    auto end() const
    {
        return hists_.end();
    }

  private:
    std::span<HistCell> reset_arena(size_t n_cells, bool alone)
    {
        // The allocator leaves the resize untouched, so this is where every
        // cell's lifetime starts. Chunked so one worker takes a run of cells;
        // a single chunk runs serially, which is the level path's case.
        arena_.clear();
        arena_.resize(n_cells);
        HistCell *const cells = arena_.data();
        size_t const chunk = alone ? (size_t{1} << 14U) : std::max<size_t>(n_cells, 1);
        parallel::for_each_index((n_cells + chunk - 1) / chunk,
                                 [cells, n_cells, chunk](size_t c)
                                 {
                                     size_t const hi =
                                         std::min((c + 1) * chunk, n_cells);
                                     for (size_t i = c * chunk; i < hi; ++i)
                                     {
                                         ::new (cells + i) HistCell{};
                                     }
                                 });
        return arena_;
    }

    std::vector<Histogram> hists_;
    // The histograms alias this buffer, so it never reallocates while they
    // live: only carve() sizes it, and moves carry both together.
    std::vector<HistCell, detail::RawPoolAllocator<HistCell>> arena_;
};

} // namespace bonsai
