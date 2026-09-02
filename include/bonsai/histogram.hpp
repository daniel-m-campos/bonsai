#pragma once

#include "bonsai/detail/hist_pool.hpp"
#include "bonsai/parallel.hpp"
#include "bonsai/types.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <mdspan>
#include <memory>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

namespace bonsai
{

// float cells: gradients/hessians arrive as float, per-cell sums are bounded
// by node size, and halving the cell halves the fill's memory traffic in
// the dominant fit stage. Reductions ACROSS cells (totals, prefix
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

// One node's per-feature cells, viewing memory it never owns: histograms are
// carved from a level's arena (NodeHistograms::carve) or leased from the
// block pool, and a Histogram must not outlive its arena. Cells store float
// sums; every reduction over them runs in double, and node totals are
// computed at split time rather than accumulated in the fill, which is what
// keeps the subtraction trick exact (invariants: subtraction-trick). The
// last cell is the missing bin: honest data, outside the split sweep, which
// missing(), sweep_cells() and cut_cells() all address by position.
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
            return;
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

    // The arena run the j-th selected feature owns, u8 padding included. The
    // runs tile the arena exactly, so a carve that walks them touches every
    // cell. Addressing only, so a const arena addresses the same way.
    template <typename C> std::span<C> run(std::span<C> arena, size_t j) const
    {
        size_t const start = u8_ ? j * k_feature_stride : starts_[j];
        size_t const width = u8_ ? k_feature_stride : bins_[j];
        return arena.subspan(start, width);
    }

    // The j-th selected feature's cells within an arena of total_cells().
    template <typename C> std::span<C> slice(std::span<C> arena, size_t j) const
    {
        return run(arena, j).first(bins_[j]);
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
// together.
class NodeHistograms
{
  public:
    // Sizes and zeroes this node's arena, then hands every feature its slot:
    // a view into the arena for the selected ones (in `selected` order, which
    // is ascending), an empty view for the rest. One selected feature's arena
    // run is the work unit, so the zero fill and the slot hand-out are one
    // pass; unselected features keep the default empty view and cost nothing.
    // `alone` spreads that pass across workers: a caller carving a whole level
    // already runs one node per worker, but a lone node has to parallelize
    // inside or the carve is serial.
    void carve(ArenaLayout const &layout, std::span<feature_id_t const> selected,
               size_t n_features, bool alone = false)
    {
        carve_storage(layout, n_features);
        parallel::for_each_index_on(alone ? parallel::n_threads() : 1, selected.size(),
                                    [&](size_t j) { carve_run(layout, selected, j); });
    }

    // Sizes the arena and the slot table without touching a cell: the
    // allocator leaves the resize untouched, so carve_run's placement-new is
    // where every cell's lifetime starts. Pairs with carve_run for callers
    // that carve inside their own fill.
    void carve_storage(ArenaLayout const &layout, size_t n_features)
    {
        arena_.clear();
        arena_.resize(layout.total_cells());
        hists_.clear();
        hists_.resize(n_features);
    }

    // Zeroes the j-th selected feature's arena run and hands it its slot.
    // Runs are disjoint, so any worker may take any j.
    void carve_run(ArenaLayout const &layout, std::span<feature_id_t const> selected,
                   size_t j)
    {
        std::span<HistCell> const arena{arena_.data(), arena_.size()};
        std::ranges::uninitialized_value_construct(layout.run(arena, j));
        hists_[selected[j]] = Histogram{layout.slice(arena, j)};
    }

    // Debug-only postcondition for a fill that carves inside its own work
    // list: every selected run must have been reached, or the cells the fill
    // read had no lifetime. A carved run's slot views its own arena offset,
    // which nothing but carve_run puts there.
    bool all_runs_carved(ArenaLayout const            &layout,
                         std::span<feature_id_t const> selected) const
    {
        std::span<HistCell const> const arena{arena_.data(), arena_.size()};
        for (size_t j = 0; j < selected.size(); ++j)
        {
            if (hists_[selected[j]].all_cells().data() != layout.slice(arena, j).data())
            {
                return false;
            }
        }
        return true;
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
    std::vector<Histogram> hists_;
    // The histograms alias this buffer, so it never reallocates while they
    // live: only carve() sizes it, and moves carry both together.
    std::vector<HistCell, detail::RawPoolAllocator<HistCell>> arena_;
};

} // namespace bonsai
