#pragma once

#include "bonsai/detail/hist_pool.hpp"
#include "bonsai/types.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
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

    // Mutable view for the row-wise fill, which accumulates straight into
    // single-block nodes' cells.
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

    // Convenience: allocates a buffer sized to prefix_size() and
    // fills it. For tests / one-off code; hot-path callers should
    // reuse a thread_local buffer with resize() + fill_prefix().
    std::vector<HistCell> create_prefix() const
    {
        std::vector<HistCell> out(prefix_size());
        fill_prefix(out);
        return out;
    }

    // Accumulates a partial-histogram block (the row-wise fill's per-thread
    // scratch). Callers merge partials in a fixed order, so sums depend on
    // the thread count but not on scheduling.
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

// One node's per-feature histograms and the single arena their cells live
// in. The two are one type because they are one lifetime: the histograms
// view the arena, and both move together whenever a node's histograms do
// (docs/architecture/2-histogram.md).
class NodeHistograms
{
  public:
    // Sizes the arena for one node and zeroes it; the views built next point
    // into the returned span, which never reallocates while they exist.
    std::span<HistCell> reset_arena(size_t n_cells)
    {
        arena_.assign(n_cells, HistCell{});
        return arena_;
    }

    void reserve(size_t n)
    {
        hists_.reserve(n);
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
    std::vector<Histogram>                                 hists_;
    std::vector<HistCell, detail::PoolAllocator<HistCell>> arena_;
};

} // namespace bonsai
