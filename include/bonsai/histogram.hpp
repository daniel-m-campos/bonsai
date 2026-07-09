#pragma once

#include "bonsai/detail/hist_pool.hpp"
#include "bonsai/types.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <span>
#include <vector>

namespace bonsai
{

struct HistCell
{
    double sum_grad = 0.0;
    double sum_hess = 0.0;

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
    explicit Histogram(size_t n_bins) : cells_(n_bins) {}

    void add(bin_id_t bin, double grad, double hess)
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
        HistCell t{};
        for (auto const &cell : cells_)
        {
            t.sum_grad += cell.sum_grad;
            t.sum_hess += cell.sum_hess;
        }
        return t;
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
        return std::span{cells_}.first(cells_.size() - 1);
    }

    cell_view_t all_cells() const
    {
        return std::span{cells_};
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
        return std::span{cells_}.first(cells_.size() - 2);
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
        HistCell run{};
        size_t   i = 0;
        for (auto const &cell : cut_cells())
        {
            run.sum_grad += cell.sum_grad;
            run.sum_hess += cell.sum_hess;
            out[i++] = run;
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
    std::vector<HistCell, detail::PoolAllocator<HistCell>> cells_;
};

using histogram_view_t = std::span<Histogram const>;

} // namespace bonsai
