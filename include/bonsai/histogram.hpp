#pragma once

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

    // Cut positions for binary splits: real bins minus the last one,
    // since "all real bins on the left, none on the right" is degenerate.
    cell_view_t cut_cells() const
    {
        return std::span{cells_}.first(cells_.size() - 2);
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
    std::vector<HistCell> cells_;
};

using histogram_view_t = std::span<Histogram const>;

} // namespace bonsai
