#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
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

class Histogram
{
  public:
    explicit Histogram(size_t n_buckets) : cells_(n_buckets) {}

    void add(uint16_t bin, double grad, double hess)
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

    HistCell const &operator[](size_t bin) const
    {
        return cells_[bin];
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

} // namespace bonsai
