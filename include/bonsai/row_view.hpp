#pragma once

#include "bonsai/types.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace bonsai
{

// One run of consecutive row ids, [start, stop).
struct RowRun
{
    row_id_t start = 0;
    row_id_t stop  = 0;

    size_t size() const
    {
        return static_cast<size_t>(stop - start);
    }
};

// A row list described as runs of consecutive plane rows, which is what lets a
// reader take a subspan of a column instead of indexing it per row.
using row_run_view = std::span<RowRun const>;

// Rows a run list covers, for the contracts that check a run list against the
// row list it claims to describe.
inline size_t rows_in(row_run_view runs)
{
    size_t n = 0;
    for (RowRun const &run : runs)
    {
        n += run.size();
    }
    return n;
}

// Which of a Dataset's rows a fit visits, in one of three forms chosen by
// run-length encoding whatever the caller passed. The ids are GLOBAL ids into
// the Dataset's plane: a view narrows the fit's row list and changes nothing
// about the plane's geometry, so grad, hess, labels and the row-major mirror
// stay full length and globally indexed.
//
// The forms differ only in storage. Range is two integers, which is what
// makes is_identity() a comparison rather than a scan over the row list.
class RowView
{
  public:
    enum class Form : uint8_t
    {
        Range,
        Segments,
        Gather
    };

    // Every row of a dataset that holds `parent_rows` of them.
    static RowView all(size_t parent_rows)
    {
        RowView view;
        view.parent_rows_ = parent_rows;
        view.size_        = parent_rows;
        view.form_        = Form::Range;
        view.runs_.push_back({.start = 0, .stop = static_cast<row_id_t>(parent_rows)});
        view.last_ = parent_rows > 0 ? static_cast<row_id_t>(parent_rows - 1) : 0;
        return view;
    }

    // Run-length encode a row list. Order and duplicates are the caller's and
    // are preserved: a node's histogram sums its rows in list order, so the
    // order is part of the answer and sorting here would silently change it.
    // Runs pay for themselves only when there are far fewer of them than
    // rows; a scattered list falls through to Gather, which stores the list.
    static RowView encode(std::span<row_id_t const> rows, size_t parent_rows)
    {
        RowView view;
        view.parent_rows_ = parent_rows;
        view.size_        = rows.size();
        if (rows.empty())
        {
            view.form_ = Form::Gather;
            return view;
        }
        view.first_ = *std::ranges::min_element(rows);
        view.last_  = *std::ranges::max_element(rows);
        // The runs arm loses once it needs more than one run per two rows, and
        // that test only ever goes from false to true as runs accumulate. So
        // stop at the crossing rather than encoding the whole list first: the
        // input this discards for is the scattered one, which is exactly where
        // the abandoned runs vector would be largest.
        size_t const cap = (rows.size() / 2) + 1;
        RowRun       cur{.start = rows[0], .stop = static_cast<row_id_t>(rows[0] + 1)};
        bool         gather = false;
        for (row_id_t const r : rows.subspan(1))
        {
            if (r == cur.stop)
            {
                ++cur.stop;
                continue;
            }
            view.runs_.push_back(cur);
            cur = {.start = r, .stop = static_cast<row_id_t>(r + 1)};
            if (view.runs_.size() > cap)
            {
                gather = true;
                break;
            }
        }
        if (!gather)
        {
            view.runs_.push_back(cur);
        }
        if (gather || view.runs_.size() * 2 > rows.size())
        {
            view.runs_.clear();
            view.runs_.shrink_to_fit();
            view.ids_.assign(rows.begin(), rows.end());
            view.form_ = Form::Gather;
            return view;
        }
        view.form_ = view.runs_.size() == 1 ? Form::Range : Form::Segments;
        return view;
    }

    Form form() const
    {
        return form_;
    }

    size_t size() const
    {
        return size_;
    }

    size_t parent_rows() const
    {
        return parent_rows_;
    }

    // Runs for the encoded forms; a gather is one run per row by definition.
    size_t n_runs() const
    {
        return form_ == Form::Gather ? size_ : runs_.size();
    }

    // The runs themselves, in the order they are visited: the fill walks them
    // to read each one's bins as a contiguous subspan. Empty for a Gather,
    // which has no contiguity to spend and keeps the per-row indirection.
    row_run_view runs() const
    {
        return runs_;
    }

    // Whether every row id lies inside a dataset of `n` rows. The fill takes
    // subspans off the runs, so a run reaching past a column's end has to be a
    // contract violation rather than a read that lands inside the allocation.
    bool can_fit(size_t n) const
    {
        return size_ == 0 || static_cast<size_t>(last_) < n;
    }

    // The values this view names, in view order: out[k] = values[rows[k]].
    // Declared here, defined after RowIndex, which does the indexing.
    std::vector<float> gather(std::span<float const> values) const;

    // Whether this view is exactly [0, parent_rows). Answered in constant
    // time, which is the whole reason the descriptor exists: the fills index
    // bins and grad by position on the identity and gather on anything else,
    // and that question used to cost a pass over the row list per tree.
    bool is_identity() const
    {
        return form_ == Form::Range && !runs_.empty() && runs_.front().start == 0 &&
               runs_.front().size() == parent_rows_;
    }

    // The selection's occupancy of its own bounding span, which is the cost
    // model for reading a view out of the plane.
    double density() const
    {
        if (size_ == 0)
        {
            return 0.0;
        }
        return static_cast<double>(size_) /
               static_cast<double>(static_cast<size_t>(last_ - first_) + 1);
    }

    // The row list itself, for the paths that still take one.
    void materialize_into(std::vector<row_id_t> &out) const
    {
        if (form_ == Form::Gather)
        {
            out.assign(ids_.begin(), ids_.end());
            return;
        }
        out.clear();
        out.reserve(size_);
        for (RowRun const &run : runs_)
        {
            for (row_id_t r = run.start; r < run.stop; ++r)
            {
                out.push_back(r);
            }
        }
    }

    std::vector<row_id_t> materialize() const
    {
        std::vector<row_id_t> out;
        materialize_into(out);
        return out;
    }

  private:
    // Range holds one run, Segments a few, Gather none (ids_ holds the list).
    std::vector<RowRun>   runs_;
    std::vector<row_id_t> ids_;
    size_t                size_        = 0;
    size_t                parent_rows_ = 0;
    row_id_t              first_       = 0;
    row_id_t              last_        = 0;
    Form                  form_        = Form::Range;
};

// A view's positions as plane row ids, for the readers that answer one row per
// VIEW row in the view's order (predict and its family, the per-round eval).
// The identity materializes nothing, so a dataset that is not a view carries an
// empty vector; anything else pays one row list per call, against a tree walk
// per row.
class RowIndex
{
  public:
    explicit RowIndex(RowView const &view) : size_(view.size())
    {
        if (!view.is_identity())
        {
            view.materialize_into(ids_);
        }
    }

    size_t size() const
    {
        return size_;
    }

    row_id_t operator[](size_t position) const
    {
        return ids_.empty() ? static_cast<row_id_t>(position) : ids_[position];
    }

  private:
    std::vector<row_id_t> ids_;
    size_t                size_ = 0;
};

// What a fill may assume about a row list beyond the ids themselves.
// `identity` is true only when the list is exactly [0, n_rows), which lets
// the fills read bins and gradients at the row's position and skip the
// gather. `runs` describes the SAME list as runs of consecutive plane rows
// when the caller knows them; empty means gather.
struct RowShape
{
    bool         identity = false;
    row_run_view runs     = {};
};

// A row list plus its shape: the grow seam's one argument. Converts from a
// bare span so "just these rows, assume nothing" stays a one-liner.
struct RowSelection
{
    row_index_view rows  = {};
    RowShape       shape = {};

    RowSelection() = default;
    // NOLINTBEGIN(google-explicit-constructor): converting from a bare row
    // list is the point, so "just these rows, assume nothing" stays a
    // one-liner at every call site.
    RowSelection(row_index_view r) : rows(r) {}
    RowSelection(std::vector<row_id_t> const &r) : rows(r) {}
    // NOLINTEND(google-explicit-constructor)
    RowSelection(row_index_view r, RowShape s) : rows(r), shape(s) {}
};

inline std::vector<float> RowView::gather(std::span<float const> values) const
{
    RowIndex const     rows{*this};
    std::vector<float> out;
    out.reserve(rows.size());
    for (size_t k = 0; k < rows.size(); ++k)
    {
        out.push_back(values[rows[k]]);
    }
    return out;
}

} // namespace bonsai
