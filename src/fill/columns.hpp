#pragma once

// The column fill: one worker per selected feature, each filling its
// own histogram in the node's row order.
//
// Included into src/grower.cpp only: the fill's pieces stay in one
// translation unit, so nothing here crosses a call the optimizer cannot see.

#include "bonsai/dataset.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/parallel.hpp"
#include "bonsai/row_view.hpp"
#include "bonsai/split.hpp"
#include "bonsai/types.hpp"
#include "fill/ordered.hpp"
#include <cstddef>
#include <span>

namespace bonsai::fill_detail
{

// Prefetch distance for the column fill's gathered arm, in rows: the loop
// carries one L1-resident add per row, so the lookahead has to cover a DRAM
// latency in row iterations rather than the mirror fill's 16
// of 16.
inline constexpr size_t k_col_ahead = 64;

// One feature's column fill: the thread owning this histogram fills it in the
// node's row order, so the cell sums are bit-identical at any thread count.
// visit_bins monomorphizes the fill per bin width. `identity` says the node's
// rows are [0, n_rows), so bins and gh are both indexed by position and no
// gather happened.
inline void fill_column(Dataset const &ds, feature_id_t fid, Histogram &h,
                        std::span<row_id_t const> rows, bool identity, GhView const &gh)
{
    size_t const                 n  = rows.size();
    std::span<float const> const og = gh.g;
    std::span<float const> const oh = gh.h;
    ds.visit_bins(fid,
                  [&](auto bins)
                  {
                      // Hoisted out of the row loop: the hessian selector is
                      // loop-invariant, and this is the fill's innermost work.
                      auto add = [&](auto hess_of)
                      {
                          for (size_t k = 0; k < n; ++k)
                          {
                              h.add(bins[k], og[k], hess_of(k));
                          }
                      };
                      // Below the root a node's rows are an ascending SUBSET, so
                      // this column's bytes sit at irregular strides the hardware
                      // prefetcher cannot follow and the gather runs
                      // DRAM-latency-bound. Pull the byte a fixed distance
                      // ahead; reads only, so the sums are untouched. The node's
                      // last rows go unprefetched, peeled out so the hot loop
                      // carries no per-row bound test.
                      auto gather = [&](auto hess_of)
                      {
                          size_t const kp = n > k_col_ahead ? n - k_col_ahead : 0;
                          for (size_t k = 0; k < kp; ++k)
                          {
                              __builtin_prefetch(&bins[rows[k + k_col_ahead]], 0, 0);
                              h.add(bins[rows[k]], og[k], hess_of(k));
                          }
                          for (size_t k = kp; k < n; ++k)
                          {
                              h.add(bins[rows[k]], og[k], hess_of(k));
                          }
                      };
                      // Identity picks the arm; the hessian selector is picked
                      // once and rides into whichever arm runs.
                      auto fill = [&](auto hess_of)
                      {
                          if (identity)
                          {
                              add(hess_of);
                              return;
                          }
                          gather(hess_of);
                      };
                      if (oh.empty())
                      {
                          fill([](size_t) { return 1.0F; });
                      }
                      else
                      {
                          fill([oh](size_t k) { return oh[k]; });
                      }
                  });
}

// One feature's column fill for a node whose rows are runs of consecutive
// plane rows: a run's bins sit contiguously, so each is a subspan indexed by
// position and no row list is read at all. The bound rides the subspan, so a
// run past the column's end cannot pass for a read inside it.
//
// Its own function rather than a third arm of fill_column: the gathered arm's
// loop is DRAM-latency-bound and measurably sensitive to what shares its
// frame, and a node that takes this path never takes that one.
inline void fill_column_runs(Dataset const &ds, feature_id_t fid, Histogram &h,
                             row_run_view runs, GhView const &gh)
{
    std::span<float const> const og = gh.g;
    std::span<float const> const oh = gh.h;
    ds.visit_bins(fid,
                  [&](auto bins)
                  {
                      auto fill = [&](auto hess_of)
                      {
                          size_t k = 0;
                          for (RowRun const &run : runs)
                          {
                              auto const seg = bins.subspan(
                                  static_cast<size_t>(run.start), run.size());
                              for (size_t j = 0; j < seg.size(); ++j, ++k)
                              {
                                  h.add(seg[j], og[k], hess_of(k));
                              }
                          }
                      };
                      if (oh.empty())
                      {
                          fill([](size_t) { return 1.0F; });
                      }
                      else
                      {
                          fill([oh](size_t k) { return oh[k]; });
                      }
                  });
}

// Whether reading runs as subspans beats gathering them. The run arm gives up
// the gathered arm's software prefetch, and that lookahead is what keeps a
// DRAM-latency-bound read fast, so a run has to be long enough to cover it:
// measured on an M2 at 2 threads, mean run 5000 recovers the whole view
// penalty while mean run 5 costs 37% MORE than gathering. The test is the mean
// run length against the prefetch distance, in O(1).
inline bool runs_beat_gather(SplitInput const &node)
{
    return !node.shape.runs.empty() &&
           node.rows.size() >= node.shape.runs.size() * k_col_ahead;
}

// Which of the two arms a node takes: runs when it has enough of a run to
// spend, and the identity is not one of them (it keeps the dense arm it has
// always taken). Asked once per node, outside the per-feature region, since
// every feature of a node answers it the same way.
inline bool node_fills_from_runs(SplitInput const &node)
{
    return !node.shape.identity && runs_beat_gather(node);
}

// The column fill, taken by u16 (high max_bin) data and by dense u8 nodes:
// one worker per selected feature.
inline void fill_columns(Dataset const &ds, floats_view grad, floats_view hess,
                         SplitInput &node, std::span<feature_id_t const> selected)
{
    GhView const gh = node_gh(ds, node, grad, hess);
    if (node_fills_from_runs(node))
    {
        parallel::for_each_index(selected.size(),
                                 [&](size_t s)
                                 {
                                     feature_id_t const fid = selected[s];
                                     fill_column_runs(ds, fid, node.hists[fid],
                                                      node.shape.runs, gh);
                                 });
        return;
    }
    parallel::for_each_index(selected.size(),
                             [&](size_t s)
                             {
                                 feature_id_t const fid = selected[s];
                                 fill_column(ds, fid, node.hists[fid], node.rows,
                                             node.shape.identity, gh);
                             });
}

// The same column fill for a lone node, with the carve and the sibling
// subtraction riding it: the worker owning a feature zeroes that feature's
// arena run and subtracts its histogram from the sibling without leaving the
// region. No run arm here: only a root carries runs, and the lone path hands
// the root back to the level plane's fill.
inline void fill_columns_lone(Dataset const &ds, floats_view grad, floats_view hess,
                              SplitInput &node, std::span<feature_id_t const> selected,
                              ArenaLayout const &carve, NodeHistograms &sibling)
{
    GhView const gh = node_gh(ds, node, grad, hess);
    parallel::for_each_index(selected.size(),
                             [&](size_t s)
                             {
                                 node.hists.carve_run(carve, selected, s);
                                 feature_id_t const fid = selected[s];
                                 Histogram         &h   = node.hists[fid];
                                 fill_column(ds, fid, h, node.rows, node.shape.identity,
                                             gh);
                                 sibling[fid] -= h;
                             });
}

} // namespace bonsai::fill_detail
