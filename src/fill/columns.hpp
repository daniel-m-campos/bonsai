#pragma once

// The column fill: one worker per selected feature, each filling its
// own histogram in the node's row order.
//
// Included into src/grower.cpp only: the fill's pieces stay in one
// translation unit, so nothing here crosses a call the optimizer cannot see.

#include "bonsai/dataset.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/parallel.hpp"
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
// (docs/architecture/7-parallel.md).
inline constexpr size_t k_col_ahead = 64;

// One feature's column fill: the thread owning this histogram fills it in the
// node's row order, so the cell sums are bit-identical at any thread count.
// visit_bins monomorphizes the fill per bin width. `dense` says the node
// covers every row, so gh is indexed by row id and no gather happened.
inline void fill_column(Dataset const &ds, feature_id_t fid, Histogram &h,
                        std::span<row_id_t const> rows, bool dense, GhView const &gh)
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
                      // DRAM-latency-bound (#367). Pull the byte a fixed distance
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
                      // Density picks the arm; the hessian selector is picked
                      // once and rides into whichever arm runs.
                      auto fill = [&](auto hess_of)
                      {
                          if (dense)
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

// The column fill, taken by u16 (high max_bin) data and by dense u8 nodes:
// one worker per selected feature.
inline void fill_columns(Dataset const &ds, floats_view grad, floats_view hess,
                         SplitInput &node, std::span<feature_id_t const> selected)
{
    bool const   dense = node.rows.size() == ds.n_rows();
    GhView const gh    = node_gh(ds, node, grad, hess);
    parallel::for_each_index(selected.size(),
                             [&](size_t s)
                             {
                                 feature_id_t const fid = selected[s];
                                 fill_column(ds, fid, node.hists[fid], node.rows, dense,
                                             gh);
                             });
}

// The same column fill for a lone node, with the carve and the sibling
// subtraction riding it: the worker owning a feature zeroes that feature's
// arena run and subtracts its histogram from the sibling without leaving the
// region.
inline void fill_columns_lone(Dataset const &ds, floats_view grad, floats_view hess,
                              SplitInput &node, std::span<feature_id_t const> selected,
                              ArenaLayout const &carve, NodeHistograms &sibling)
{
    bool const   dense = node.rows.size() == ds.n_rows();
    GhView const gh    = node_gh(ds, node, grad, hess);
    parallel::for_each_index(selected.size(),
                             [&](size_t s)
                             {
                                 node.hists.carve_run(carve, selected, s);
                                 feature_id_t const fid = selected[s];
                                 Histogram         &h   = node.hists[fid];
                                 fill_column(ds, fid, h, node.rows, dense, gh);
                                 sibling[fid] -= h;
                             });
}

} // namespace bonsai::fill_detail
