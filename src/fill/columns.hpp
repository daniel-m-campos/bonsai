#pragma once

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

// perf: Prefetch distance for the column fill's gathered arm, in rows: the loop
// carries one L1-resident add per row, so the lookahead has to cover a DRAM
// latency in row iterations rather than the mirror fill's 16
// of 16.
inline constexpr size_t k_col_ahead = 64;

inline void fill_column(Dataset const &ds, feature_id_t fid, Histogram &h,
                        std::span<row_id_t const> rows, bool identity, GhView const &gh)
{
    size_t const                 n  = rows.size();
    std::span<float const> const og = gh.g;
    std::span<float const> const oh = gh.h;
    ds.visit_bins(fid,
                  [&](auto bins)
                  {
                      auto add = [&](auto hess_of)
                      {
                          for (size_t k = 0; k < n; ++k)
                          {
                              h.add(bins[k], og[k], hess_of(k));
                          }
                      };
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

// perf: Whether reading runs as subspans beats gathering them. The run arm gives up
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

inline bool node_fills_from_runs(SplitInput const &node)
{
    return !node.shape.identity && runs_beat_gather(node);
}

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
