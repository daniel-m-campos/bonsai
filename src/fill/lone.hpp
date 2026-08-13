#pragma once

// The lone node's fill: the leaf plane has no frontier to spread over,
// so the fill cuts the node itself into feature ranges by row blocks.
//
// Included into src/grower.cpp only: the fill's pieces stay in one
// translation unit, so nothing here crosses a call the optimizer cannot see.

#include "bonsai/dataset.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/parallel.hpp"
#include "bonsai/split.hpp"
#include "bonsai/types.hpp"
#include "fill/ordered.hpp"
#include "fill/plan.hpp"
#include "fill/rows.hpp"
#include <algorithm>
#include <cstddef>
#include <mdspan>
#include <span>
#include <vector>

namespace bonsai::fill_detail
{

// One worker's share of a lone node's fill: the selected features [b0, b1) of
// one mirror tile, over the rows of one row block. Feature ranges are
// contiguous so the arena cells the worker touches are its own; row blocks
// beyond the first accumulate into a partial instead.
struct FillBlock
{
    size_t slice, b0, b1, block;
};

// A lone node's fill, cut into `blocks` row blocks by `ranges` feature ranges.
// A feature range keeps a worker's arena cells to itself; a row block keeps
// its mirror bytes to itself, at the price of one partial arena. Block 0
// accumulates straight into the node's arena, zeroing its own feature range
// first; a later block zeroes and fills its own partial, and a reduce sums the
// partials into the arena in ascending block order. A single block leaves
// nothing to reduce, so the sibling subtraction rides the fill and every cell
// takes its rows in the serial order whatever the range count. The region asks
// for `ranges * blocks` workers, so a plan resolving to the configured count
// leaves the team the runtime keeps hot alone and one that does not pays to
// rebuild it. The row-chunk fill the level plane uses stays as it is: there a
// worker owns whole nodes, and the partials it does write are the price of
// spreading a frontier.
inline void fill_lone(Dataset const &ds, SplitInput &node,
                      std::span<feature_id_t const> selected, SelectionPlan const &sp,
                      ArenaLayout const &carve, NodeHistograms &sibling, size_t ranges,
                      size_t blocks, GhView const &gh)
{
    size_t const              n         = node.rows.size();
    size_t const              n_sel     = selected.size();
    size_t const              row_cells = n_sel * k_feature_stride;
    std::span<HistCell> const cells     = partials_storage((blocks - 1) * row_cells);
    PartialsView const        view{cells.data(), blocks - 1, row_cells};
    static thread_local std::vector<FillBlock> work;
    work.clear();
    // A range walks its block's rows, so equal feature counts are equal work:
    // one range per worker of a block, cut inside the mirror tiles it spans.
    size_t const width = std::max<size_t>(1, (n_sel + ranges - 1) / ranges);
    for (size_t i = 0; i < sp.slices.size(); ++i)
    {
        size_t const n_sel_b = sp.slices[i].n_selected();
        for (size_t b = 0; b < n_sel_b; b += width)
        {
            for (size_t j = 0; j < blocks; ++j)
            {
                work.push_back({.slice = i,
                                .b0    = b,
                                .b1    = std::min(b + width, n_sel_b),
                                .block = j});
            }
        }
    }
    std::span<uint8_t const> const rm_all = ds.row_major_bins();
    NodeHistograms                &hists  = node.hists;
    // Capture a view, not the container: naming the thread_local inside the
    // parallel region would resolve to each worker's own (empty) vector.
    std::span<FillBlock const> const items = work;
    parallel::for_each_index_on(
        static_cast<int>(ranges * blocks), work.size(),
        [&, view, rm_all, selected, items](size_t t)
        {
            FillBlock const   &r      = items[t];
            MirrorSlice const &sl     = sp.slices[r.slice];
            bool const         direct = r.block == 0;
            // A fully selected tile indexes the row's bins directly, so the
            // range addresses its cells from one base; a sampled one pays a
            // selection lookup and a base load per feature.
            bool const        dense_sel = sl.n_selected() == sl.rm_width;
            MirrorSlice const sub{.s0       = sl.s0 + r.b0,
                                  .s1       = sl.s0 + r.b1,
                                  .rm_base  = sl.rm_base,
                                  .rm_width = sl.rm_width,
                                  .cell0    = sl.cell0,
                                  .cells    = sl.cells,
                                  .fid0     = sl.fid0 + r.b0};
            HistCell *const   part    = direct ? nullptr : &view[r.block - 1, 0];
            auto const        base_of = [&](size_t s)
            {
                return direct ? hists[selected[s]].cells().data()
                              : part + (s * k_feature_stride);
            };
            if (direct)
            {
                for (size_t s = sub.s0; s < sub.s1; ++s)
                {
                    node.hists.carve_run(carve, selected, s);
                }
            }
            else
            {
                std::fill_n(base_of(sub.s0), (sub.s1 - sub.s0) * k_feature_stride,
                            HistCell{});
            }
            static thread_local std::vector<HistCell *> bases;
            bases.clear();
            bases.push_back(base_of(sub.s0));
            if (!dense_sel)
            {
                for (size_t s = sub.s0 + 1; s < sub.s1; ++s)
                {
                    bases.push_back(base_of(s));
                }
            }
            FillTarget const target{.slice    = sub,
                                    .rm       = rm_all.subspan(sl.rm_base + r.b0),
                                    .bases    = bases,
                                    .selected = selected,
                                    .mode     = dense_sel ? CellMode::uniform
                                                          : CellMode::gathered};
            fill_rows<true>(node, n * r.block / blocks, n * (r.block + 1) / blocks,
                            gh.g, gh.h, target);
            if (blocks == 1)
            {
                for (size_t s = sub.s0; s < sub.s1; ++s)
                {
                    sibling[selected[s]] -= hists[selected[s]];
                }
            }
        });
    if (blocks == 1)
    {
        return;
    }
    // The reduce owns the sibling subtraction too: the arena is not final
    // until a feature's partials are in, so the fill cannot carry it.
    parallel::for_each_index_on(
        static_cast<int>(ranges * blocks), n_sel,
        [&, view, selected](size_t s)
        {
            Histogram &h = hists[selected[s]];
            for (size_t j = 1; j < blocks; ++j)
            {
                h.add_cells({&view[j - 1, s * k_feature_stride], h.size()});
            }
            sibling[selected[s]] -= h;
        });
}

// Cache line the mirror is read in: a feature range narrower than one line
// shares its lines with its neighbours, so each of those workers refetches
// what the others already pulled.
inline constexpr size_t k_line_bytes = 64;

// Cells one worker must own for a fill to spread at all: below it the region
// entry costs more than the work it splits (measured, #367). A fill's cells
// are the arena it zeroes plus the accumulates it makes, so a wide node is
// never small however few rows it holds.
inline constexpr size_t k_fill_cells_per_worker = 1 << 15;

// Rows one row block must own to earn its partial arena, which is zeroed and
// reduced whatever the block fills (measured, #367).
inline constexpr size_t k_fill_rows_per_block = 1 << 13;

// How a lone node's fill decomposes: feature ranges by row blocks.
struct FillPlan
{
    size_t ranges;
    size_t blocks;
};

// The decomposition rule: size the team to the work, spend it on row blocks
// while each one earns its partial and the mirror row is too narrow to give
// every worker its own cache line, and on feature ranges after that.
inline FillPlan plan_lone_fill(size_t n_rows, SelectionPlan const &sp, size_t threads)
{
    size_t const n_sel   = sp.offsets.size();
    size_t const workers = std::clamp(((n_sel * k_feature_stride) + (n_rows * n_sel)) /
                                          k_fill_cells_per_worker,
                                      size_t{1}, threads);
    size_t       lines   = 0;
    for (MirrorSlice const &sl : sp.slices)
    {
        // A sampled tile's ranges are scattered over the whole tile, so they
        // are line-private however narrow the selection is.
        lines += sl.n_selected() == sl.rm_width
                     ? (sl.n_selected() + k_line_bytes - 1) / k_line_bytes
                     : sl.n_selected();
    }
    size_t const blocks =
        std::clamp(workers / std::max(lines, size_t{1}), size_t{1},
                   std::max(n_rows / k_fill_rows_per_block, size_t{1}));
    return {.ranges = std::clamp(workers / blocks, size_t{1}, n_sel), .blocks = blocks};
}

} // namespace bonsai::fill_detail
