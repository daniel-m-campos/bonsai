#pragma once

// What every fill of one tree derives from the dataset and the tree's
// feature selection, held in a per-thread cache: the arena packing, the
// per-feature cell offsets, the mirror-tile slices, and the partials
// storage the spreading fills scratch in.
//
// Included into src/grower.cpp only: the fill's pieces stay in one
// translation unit, so nothing here crosses a call the optimizer cannot see.

#include "bonsai/dataset.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/split.hpp"
#include "bonsai/types.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <optional>
#include <ranges>
#include <span>
#include <vector>

namespace bonsai::fill_detail
{

// Grow-only storage for the fill's partials: kept at its high-water mark for
// the whole fit, so its zero fill is paid a handful of times per process
// per process.
inline std::span<HistCell> partials_storage(size_t n_cells)
{
    static thread_local std::vector<HistCell> cells;
    if (cells.size() < n_cells)
    {
        cells.assign(n_cells, HistCell{});
    }
    return {cells.data(), n_cells};
}

// One mirror tile's slice of the selection: sel indices [s0, s1), the tile's
// byte offset and row-bins width in the tiled mirror, and the slice's cell
// range within the full selection's offsets. A fill runs one pass per slice
// (tiles outer, rows inner), so the live scatter target is one tile's
// histograms, cache-resident by construction, while every read stays
// sequential inside the tile. Per-feature accumulation order is unchanged
// from the untiled fill, so models are bit-identical.
struct MirrorSlice
{
    size_t s0, s1;
    size_t rm_base, rm_width;
    size_t cell0, cells;
    size_t fid0; // first feature id of the slice's mirror tile

    size_t n_selected() const
    {
        return s1 - s0;
    }
};

inline std::vector<MirrorSlice> mirror_slices(Dataset const                &ds,
                                              std::span<feature_id_t const> selected,
                                              std::span<size_t const>       offsets,
                                              size_t                        total_cells)
{
    std::vector<MirrorSlice> slices;
    size_t const             width = Dataset::mirror_tile_width();
    size_t const             f     = ds.n_features();
    size_t                   s     = 0;
    while (s < selected.size())
    {
        size_t const tile       = selected[s] / width;
        size_t const tile_width = std::min(width, f - (tile * width));
        size_t       e          = s;
        while (e < selected.size() && selected[e] / width == tile)
        {
            ++e;
        }
        size_t const cell_end = e < selected.size() ? offsets[e] : total_cells;
        slices.push_back({.s0       = s,
                          .s1       = e,
                          .rm_base  = ds.n_rows() * tile * width,
                          .rm_width = tile_width,
                          .cell0    = offsets[s],
                          .cells    = cell_end - offsets[s],
                          .fid0     = tile * width});
        s = e;
    }
    return slices;
}

// The selected features' bin counts, in selection order: all the arena
// layout needs from the dataset.
inline std::vector<size_t> selected_bins(Dataset const                &ds,
                                         std::span<feature_id_t const> selected)
{
    return selected |
           std::views::transform([&](feature_id_t fid) { return ds.n_bins(fid); }) |
           std::ranges::to<std::vector<size_t>>();
}

// Everything a fill derives from the dataset and the tree's feature
// selection: the arena packing, each selected feature's cell offset, and the
// mirror-tile slices. Identical for every node of the tree, and O(n_features)
// to build, so the leafwise grower's one-node fills would otherwise rebuild it
// once per split instead of once per level.
struct SelectionPlan
{
    SelectionPlan(Dataset const &ds, std::span<feature_id_t const> selected)
        : bins(selected_bins(ds, selected)), layout(bins, ds.bins_are_u8()),
          offsets(selected.size())
    {
        for (size_t s = 0; s < selected.size(); ++s)
        {
            offsets[s] = total_cells;
            total_cells += bins[s];
        }
        slices = mirror_slices(ds, selected, offsets, total_cells);
    }

    std::vector<size_t>      bins; // layout views this, so it is declared first
    ArenaLayout              layout;
    std::vector<size_t>      offsets;
    size_t                   total_cells = 0;
    std::vector<MirrorSlice> slices;
};

// The tree's plan, held across its fills. A grower samples its features once
// per tree and hands the same span to every fill, so one plan serves the whole
// tree; begin_tree drops it, since the next tree may draw a different
// selection into the same buffer. That call is the whole freshness protocol,
// for the plan and for the unit-hessian flag alike: a fill reached without it
// reads whatever the last tree left.
struct PlanCache
{
    std::optional<SelectionPlan> plan;
    // What the held plan was built from, checked on every use in debug builds
    // the way the carve's work list is: begin_tree is the whole protocol, and
    // the one path that ever reached a fill without it was a live bug.
    Dataset const      *ds  = nullptr;
    feature_id_t const *sel = nullptr;
    size_t              n   = 0;
    // Every row's hessian is exactly 1.0F this tree, so the fills add the
    // literal instead of reading one. begin_tree owns this: it is the one
    // call that says which hessians a tree will fill from, and the pointer
    // it saw guards against a fill handed a different array.
    float const *hess      = nullptr;
    bool         unit_hess = false;
};

inline PlanCache &plan_cache()
{
    static thread_local PlanCache cache;
    return cache;
}

// What the fills read for hessians: an empty view when every row's hessian is
// 1.0F, which the fills answer with the literal. Adding 1.0F is the same
// float operation the gathered value would have performed, so cell sums,
// their order, and the sibling subtraction are untouched; what goes is the
// gather and the stream the fill re-reads once per feature.
inline floats_view fill_hess(floats_view hess)
{
    PlanCache const &cache = plan_cache();
    return cache.unit_hess && cache.hess == hess.data() ? floats_view{} : hess;
}

inline SelectionPlan const &selection_plan(Dataset const                &ds,
                                           std::span<feature_id_t const> selected)
{
    PlanCache &cache = plan_cache();
    if (!cache.plan)
    {
        cache.plan.emplace(ds, selected);
        cache.ds  = &ds;
        cache.sel = selected.data();
        cache.n   = selected.size();
    }
    assert(cache.ds == &ds && cache.sel == selected.data() &&
           cache.n == selected.size());
    return *cache.plan;
}

} // namespace bonsai::fill_detail
