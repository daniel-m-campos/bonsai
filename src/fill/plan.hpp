#pragma once

#include "bonsai/dataset.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/split.hpp"
#include "bonsai/types.hpp"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <optional>
#include <ranges>
#include <span>
#include <vector>

namespace bonsai::fill_detail
{

inline std::span<HistCell> partials_storage(size_t n_cells)
{
    static thread_local std::vector<HistCell> cells;
    if (cells.size() < n_cells)
    {
        cells.assign(n_cells, HistCell{});
    }
    return {cells.data(), n_cells};
}

struct MirrorSlice
{
    size_t s0, s1;
    size_t rm_base, rm_width;
    size_t cell0, cells;
    size_t fid0;

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
                          .rm_base  = ds.plane_n_rows() * tile * width,
                          .rm_width = tile_width,
                          .cell0    = offsets[s],
                          .cells    = cell_end - offsets[s],
                          .fid0     = tile * width});
        s = e;
    }
    return slices;
}

inline std::vector<size_t> selected_bins(Dataset const                &ds,
                                         std::span<feature_id_t const> selected)
{
    return selected |
           std::views::transform([&](feature_id_t fid) { return ds.n_bins(fid); }) |
           std::ranges::to<std::vector<size_t>>();
}

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

    std::vector<size_t>      bins;
    ArenaLayout              layout;
    std::vector<size_t>      offsets;
    size_t                   total_cells = 0;
    std::vector<MirrorSlice> slices;
};

struct PlanCache
{
    std::optional<SelectionPlan>                         plan;
    std::optional<SelectionPlan>                         totals;
    feature_id_t                                         totals_feature = 0;
    std::optional<std::reference_wrapper<Dataset const>> ds;
    std::span<feature_id_t const>                        selected;
    floats_view                                          hess;
    bool                                                 unit_hess = false;

    bool holds(Dataset const &other) const
    {
        return ds && &ds->get() == &other;
    }
};

inline PlanCache &plan_cache()
{
    static thread_local PlanCache cache;
    return cache;
}

inline floats_view fill_hess(floats_view hess)
{
    PlanCache const &cache = plan_cache();
    return cache.unit_hess && cache.hess.data() == hess.data() ? floats_view{} : hess;
}

inline SelectionPlan const &selection_plan(Dataset const                &ds,
                                           std::span<feature_id_t const> selected)
{
    PlanCache &cache = plan_cache();
    if (!cache.plan)
    {
        cache.plan.emplace(ds, selected);
        cache.ds       = std::cref(ds);
        cache.selected = selected;
    }
    assert(cache.holds(ds) && cache.selected.data() == selected.data() &&
           cache.selected.size() == selected.size());
    return *cache.plan;
}
inline SelectionPlan const &totals_plan(Dataset const &ds, feature_id_t feature)
{
    PlanCache &cache = plan_cache();
    if (!cache.totals || cache.totals_feature != feature)
    {
        cache.totals_feature = feature;
        cache.totals.emplace(ds,
                             std::span<feature_id_t const>{&cache.totals_feature, 1});
    }
    assert(!cache.ds || cache.holds(ds));
    return *cache.totals;
}

} // namespace bonsai::fill_detail
