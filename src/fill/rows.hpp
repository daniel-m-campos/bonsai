#pragma once

#include "bonsai/dataset.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/parallel.hpp"
#include "bonsai/split.hpp"
#include "bonsai/types.hpp"
#include "fill/plan.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mdspan>
#include <span>
#include <type_traits>
#include <vector>

namespace bonsai::fill_detail
{

using PartialsView = std::mdspan<HistCell, std::dextents<size_t, 2>>;

struct Partials
{
    PartialsView       cells;
    std::span<uint8_t> used;
};

enum class CellMode : uint8_t
{
    uniform,
    dense,
    gathered,
};

struct FillTarget
{
    MirrorSlice const            &slice;
    std::span<uint8_t const>      rm;
    std::span<HistCell *const>    bases;
    std::span<feature_id_t const> selected;
    CellMode                      mode;
};

template <bool NodeOrder = false>
inline void fill_rows(SplitInput const &node, size_t first, size_t last,
                      floats_view grad, floats_view hess, FillTarget const &target)
{
    uint8_t const *const                rm_ptr  = target.rm.data();
    std::span<HistCell *const> const    bases   = target.bases;
    std::span<feature_id_t const> const sel     = target.selected;
    size_t const                        width   = target.slice.rm_width;
    size_t const                        s0      = target.slice.s0;
    size_t const                        fid0    = target.slice.fid0;
    size_t const                        n_sel_b = target.slice.n_selected();
    std::span<row_id_t const> const     rows    = node.rows;
    constexpr size_t                    k_ahead = 16;
    size_t const                        n_rows  = node.rows.size();
    size_t const                        kp =
        n_rows > k_ahead ? std::clamp(n_rows - k_ahead, first, last) : first;
    ArenaView const a0{target.mode == CellMode::uniform ? bases[0] : nullptr, n_sel_b};
    auto            run_rows = [&](auto cell_at, auto unit_hess)
    {
        auto walk = [&](auto prefetch, size_t a, size_t b)
        {
            for (size_t k = a; k < b; ++k)
            {
                if constexpr (decltype(prefetch)::value)
                {
                    size_t const rp = rows[k + k_ahead];
                    __builtin_prefetch(rm_ptr + (rp * width), 0, 0);
                    __builtin_prefetch(rm_ptr + (rp * width) + 64, 0, 0);
                    if constexpr (!NodeOrder)
                    {
                        __builtin_prefetch(&grad[rp], 0, 0);
                        if constexpr (!decltype(unit_hess)::value)
                        {
                            __builtin_prefetch(&hess[rp], 0, 0);
                        }
                    }
                }
                size_t const         r        = rows[k];
                uint8_t const *const row_bins = rm_ptr + (r * width);
                size_t const         gk       = NodeOrder ? k : r;
                float const          g        = grad[gk];
                float const          h = decltype(unit_hess)::value ? 1.0F : hess[gk];
                for (size_t s = 0; s < n_sel_b; ++s)
                {
                    HistCell &c = cell_at(s, row_bins);
                    c.sum_grad += g;
                    c.sum_hess += h;
                }
            }
        };
        walk(std::true_type{}, first, kp);
        walk(std::false_type{}, kp, last);
    };
    auto by_mode = [&](auto unit_hess)
    {
        if (target.mode == CellMode::uniform)
        {
            run_rows([&](size_t s, uint8_t const *row_bins) -> HistCell &
                     { return a0[s, row_bins[s]]; }, unit_hess);
        }
        else if (target.mode == CellMode::dense)
        {
            run_rows([&](size_t s, uint8_t const *row_bins) -> HistCell &
                     { return bases[s][row_bins[s]]; }, unit_hess);
        }
        else
        {
            run_rows([&](size_t s, uint8_t const *row_bins) -> HistCell &
                     { return bases[s][row_bins[sel[s0 + s] - fid0]]; }, unit_hess);
        }
    };
    if (hess.empty())
    {
        by_mode(std::true_type{});
    }
    else
    {
        by_mode(std::false_type{});
    }
}

inline constexpr size_t no_thread = static_cast<size_t>(-1);

struct RowChunk
{
    size_t node;
    size_t first, last;
};

struct ReduceNode
{
    size_t first_thread = no_thread;
    size_t last_thread  = no_thread;
    size_t slot0        = 0;
};

struct ReducePlan
{
    std::vector<RowChunk>   chunks;
    std::vector<ReduceNode> nodes;
    std::vector<size_t>     reduce_nodes;
    size_t                  n_threads = 0;
    size_t                  n_slots   = 0;

    size_t begin(size_t t) const
    {
        return t * chunks.size() / n_threads;
    }
};

inline ReducePlan const &plan_reduce(split_input_refs nodes, size_t grain)
{
    static thread_local ReducePlan plan;
    plan.chunks.clear();
    plan.reduce_nodes.clear();
    plan.nodes.assign(nodes.size(), ReduceNode{});
    for (size_t i = 0; i < nodes.size(); ++i)
    {
        size_t const n = nodes[i].get().rows.size();
        if (n == 0)
        {
            continue;
        }
        size_t const n_chunks = (n + grain - 1) / grain;
        for (size_t b = 0; b < n_chunks; ++b)
        {
            plan.chunks.push_back({i, b * n / n_chunks, (b + 1) * n / n_chunks});
        }
    }
    plan.n_threads = static_cast<size_t>(parallel::n_threads());
    for (size_t t = 0; t < plan.n_threads; ++t)
    {
        for (size_t i = plan.begin(t); i < plan.begin(t + 1); ++i)
        {
            ReduceNode &rn = plan.nodes[plan.chunks[i].node];
            if (rn.first_thread == no_thread)
            {
                rn.first_thread = t;
            }
            rn.last_thread = t;
        }
    }
    plan.n_slots = 0;
    for (size_t i = 0; i < plan.nodes.size(); ++i)
    {
        ReduceNode &rn = plan.nodes[i];
        if (rn.first_thread == no_thread || rn.last_thread == rn.first_thread)
        {
            continue;
        }
        rn.slot0 = plan.n_slots;
        plan.n_slots += rn.last_thread - rn.first_thread;
        plan.reduce_nodes.push_back(i);
    }
    return plan;
}

inline void run_fill_reduce(ReducePlan const &plan, split_input_refs nodes,
                            Dataset const &ds, floats_view grad, floats_view hess,
                            std::span<feature_id_t const> selected,
                            SelectionPlan const &sp, MirrorSlice const &sl)
{
    static thread_local std::vector<uint8_t> touched;
    size_t const                             n_sel_b   = sl.n_selected();
    bool const                               dense_sel = n_sel_b == sl.rm_width;
    bool const                               padded    = ds.bins_are_u8();
    size_t const row_cells          = padded ? n_sel_b * k_feature_stride : sl.cells;
    std::span<HistCell> const cells = partials_storage(plan.n_slots * row_cells);
    touched.assign(plan.n_slots, 0);
    PartialsView const                view{cells.data(), plan.n_slots, row_cells};
    Partials const                    parts{.cells = view, .used = touched};
    std::span<size_t const> const     off    = sp.offsets;
    std::span<uint8_t const> const    rm     = ds.mirror().bins().subspan(sl.rm_base);
    std::span<RowChunk const> const   chunks = plan.chunks;
    std::span<ReduceNode const> const rns    = plan.nodes;
    auto const                        part_cell = [&, off](size_t s)
    { return padded ? (s - sl.s0) * k_feature_stride : off[s] - sl.cell0; };
    CellMode const mode = dense_sel && padded ? CellMode::uniform
                          : dense_sel         ? CellMode::dense
                                              : CellMode::gathered;
    parallel::for_each_index(
        plan.n_threads,
        [&, parts, selected, rm, chunks, rns, nodes](size_t t)
        {
            static thread_local std::vector<HistCell *> bases;
            bases.resize(n_sel_b);
            size_t last_node = nodes.size();
            for (size_t i = plan.begin(t); i < plan.begin(t + 1); ++i)
            {
                RowChunk const   &chunk  = chunks[i];
                ReduceNode const &rn     = rns[chunk.node];
                bool const        direct = t == rn.first_thread;
                size_t const slot = direct ? 0 : rn.slot0 + (t - rn.first_thread - 1);
                if (!direct && parts.used[slot] == 0)
                {
                    std::fill_n(&parts.cells[slot, 0], row_cells, HistCell{});
                    parts.used[slot] = 1;
                }
                if (chunk.node != last_node)
                {
                    SplitInput &node = nodes[chunk.node];
                    for (size_t s = sl.s0; s < sl.s1; ++s)
                    {
                        bases[s - sl.s0] = direct
                                               ? node.hists[selected[s]].cells().data()
                                               : &parts.cells[slot, part_cell(s)];
                    }
                    last_node = chunk.node;
                }
                FillTarget const target{.slice    = sl,
                                        .rm       = rm,
                                        .bases    = bases,
                                        .selected = selected,
                                        .mode     = mode};
                fill_rows(nodes[chunk.node], chunk.first, chunk.last, grad, hess,
                          target);
            }
        });
    std::span<size_t const> const reds = plan.reduce_nodes;
    parallel::for_each_index(
        plan.reduce_nodes.size() * n_sel_b,
        [&, parts, selected, rns, nodes, reds](size_t j)
        {
            size_t const      ni   = reds[j / n_sel_b];
            ReduceNode const &rn   = rns[ni];
            size_t const      s    = sl.s0 + (j % n_sel_b);
            size_t const      cell = part_cell(s);
            Histogram        &h    = nodes[ni].get().hists[selected[s]];
            for (size_t t = rn.first_thread + 1; t <= rn.last_thread; ++t)
            {
                size_t const slot = rn.slot0 + (t - rn.first_thread - 1);
                if (parts.used[slot] != 0)
                {
                    h.add_cells({&parts.cells[slot, cell], h.size()});
                }
            }
        });
}

// perf: Rows per chunk in the partition; measured, not tunable
// by measurement.
inline constexpr size_t k_reduce_grain = 1024;

inline void fill_sparse(Dataset const &ds, floats_view grad, floats_view hess,
                        split_input_refs nodes, std::span<feature_id_t const> selected,
                        SelectionPlan const &sp)
{
    ReducePlan const &plan = plan_reduce(nodes, k_reduce_grain);
    for (MirrorSlice const &sl : sp.slices)
    {
        run_fill_reduce(plan, nodes, ds, grad, hess, selected, sp, sl);
    }
}

} // namespace bonsai::fill_detail
