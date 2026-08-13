#pragma once

// The row-wise fill over the row-major mirror, one pass per mirror slice
// with rows inner, and the level plane's buffer-and-reduce partition of
// a sparse frontier's row chunks.
//
// Included into src/grower.cpp only: the fill's pieces stay in one
// translation unit, so nothing here crosses a call the optimizer cannot see.

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

// The partials are the same grid with a run-time row width: one row per
// partition slot, packed the way the slice's node arenas are.
using PartialsView = std::mdspan<HistCell, std::dextents<size_t, 2>>;

// The per-(thread, node) scratch histograms the reduce merges, plus the flag
// per slot saying whether this slice has zeroed it yet.
struct Partials
{
    PartialsView       cells;
    std::span<uint8_t> used;
};

// How a byte of the row's bins becomes a histogram cell address.
enum class CellMode : uint8_t
{
    uniform,  // arithmetic over a node's padded u8 arena, no base load
    dense,    // one base per column of the row's bins, the whole tile selected
    gathered, // column sampling: a selection lookup per add
};

// Where one row chunk's adds land: the slice being filled, its mirror tile,
// one histogram base per selected feature of the slice, the full selection
// (indexed from `slice.s0`), and the addressing mode.
struct FillTarget
{
    MirrorSlice const            &slice;
    std::span<uint8_t const>      rm;
    std::span<HistCell *const>    bases;
    std::span<feature_id_t const> selected;
    CellMode                      mode;
};

// Accumulates rows [first, last) of one node into `target`, reading each row's
// bins as contiguous bytes of the slice's mirror tile and grad/hess once per
// row. NodeOrder says grad/hess are already gathered into the node's row
// order, so they are read at the row's position instead of its id. An empty
// `hess` is the unit hessian: every row adds the literal 1.0F.
template <bool NodeOrder = false>
inline void fill_rows(SplitInput const &node, size_t first, size_t last,
                      floats_view grad, floats_view hess, FillTarget const &target)
{
    // The mirror is walked by hand from one base: a row's bins are
    // rm_ptr + (row * width), which is this loop's whole addressing.
    uint8_t const *const                rm_ptr  = target.rm.data();
    std::span<HistCell *const> const    bases   = target.bases;
    std::span<feature_id_t const> const sel     = target.selected;
    size_t const                        width   = target.slice.rm_width;
    size_t const                        s0      = target.slice.s0;
    size_t const                        fid0    = target.slice.fid0;
    size_t const                        n_sel_b = target.slice.n_selected();
    std::span<row_id_t const> const     rows    = node.rows;
    // Prefetch distance: the row loop is DRAM-latency-bound at depth, and the
    // lookahead reads the node's row list rather than this chunk's, so a chunk
    // boundary costs no dead zone (docs/architecture/7-parallel.md). Reads
    // only, so results are bit-identical.
    constexpr size_t k_ahead = 16;
    // The node's last rows go unprefetched, peeled out so the hot loop
    // carries no per-row bound test.
    size_t const n_rows = node.rows.size();
    size_t const kp =
        n_rows > k_ahead ? std::clamp(n_rows - k_ahead, first, last) : first;
    // A fully selected tile indexes the row's bins directly (the common case);
    // the general loop pays a selection lookup per add to support column
    // sampling. Direct fills into a u8 node compute the cell address
    // arithmetically over the node's padded arena (feature stride 256), with
    // no per-feature base load at all.
    ArenaView const a0{target.mode == CellMode::uniform ? bases[0] : nullptr, n_sel_b};
    // The hessian source is a compile-time choice, not a per-row test: this is
    // the fill's innermost loop and a branch inside it costs more than the
    // load it skips.
    auto run_rows = [&](auto cell_at, auto unit_hess)
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

// A contiguous range of one node's rows: the unit of thread work in the
// buffer-and-reduce fill.
struct RowChunk
{
    size_t node;
    size_t first, last;
};

// A sparse node's place in the row-chunk partition. The lowest thread touching
// it accumulates straight into its arena; every higher thread owns one
// partial at slot0 + (thread - first_thread - 1).
struct ReduceNode
{
    size_t first_thread = no_thread;
    size_t last_thread  = no_thread;
    size_t slot0        = 0;
};

// A level's row-chunk partition: the flat work list, each node's thread range,
// and the nodes that need a reduce at all.
struct ReducePlan
{
    std::vector<RowChunk>   chunks;
    std::vector<ReduceNode> nodes;
    std::vector<size_t>     reduce_nodes;
    size_t                  n_threads = 0;
    size_t                  n_slots   = 0;

    // Thread t owns chunks [begin(t), begin(t + 1)).
    size_t begin(size_t t) const
    {
        return t * chunks.size() / n_threads;
    }
};

// Uniform static partition: node-major fixed-grain row chunks split into
// contiguous per-thread ranges, which keeps a thread's live scatter target to
// one node histogram, bounds the partials at n_threads - 1, and fixes the
// summation order (docs/architecture/7-parallel.md).
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

// Fills every sparse node for one mirror slice through the static row-chunk
// partition, then sums each node's partials into its arena in ascending
// thread order. A partial is zeroed on first touch, so an untouched
// (thread, node) pair costs neither a zero fill nor a reduce.
inline void run_fill_reduce(ReducePlan const &plan, split_input_refs nodes,
                            Dataset const &ds, floats_view grad, floats_view hess,
                            std::span<feature_id_t const> selected,
                            SelectionPlan const &sp, MirrorSlice const &sl)
{
    static thread_local std::vector<uint8_t> touched;
    size_t const                             n_sel_b   = sl.n_selected();
    bool const                               dense_sel = n_sel_b == sl.rm_width;
    // A partial row repeats the node arena's packing, so both take the same
    // addressing: u8 pads every feature to k_feature_stride, wider bins pack.
    // Only the scratch layout moves; the adds and their order do not.
    bool const   padded             = ds.bins_are_u8();
    size_t const row_cells          = padded ? n_sel_b * k_feature_stride : sl.cells;
    std::span<HistCell> const cells = partials_storage(plan.n_slots * row_cells);
    touched.assign(plan.n_slots, 0);
    // Capture views, not the containers: naming a thread_local inside the
    // parallel region would resolve to each worker's own (empty) one.
    PartialsView const                view{cells.data(), plan.n_slots, row_cells};
    Partials const                    parts{.cells = view, .used = touched};
    std::span<size_t const> const     off    = sp.offsets;
    std::span<uint8_t const> const    rm     = ds.row_major_bins().subspan(sl.rm_base);
    std::span<RowChunk const> const   chunks = plan.chunks;
    std::span<ReduceNode const> const rns    = plan.nodes;
    // Where selected feature s starts inside one partial row.
    auto const part_cell = [&, off](size_t s)
    { return padded ? (s - sl.s0) * k_feature_stride : off[s] - sl.cell0; };
    // A fully selected tile over padded cells addresses arithmetically, with
    // no per-feature base load; anything else loads a base per feature.
    CellMode const mode = dense_sel && padded ? CellMode::uniform
                          : dense_sel         ? CellMode::dense
                                              : CellMode::gathered;
    // One index per partition slot, not per worker: buffers are keyed by the
    // index, so which worker picks it up changes nothing.
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

// Rows per chunk in the partition; measured, not tunable
// (docs/architecture/7-parallel.md).
inline constexpr size_t k_reduce_grain = 1024;

// A level's fill routes each node either to the dense column fill or to this
// sparse one. A sparse level's rows are cut into row chunks, node-major, and
// the chunk list is dealt to the threads as contiguous ranges. The lowest
// thread touching a node fills straight into that node's arena; every higher
// thread fills its own partials instead. A reduce then sums a node's partials
// into its arena in ascending thread order. The histograms being filled are
// slices carved from each node's arena, one per selected feature, at a fixed
// feature stride when the data is u8.
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
