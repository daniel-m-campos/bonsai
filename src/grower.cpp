#include "bonsai/grower.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/parallel.hpp"
#include "bonsai/split.hpp"
#include "bonsai/types.hpp"
#include "grower_impl.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mdspan>
#include <optional>
#include <ranges>
#include <span>
#include <type_traits>
#include <vector>

namespace bonsai
{

namespace
{

// grad and hess in one node's row order. A fill that walks the node once per
// feature reads them sequentially from here instead of re-walking the full
// arrays with scattered indices (n_features x full-array traffic otherwise).
struct OrderedGh
{
    float const *g;
    float const *h;
};

// Below this the gather runs serially: one pass over a node this small costs
// less than the parallel region that would spread it.
constexpr size_t k_gather_region_rows = 1 << 14;

OrderedGh ordered_gh(std::span<row_id_t const> rows, floats_view grad, floats_view hess)
{
    static thread_local std::vector<float> ordered_grad;
    static thread_local std::vector<float> ordered_hess;
    size_t const                           n    = rows.size();
    bool const                             unit = hess.empty();
    ordered_grad.resize(n);
    ordered_hess.resize(unit ? 0 : n);
    // Capture raw pointers: naming the thread_local inside the parallel
    // region would resolve to each worker's own (empty) vector.
    float *const          g     = ordered_grad.data();
    float *const          h     = unit ? nullptr : ordered_hess.data();
    row_id_t const *const r_ptr = rows.data();
    // A team of one runs the loop inline and enters no region, which is what
    // a node below the threshold wants.
    parallel::for_each_index_on(n < k_gather_region_rows ? 1 : parallel::n_threads(), n,
                                [&, g, h, r_ptr, unit](size_t k)
                                {
                                    g[k] = grad[r_ptr[k]];
                                    if (!unit)
                                    {
                                        h[k] = hess[r_ptr[k]];
                                    }
                                });
    return {.g = g, .h = h};
}

// Prefetch distance for the column fill's gathered arm, in rows: the loop
// carries one L1-resident add per row, so the lookahead has to cover a DRAM
// latency in row iterations rather than the mirror fill's 16
// (docs/architecture/7-parallel.md).
constexpr size_t k_col_ahead = 64;

// The column fill, taken by u16 (high max_bin) data and by dense u8 nodes:
// one thread owns one feature's histogram and fills it in row order, so
// results are bit-identical at any thread count. visit_bins monomorphizes
// the fill per bin width. A lone node passes `carve` and its sibling: the
// worker owning a feature zeroes its arena run and subtracts it from the
// sibling without leaving the region.
void fill_feature_parallel(Dataset const &ds, floats_view grad, floats_view hess,
                           SplitInput                   &split_input,
                           std::span<feature_id_t const> selected,
                           ArenaLayout const            *carve   = nullptr,
                           NodeHistograms               *sibling = nullptr)
{
    size_t const n = split_input.rows.size();
    // A node covering every row (the root, absent row sampling) needs no
    // gather at all: rows is the identity, so grad/hess are used in place.
    bool const      dense = n == ds.n_rows();
    OrderedGh const gh    = dense ? OrderedGh{.g = grad.data(), .h = hess.data()}
                                  : ordered_gh(split_input.rows, grad, hess);
    float const    *og    = gh.g;
    float const    *oh    = gh.h;
    parallel::for_each_index(
        selected.size(),
        [&](size_t s)
        {
            if (carve != nullptr)
            {
                split_input.hists.carve_run(*carve, selected, s);
            }
            feature_id_t const fid = selected[s];
            Histogram         &h   = split_input.hists[fid];
            ds.visit_bins(
                fid,
                [&](auto bins)
                {
                    row_id_t const *rows = split_input.rows.data();
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
                    if (oh == nullptr)
                    {
                        fill([](size_t) { return 1.0F; });
                    }
                    else
                    {
                        fill([oh](size_t k) { return oh[k]; });
                    }
                });
            if (sibling != nullptr)
            {
                (*sibling)[fid] -= h;
            }
        });
}

// Grow-only storage for the fill's partials: kept at its high-water mark for
// the whole fit, so its zero fill is paid a handful of times per process
// (docs/architecture/7-parallel.md).
std::span<HistCell> partials_storage(size_t n_cells)
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
// from the untiled fill, so models are bit-identical (#217).
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

std::vector<MirrorSlice> mirror_slices(Dataset const                &ds,
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
std::vector<size_t> selected_bins(Dataset const                &ds,
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
// once per split instead of once per level (#367).
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

PlanCache &plan_cache()
{
    static thread_local PlanCache cache;
    return cache;
}

// What the fills read for hessians: an empty view when every row's hessian is
// 1.0F, which the fills answer with the literal. Adding 1.0F is the same
// float operation the gathered value would have performed, so cell sums,
// their order, and the sibling subtraction are untouched; what goes is the
// gather and the stream the fill re-reads once per feature.
floats_view fill_hess(floats_view hess)
{
    PlanCache const &cache = plan_cache();
    return cache.unit_hess && cache.hess == hess.data() ? floats_view{} : hess;
}

SelectionPlan const &selection_plan(Dataset const                &ds,
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

// The partials are the same grid with a run-time row width: one row per
// partition slot, packed the way the slice's node arenas are.
using PartialsView = std::mdspan<HistCell, std::dextents<size_t, 2>>;

// The per-(thread, node) scratch histograms the reduce merges, plus the flag
// per slot saying whether this slice has zeroed it yet.
struct Partials
{
    PartialsView cells;
    uint8_t     *used;
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
    MirrorSlice const  &slice;
    uint8_t const      *rm;
    HistCell *const    *bases;
    feature_id_t const *selected;
    CellMode            mode;
};

// Accumulates rows [first, last) of one node into `target`, reading each row's
// bins as contiguous bytes of the slice's mirror tile and grad/hess once per
// row. NodeOrder says grad/hess are already gathered into the node's row
// order, so they are read at the row's position instead of its id. A null
// `hess` is the unit hessian: every row adds the literal 1.0F.
template <bool NodeOrder = false>
void fill_rows(SplitInput const &node, size_t first, size_t last, float const *grad,
               float const *hess, FillTarget const &target)
{
    uint8_t const *const      rm_ptr   = target.rm;
    HistCell *const *const    base_ptr = target.bases;
    feature_id_t const *const sel_ptr  = target.selected;
    size_t const              width    = target.slice.rm_width;
    size_t const              s0       = target.slice.s0;
    size_t const              fid0     = target.slice.fid0;
    size_t const              n_sel_b  = target.slice.n_selected();
    row_id_t const           *rows     = node.rows.data();
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
    ArenaView const a0{target.mode == CellMode::uniform ? base_ptr[0] : nullptr,
                       n_sel_b};
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
                        __builtin_prefetch(grad + rp, 0, 0);
                        if constexpr (!decltype(unit_hess)::value)
                        {
                            __builtin_prefetch(hess + rp, 0, 0);
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
                     { return base_ptr[s][row_bins[s]]; }, unit_hess);
        }
        else
        {
            run_rows([&](size_t s, uint8_t const *row_bins) -> HistCell &
                     { return base_ptr[s][row_bins[sel_ptr[s0 + s] - fid0]]; },
                     unit_hess);
        }
    };
    if (hess == nullptr)
    {
        by_mode(std::true_type{});
    }
    else
    {
        by_mode(std::false_type{});
    }
}

constexpr size_t no_thread = static_cast<size_t>(-1);

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
ReducePlan const &plan_reduce(split_input_refs nodes, size_t grain)
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
void run_fill_reduce(ReducePlan const &plan, split_input_refs nodes, Dataset const &ds,
                     floats_view grad, floats_view hess,
                     std::span<feature_id_t const> selected, SelectionPlan const &sp,
                     MirrorSlice const &sl)
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
    // Capture views and raw pointers: naming a thread_local inside the
    // parallel region would resolve to each worker's own (empty) container.
    PartialsView const        view{cells.data(), plan.n_slots, row_cells};
    Partials const            parts{.cells = view, .used = touched.data()};
    size_t const             *off_ptr   = sp.offsets.data();
    feature_id_t const *const sel_ptr   = selected.data();
    uint8_t const *const      rm_ptr    = ds.row_major_bins().data() + sl.rm_base;
    RowChunk const *const     chunk_ptr = plan.chunks.data();
    ReduceNode const *const   rn_ptr    = plan.nodes.data();
    std::reference_wrapper<SplitInput> const *node_ptr = nodes.data();
    // Where selected feature s starts inside one partial row.
    auto const part_cell = [&, off_ptr](size_t s)
    { return padded ? (s - sl.s0) * k_feature_stride : off_ptr[s] - sl.cell0; };
    // A fully selected tile over padded cells addresses arithmetically, with
    // no per-feature base load; anything else loads a base per feature.
    CellMode const mode = dense_sel && padded ? CellMode::uniform
                          : dense_sel         ? CellMode::dense
                                              : CellMode::gathered;
    // One index per partition slot, not per worker: buffers are keyed by the
    // index, so which worker picks it up changes nothing.
    parallel::for_each_index(
        plan.n_threads,
        [&, parts, sel_ptr, rm_ptr, chunk_ptr, rn_ptr, node_ptr](size_t t)
        {
            static thread_local std::vector<HistCell *> bases;
            bases.resize(n_sel_b);
            size_t last_node = nodes.size();
            for (size_t i = plan.begin(t); i < plan.begin(t + 1); ++i)
            {
                RowChunk const   &chunk  = chunk_ptr[i];
                ReduceNode const &rn     = rn_ptr[chunk.node];
                bool const        direct = t == rn.first_thread;
                size_t const slot = direct ? 0 : rn.slot0 + (t - rn.first_thread - 1);
                if (!direct && parts.used[slot] == 0)
                {
                    std::fill_n(&parts.cells[slot, 0], row_cells, HistCell{});
                    parts.used[slot] = 1;
                }
                if (chunk.node != last_node)
                {
                    SplitInput &node = node_ptr[chunk.node];
                    for (size_t s = sl.s0; s < sl.s1; ++s)
                    {
                        bases[s - sl.s0] = direct
                                               ? node.hists[sel_ptr[s]].cells().data()
                                               : &parts.cells[slot, part_cell(s)];
                    }
                    last_node = chunk.node;
                }
                FillTarget const target{.slice    = sl,
                                        .rm       = rm_ptr,
                                        .bases    = bases.data(),
                                        .selected = sel_ptr,
                                        .mode     = mode};
                fill_rows(node_ptr[chunk.node], chunk.first, chunk.last, grad.data(),
                          hess.data(), target);
            }
        });
    size_t const *const red_ptr = plan.reduce_nodes.data();
    parallel::for_each_index(
        plan.reduce_nodes.size() * n_sel_b,
        [&, parts, sel_ptr, rn_ptr, node_ptr, red_ptr](size_t j)
        {
            size_t const      ni   = red_ptr[j / n_sel_b];
            ReduceNode const &rn   = rn_ptr[ni];
            size_t const      s    = sl.s0 + (j % n_sel_b);
            size_t const      cell = part_cell(s);
            Histogram        &h    = node_ptr[ni].get().hists[sel_ptr[s]];
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
constexpr size_t k_reduce_grain = 1024;

// A level's fill routes each node either to the dense column fill or to this
// sparse one. A sparse level's rows are cut into row chunks, node-major, and
// the chunk list is dealt to the threads as contiguous ranges. The lowest
// thread touching a node fills straight into that node's arena; every higher
// thread fills its own partials instead. A reduce then sums a node's partials
// into its arena in ascending thread order. The histograms being filled are
// slices carved from each node's arena, one per selected feature, at a fixed
// feature stride when the data is u8.
void fill_sparse(Dataset const &ds, floats_view grad, floats_view hess,
                 split_input_refs nodes, std::span<feature_id_t const> selected,
                 SelectionPlan const &sp)
{
    ReducePlan const &plan = plan_reduce(nodes, k_reduce_grain);
    for (MirrorSlice const &sl : sp.slices)
    {
        run_fill_reduce(plan, nodes, ds, grad, hess, selected, sp, sl);
    }
}

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
void fill_lone(Dataset const &ds, SplitInput &node,
               std::span<feature_id_t const> selected, SelectionPlan const &sp,
               ArenaLayout const *carve, NodeHistograms *sibling, size_t ranges,
               size_t blocks, OrderedGh const &gh)
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
    uint8_t const *const      rm_all  = ds.row_major_bins().data();
    feature_id_t const *const sel_ptr = selected.data();
    NodeHistograms           &hists   = node.hists;
    // Capture a raw pointer: naming the thread_local inside the parallel
    // region would resolve to each worker's own (empty) vector.
    FillBlock const *const work_ptr = work.data();
    parallel::for_each_index_on(
        static_cast<int>(ranges * blocks), work.size(),
        [&, view, rm_all, sel_ptr, work_ptr](size_t t)
        {
            FillBlock const   &r      = work_ptr[t];
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
                return direct ? hists[sel_ptr[s]].cells().data()
                              : part + (s * k_feature_stride);
            };
            if (direct)
            {
                if (carve != nullptr)
                {
                    for (size_t s = sub.s0; s < sub.s1; ++s)
                    {
                        node.hists.carve_run(*carve, selected, s);
                    }
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
                                    .rm       = rm_all + sl.rm_base + r.b0,
                                    .bases    = bases.data(),
                                    .selected = sel_ptr,
                                    .mode     = dense_sel ? CellMode::uniform
                                                          : CellMode::gathered};
            fill_rows<true>(node, n * r.block / blocks, n * (r.block + 1) / blocks,
                            gh.g, gh.h, target);
            if (blocks == 1 && sibling != nullptr)
            {
                for (size_t s = sub.s0; s < sub.s1; ++s)
                {
                    (*sibling)[sel_ptr[s]] -= hists[sel_ptr[s]];
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
        [&, view, sel_ptr](size_t s)
        {
            Histogram &h = hists[sel_ptr[s]];
            for (size_t j = 1; j < blocks; ++j)
            {
                h.add_cells({&view[j - 1, s * k_feature_stride], h.size()});
            }
            if (sibling != nullptr)
            {
                (*sibling)[sel_ptr[s]] -= h;
            }
        });
}

// Cache line the mirror is read in: a feature range narrower than one line
// shares its lines with its neighbours, so each of those workers refetches
// what the others already pulled.
constexpr size_t k_line_bytes = 64;

// Cells one worker must own for a fill to spread at all: below it the region
// entry costs more than the work it splits (measured, #367). A fill's cells
// are the arena it zeroes plus the accumulates it makes, so a wide node is
// never small however few rows it holds.
constexpr size_t k_fill_cells_per_worker = 1 << 15;

// Rows one row block must own to earn its partial arena, which is zeroed and
// reduced whatever the block fills (measured, #367).
constexpr size_t k_fill_rows_per_block = 1 << 13;

// How a lone node's fill decomposes: feature ranges by row blocks.
struct FillPlan
{
    size_t ranges;
    size_t blocks;
};

// The decomposition rule: size the team to the work, spend it on row blocks
// while each one earns its partial and the mirror row is too narrow to give
// every worker its own cache line, and on feature ranges after that.
FillPlan plan_lone_fill(size_t n_rows, SelectionPlan const &sp, size_t threads)
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

// Density cutoff: a node holding rows >= n_rows / den routes to the column
// fill, sparser ones to the row-chunk fill; measured, not tunable
// (docs/architecture/7-parallel.md).
constexpr size_t k_col_fill_den = 4;

} // namespace

// One node's fill. The leaf plane calls this per split, and its node has no
// frontier to spread the work over, so the fill cuts the node itself: a sparse
// one into feature ranges by row blocks over the mirror, a dense or u16 one
// column by column. The carve and the sibling subtraction ride the same cut,
// so a split's whole histogram step costs one parallel region, two when row
// blocks leave partials to reduce. The root keeps the level growers' path: it
// is the one lone node they populate too.
bool CpuHistogramEngine::populate(Dataset const &ds, floats_view grad, floats_view hess,
                                  SplitInput                   &split_input,
                                  std::span<feature_id_t const> selected,
                                  NodeHistograms               *sibling)
{
    hess = fill_hess(hess);
    if (split_input.id != 0 && !selected.empty() && !split_input.rows.empty())
    {
        SelectionPlan const &sp = selection_plan(ds, selected);
        split_input.hists.carve_storage(sp.layout, ds.n_features());
        if (ds.bins_are_u8() && split_input.rows.size() * k_col_fill_den < ds.n_rows())
        {
            // Decomposed to the node and the mirror row's width.
            FillPlan const plan =
                plan_lone_fill(split_input.rows.size(), sp,
                               static_cast<size_t>(parallel::n_threads()));
            fill_lone(ds, split_input, selected, sp, &sp.layout, sibling, plan.ranges,
                      plan.blocks, ordered_gh(split_input.rows, grad, hess));
        }
        else
        {
            fill_feature_parallel(ds, grad, hess, split_input, selected, &sp.layout,
                                  sibling);
        }
        // The carve rides the fill's decomposition, so a work list that missed
        // a run would read cells whose lifetime never started.
        assert(split_input.hists.all_runs_carved(sp.layout, selected));
        return true;
    }
    std::array one = {std::ref(split_input)};
    populate_many(ds, grad, hess, one, selected);
    return false;
}

// Builds histograms for `selected` features only; unselected slots stay
// zero-binned placeholders the split finders skip. u8 data fills row-wise
// over the row-major mirror — cache-friendly at any node sparsity, with
// sums reproducible at a fixed thread count; single-chunk nodes and the u16
// feature-parallel path stay bit-identical at any thread count
// (docs/architecture/7-parallel.md).
void CpuHistogramEngine::populate_many(Dataset const &ds, floats_view grad,
                                       floats_view hess, split_input_refs nodes,
                                       std::span<feature_id_t const> selected)
{
    hess = fill_hess(hess);
    // One layout for the tree: every node's arena packs the same way.
    SelectionPlan const &sp = selection_plan(ds, selected);
    // Placeholder construction is one arena allocation plus a ~256KB zero
    // fill per node; run serially it is a serial fraction that caps the level
    // fill near half its thread efficiency. Nodes are independent, so
    // workers build them concurrently; the pool's mutex serializes only the
    // per-node arena take. A lone node (the leafwise grower's every fill) has
    // no such spread and carves its arena across workers instead.
    bool const alone = nodes.size() == 1;
    parallel::for_each_index(
        nodes.size(), [&](size_t i)
        { nodes[i].get().hists.carve(sp.layout, selected, ds.n_features(), alone); });
    if (selected.empty())
    {
        return;
    }
    if (!ds.bins_are_u8())
    {
        for (SplitInput &node : nodes)
        {
            fill_feature_parallel(ds, grad, hess, node, selected);
        }
        return;
    }
    static thread_local std::vector<std::reference_wrapper<SplitInput>> sparse_nodes;
    sparse_nodes.clear();
    for (SplitInput &node : nodes)
    {
        if (node.rows.size() * k_col_fill_den >= ds.n_rows())
        {
            fill_feature_parallel(ds, grad, hess, node, selected);
        }
        else
        {
            sparse_nodes.emplace_back(node);
        }
    }
    if (!sparse_nodes.empty())
    {
        fill_sparse(ds, grad, hess, sparse_nodes, selected, sp);
    }
}

// A fresh tree may draw a different selection into the same buffer, so the
// cached plan cannot outlive the tree that built it. The unit-hessian test
// is a property of this tree's hessians, not of the objective: one pass over
// the array per tree, against a fill that walks it once per feature.
void CpuHistogramEngine::begin_tree(Dataset const & /*ds*/, floats_view /*grad*/,
                                    floats_view hess)
{
    bool const unit =
        !hess.empty() && std::ranges::all_of(hess, [](float h) { return h == 1.0F; });
    plan_cache()           = {};
    plan_cache().hess      = hess.data();
    plan_cache().unit_hess = unit;
}

template class DepthwiseGrower<CpuHistogramEngine, HistogramNodeSplitFinder>;
template class ObliviousGrower<CpuHistogramEngine, HistogramLevelSplitFinder>;
template class LeafwiseGrower<CpuHistogramEngine, HistogramNodeSplitFinder>;

} // namespace bonsai
