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
    size_t const                           n = rows.size();
    ordered_grad.resize(n);
    ordered_hess.resize(n);
    // Capture raw pointers: naming the thread_local inside the parallel
    // region would resolve to each worker's own (empty) vector.
    float *const          g     = ordered_grad.data();
    float *const          h     = ordered_hess.data();
    row_id_t const *const r_ptr = rows.data();
    if (n < k_gather_region_rows)
    {
        for (size_t k = 0; k < n; ++k)
        {
            g[k] = grad[r_ptr[k]];
            h[k] = hess[r_ptr[k]];
        }
        return {.g = g, .h = h};
    }
    parallel::for_each_index(n,
                             [&, g, h, r_ptr](size_t k)
                             {
                                 g[k] = grad[r_ptr[k]];
                                 h[k] = hess[r_ptr[k]];
                             });
    return {.g = g, .h = h};
}

// The column fill, taken by u16 (high max_bin) data and by dense u8 nodes:
// one thread owns one feature's histogram and fills it in row order, so
// results are bit-identical at any thread count. visit_bins monomorphizes
// the fill per bin width.
void fill_feature_parallel(Dataset const &ds, floats_view grad, floats_view hess,
                           SplitInput                   &split_input,
                           std::span<feature_id_t const> selected)
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
            feature_id_t const fid = selected[s];
            Histogram         &h   = split_input.hists[fid];
            ds.visit_bins(fid,
                          [&](auto bins)
                          {
                              if (dense)
                              {
                                  for (size_t k = 0; k < n; ++k)
                                  {
                                      h.add(bins[k], og[k], oh[k]);
                                  }
                              }
                              else
                              {
                                  row_id_t const *rows = split_input.rows.data();
                                  for (size_t k = 0; k < n; ++k)
                                  {
                                      h.add(bins[rows[k]], og[k], oh[k]);
                                  }
                              }
                          });
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

// The tree's plan, held across its fills. Keyed by dataset and selection so a
// stale plan can never be read; begin_tree drops it, since a fresh tree may
// draw a different selection into the same buffer.
struct PlanCache
{
    std::optional<SelectionPlan> plan;
    Dataset const               *ds  = nullptr;
    feature_id_t const          *sel = nullptr;
    size_t                       n   = 0;
};

PlanCache &plan_cache()
{
    static thread_local PlanCache cache;
    return cache;
}

SelectionPlan const &selection_plan(Dataset const                &ds,
                                    std::span<feature_id_t const> selected)
{
    PlanCache &cache = plan_cache();
    if (!cache.plan || cache.ds != &ds || cache.sel != selected.data() ||
        cache.n != selected.size())
    {
        cache.plan.emplace(ds, selected);
        cache.ds  = &ds;
        cache.sel = selected.data();
        cache.n   = selected.size();
    }
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
// order, so they are read at the row's position instead of its id.
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
    auto run_rows = [&](auto cell_at)
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
                        __builtin_prefetch(hess + rp, 0, 0);
                    }
                }
                size_t const         r        = rows[k];
                uint8_t const *const row_bins = rm_ptr + (r * width);
                size_t const         gk       = NodeOrder ? k : r;
                float const          g        = grad[gk];
                float const          h        = hess[gk];
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
    if (target.mode == CellMode::uniform)
    {
        run_rows([&](size_t s, uint8_t const *row_bins) -> HistCell &
                 { return a0[s, row_bins[s]]; });
    }
    else if (target.mode == CellMode::dense)
    {
        run_rows([&](size_t s, uint8_t const *row_bins) -> HistCell &
                 { return base_ptr[s][row_bins[s]]; });
    }
    else
    {
        run_rows([&](size_t s, uint8_t const *row_bins) -> HistCell &
                 { return base_ptr[s][row_bins[sel_ptr[s0 + s] - fid0]]; });
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
// one mirror tile, contiguous so the arena cells the worker touches are its
// own.
struct FeatureRange
{
    size_t slice, b0, b1;
};

// A lone node's fill, cut by feature: each worker walks the whole node over
// one contiguous range of a mirror tile and accumulates straight into that
// range of the node's arena. Nothing is shared, so no partial is zeroed and
// none is reduced, and each cell takes its rows in ascending order whatever
// the thread count. The row-chunk fill the level plane uses stays as it is:
// there a worker owns whole nodes, and the partials it does write are the
// price of spreading a frontier.
void fill_one_node(Dataset const &ds, floats_view grad, floats_view hess,
                   SplitInput &node, std::span<feature_id_t const> selected,
                   SelectionPlan const &sp)
{
    size_t const n = node.rows.size();
    if (n == 0)
    {
        return;
    }
    OrderedGh const gh = ordered_gh(node.rows, grad, hess);
    static thread_local std::vector<FeatureRange> ranges;
    ranges.clear();
    // A range walks the whole node, so equal feature counts are equal work:
    // one range per worker, cut inside the mirror tiles it spans.
    auto const   workers = static_cast<size_t>(parallel::n_threads());
    size_t const width = std::max<size_t>(1, (selected.size() + workers - 1) / workers);
    for (size_t i = 0; i < sp.slices.size(); ++i)
    {
        size_t const n_sel_b = sp.slices[i].n_selected();
        for (size_t b = 0; b < n_sel_b; b += width)
        {
            ranges.push_back({.slice = i, .b0 = b, .b1 = std::min(b + width, n_sel_b)});
        }
    }
    uint8_t const *const      rm_all  = ds.row_major_bins().data();
    feature_id_t const *const sel_ptr = selected.data();
    NodeHistograms           &hists   = node.hists;
    // Capture a raw pointer: naming the thread_local inside the parallel
    // region would resolve to each worker's own (empty) vector.
    FeatureRange const *const range_ptr = ranges.data();
    parallel::for_each_index(
        ranges.size(),
        [&, rm_all, sel_ptr, range_ptr](size_t t)
        {
            FeatureRange const &r  = range_ptr[t];
            MirrorSlice const  &sl = sp.slices[r.slice];
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
            static thread_local std::vector<HistCell *> bases;
            bases.clear();
            bases.push_back(hists[sel_ptr[sub.s0]].cells().data());
            if (!dense_sel)
            {
                for (size_t s = sub.s0 + 1; s < sub.s1; ++s)
                {
                    bases.push_back(hists[sel_ptr[s]].cells().data());
                }
            }
            FillTarget const target{.slice    = sub,
                                    .rm       = rm_all + sl.rm_base + r.b0,
                                    .bases    = bases.data(),
                                    .selected = sel_ptr,
                                    .mode     = dense_sel ? CellMode::uniform
                                                          : CellMode::gathered};
            fill_rows<true>(node, 0, n, gh.g, gh.h, target);
        });
}

// Density cutoff: a node holding rows >= n_rows / den routes to the column
// fill, sparser ones to the row-chunk fill; measured, not tunable
// (docs/architecture/7-parallel.md).
constexpr size_t k_col_fill_den = 4;

} // namespace

// One node's fill. The leaf plane calls this per split, and its node is a
// sparse child with no frontier to spread the work over, so the fill is cut
// by feature instead of by row chunk (#367). Dense nodes already fill
// feature-major over the columns, and the root keeps the level growers' path:
// it is the one lone node they populate too.
void CpuHistogramEngine::populate(Dataset const &ds, floats_view grad, floats_view hess,
                                  SplitInput                   &split_input,
                                  std::span<feature_id_t const> selected)
{
    if (split_input.id != 0 && ds.bins_are_u8() && !selected.empty() &&
        split_input.rows.size() * k_col_fill_den < ds.n_rows())
    {
        SelectionPlan const &sp = selection_plan(ds, selected);
        split_input.hists.carve(sp.layout, selected, ds.n_features(), true);
        fill_one_node(ds, grad, hess, split_input, selected, sp);
        return;
    }
    std::array one = {std::ref(split_input)};
    populate_many(ds, grad, hess, one, selected);
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
// cached plan cannot outlive the tree that built it.
void CpuHistogramEngine::begin_tree(Dataset const & /*ds*/, floats_view /*grad*/,
                                    floats_view /*hess*/)
{
    plan_cache() = {};
}

template class DepthwiseGrower<CpuHistogramEngine, HistogramNodeSplitFinder>;
template class ObliviousGrower<CpuHistogramEngine, HistogramLevelSplitFinder>;
template class LeafwiseGrower<CpuHistogramEngine, HistogramNodeSplitFinder>;

} // namespace bonsai
