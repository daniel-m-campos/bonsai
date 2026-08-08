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
#include <ranges>
#include <span>
#include <type_traits>
#include <vector>

namespace bonsai
{

namespace
{

// The column fill, taken by u16 (high max_bin) data and by dense u8 nodes:
// one thread owns one feature's histogram and fills it in row order, so
// results are bit-identical at any thread count. visit_bins monomorphizes
// the fill per bin width.
void fill_feature_parallel(Dataset const &ds, floats_view grad, floats_view hess,
                           SplitInput                   &split_input,
                           std::span<feature_id_t const> selected)
{
    // Gather grad/hess into node-row order once, so every feature's scan
    // below reads them sequentially instead of re-walking the full arrays
    // with scattered indices (n_features x full-array traffic otherwise).
    // A node covering every row (the root, absent row sampling) needs no
    // gather at all: rows is the identity, so grad/hess are used in place.
    static thread_local std::vector<float> ordered_grad;
    static thread_local std::vector<float> ordered_hess;
    size_t const                           n     = split_input.rows.size();
    bool const                             dense = n == ds.n_rows();
    float const                           *og    = grad.data();
    float const                           *oh    = hess.data();
    if (!dense)
    {
        ordered_grad.resize(n);
        ordered_hess.resize(n);
        // Capture raw pointers: naming the thread_local inside the parallel
        // region would resolve to each worker's own (empty) vector.
        float *const g = ordered_grad.data();
        float *const h = ordered_hess.data();
        parallel::for_each_index(n,
                                 [&, g, h](size_t k)
                                 {
                                     row_id_t const r = split_input.rows[k];
                                     g[k]             = grad[r];
                                     h[k]             = hess[r];
                                 });
        og = g;
        oh = h;
    }
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

// Cell offset of each selected feature inside one row of partials.
struct SelectedOffsets
{
    std::span<size_t const> cells;
    size_t                  total_cells = 0;
};

SelectedOffsets selected_offsets(Dataset const                &ds,
                                 std::span<feature_id_t const> selected)
{
    static thread_local std::vector<size_t> offsets;
    offsets.resize(selected.size());
    size_t total = 0;
    for (size_t s = 0; s < selected.size(); ++s)
    {
        offsets[s] = total;
        total += ds.n_bins(selected[s]);
    }
    return {.cells = offsets, .total_cells = total};
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

std::span<MirrorSlice const> mirror_slices(Dataset const                &ds,
                                           std::span<feature_id_t const> selected,
                                           SelectedOffsets const        &offsets)
{
    static thread_local std::vector<MirrorSlice> slices;
    slices.clear();
    size_t const width = Dataset::mirror_tile_width();
    size_t const f     = ds.n_features();
    size_t       s     = 0;
    while (s < selected.size())
    {
        size_t const tile       = selected[s] / width;
        size_t const tile_width = std::min(width, f - (tile * width));
        size_t       e          = s;
        while (e < selected.size() && selected[e] / width == tile)
        {
            ++e;
        }
        size_t const cell_end =
            e < selected.size() ? offsets.cells[e] : offsets.total_cells;
        slices.push_back({.s0       = s,
                          .s1       = e,
                          .rm_base  = ds.n_rows() * tile * width,
                          .rm_width = tile_width,
                          .cell0    = offsets.cells[s],
                          .cells    = cell_end - offsets.cells[s],
                          .fid0     = tile * width});
        s = e;
    }
    return slices;
}

// The partials are the same grid with a run-time row width: one row per
// partition slot, packed to the slice's cells.
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
// row.
void fill_rows(SplitInput const &node, size_t first, size_t last, floats_view grad,
               floats_view hess, FillTarget const &target)
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
                    __builtin_prefetch(&grad[rp], 0, 0);
                    __builtin_prefetch(&hess[rp], 0, 0);
                }
                size_t const         r        = rows[k];
                uint8_t const *const row_bins = rm_ptr + (r * width);
                float const          g        = grad[r];
                float const          h        = hess[r];
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
                     std::span<feature_id_t const> selected,
                     SelectedOffsets const &offsets, MirrorSlice const &sl)
{
    static thread_local std::vector<uint8_t> touched;
    std::span<HistCell> const cells = partials_storage(plan.n_slots * sl.cells);
    touched.assign(plan.n_slots, 0);
    // Capture views and raw pointers: naming a thread_local inside the
    // parallel region would resolve to each worker's own (empty) container.
    PartialsView const        view{cells.data(), plan.n_slots, sl.cells};
    Partials const            parts{.cells = view, .used = touched.data()};
    size_t const             *off_ptr   = offsets.cells.data();
    feature_id_t const *const sel_ptr   = selected.data();
    uint8_t const *const      rm_ptr    = ds.row_major_bins().data() + sl.rm_base;
    RowChunk const *const     chunk_ptr = plan.chunks.data();
    ReduceNode const *const   rn_ptr    = plan.nodes.data();
    std::reference_wrapper<SplitInput> const *node_ptr  = nodes.data();
    size_t const                              n_sel_b   = sl.n_selected();
    bool const                                dense_sel = n_sel_b == sl.rm_width;
    CellMode const partial_mode = dense_sel ? CellMode::dense : CellMode::gathered;
    // Only a direct fill addresses a node's padded u8 arena arithmetically;
    // partials are packed to the slice's cells.
    CellMode const direct_mode =
        dense_sel && ds.bins_are_u8() ? CellMode::uniform : partial_mode;
    // One index per partition slot, not per worker: buffers are keyed by the
    // index, so which worker picks it up changes nothing.
    parallel::for_each_index(
        plan.n_threads,
        [&, parts, off_ptr, sel_ptr, rm_ptr, chunk_ptr, rn_ptr, node_ptr](size_t t)
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
                    std::fill_n(&parts.cells[slot, 0], sl.cells, HistCell{});
                    parts.used[slot] = 1;
                }
                if (chunk.node != last_node)
                {
                    SplitInput &node = node_ptr[chunk.node];
                    for (size_t s = sl.s0; s < sl.s1; ++s)
                    {
                        bases[s - sl.s0] =
                            direct ? node.hists[sel_ptr[s]].cells().data()
                                   : &parts.cells[slot, off_ptr[s] - sl.cell0];
                    }
                    last_node = chunk.node;
                }
                FillTarget const target{.slice    = sl,
                                        .rm       = rm_ptr,
                                        .bases    = bases.data(),
                                        .selected = sel_ptr,
                                        .mode = direct ? direct_mode : partial_mode};
                fill_rows(node_ptr[chunk.node], chunk.first, chunk.last, grad, hess,
                          target);
            }
        });
    size_t const *const red_ptr = plan.reduce_nodes.data();
    parallel::for_each_index(
        plan.reduce_nodes.size() * n_sel_b,
        [&, parts, off_ptr, sel_ptr, rn_ptr, node_ptr, red_ptr](size_t j)
        {
            size_t const      ni   = red_ptr[j / n_sel_b];
            ReduceNode const &rn   = rn_ptr[ni];
            size_t const      s    = sl.s0 + (j % n_sel_b);
            size_t const      cell = off_ptr[s] - sl.cell0;
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
                 split_input_refs nodes, std::span<feature_id_t const> selected)
{
    SelectedOffsets const offsets = selected_offsets(ds, selected);
    ReducePlan const     &plan    = plan_reduce(nodes, k_reduce_grain);
    for (MirrorSlice const &sl : mirror_slices(ds, selected, offsets))
    {
        run_fill_reduce(plan, nodes, ds, grad, hess, selected, offsets, sl);
    }
}

// Density cutoff: a node holding rows >= n_rows / den routes to the column
// fill, sparser ones to the row-chunk fill; measured, not tunable
// (docs/architecture/7-parallel.md).
constexpr size_t k_col_fill_den = 4;

// The selected features' bin counts, in selection order: all the arena
// layout needs from the dataset.
std::vector<size_t> selected_bins(Dataset const                &ds,
                                  std::span<feature_id_t const> selected)
{
    return selected |
           std::views::transform([&](feature_id_t fid) { return ds.n_bins(fid); }) |
           std::ranges::to<std::vector<size_t>>();
}

} // namespace

void CpuHistogramEngine::populate(Dataset const &ds, floats_view grad, floats_view hess,
                                  SplitInput                   &split_input,
                                  std::span<feature_id_t const> selected)
{
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
    // One layout for the level: every node's arena packs the same way.
    std::vector<size_t> const bins = selected_bins(ds, selected);
    ArenaLayout const         layout{bins, ds.bins_are_u8()};
    // Placeholder construction is one arena allocation plus a ~256KB zero
    // fill per node; run serially it is a serial fraction that caps the level
    // fill near half its thread efficiency. Nodes are independent, so
    // workers build them concurrently; the pool's mutex serializes only the
    // per-node arena take.
    parallel::for_each_index(
        nodes.size(), [&](size_t i)
        { nodes[i].get().hists.carve(layout, selected, ds.n_features()); });
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
        fill_sparse(ds, grad, hess, sparse_nodes, selected);
    }
}

template class DepthwiseGrower<CpuHistogramEngine, HistogramNodeSplitFinder>;
template class ObliviousGrower<CpuHistogramEngine, HistogramLevelSplitFinder>;
template class LeafwiseGrower<CpuHistogramEngine, HistogramNodeSplitFinder>;

} // namespace bonsai
