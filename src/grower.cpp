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

// Cell offset of each selected feature inside one partial slab.
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

// Grow-only slab for the fill's partials: kept at its high-water mark for
// the whole fit, so its zero fill is paid a handful of times per process
// (docs/architecture/7-parallel.md).
std::span<HistCell> partials_slab(size_t n_cells)
{
    static thread_local std::vector<HistCell> slab;
    if (slab.size() < n_cells)
    {
        slab.assign(n_cells, HistCell{});
    }
    return {slab.data(), n_cells};
}

// One mirror block's slice of the selection: sel indices [s0, s1), the
// block's byte offset and strip width in the tiled mirror, and the slice's
// cell range within the full selection's offsets. A fill runs one pass per
// slice (tiles outer, rows inner), so the live scatter target is one
// block's histograms — cache-resident by construction — while every read
// stays sequential inside the block. Per-feature accumulation order is
// unchanged from the untiled fill, so models are bit-identical (#217).
struct MirrorSlice
{
    size_t s0, s1;
    size_t rm_base, rm_width;
    size_t cell0, cells;
    size_t fid0; // first feature id of the slice's mirror block

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
        size_t const mb      = selected[s] / width;
        size_t const width_b = std::min(width, f - (mb * width));
        size_t       e       = s;
        while (e < selected.size() && selected[e] / width == mb)
        {
            ++e;
        }
        size_t const cell_end =
            e < selected.size() ? offsets.cells[e] : offsets.total_cells;
        slices.push_back({.s0       = s,
                          .s1       = e,
                          .rm_base  = ds.n_rows() * mb * width,
                          .rm_width = width_b,
                          .cell0    = offsets.cells[s],
                          .cells    = cell_end - offsets.cells[s],
                          .fid0     = mb * width});
        s = e;
    }
    return slices;
}

// One arena per node backs the selected histograms as contiguous views; u8
// datasets pad every chunk to this many cells so the dense row-wise fill can
// address any cell as arena[(feature * k_u8_chunk) + bin] with no per-feature
// base load. Unselected slots stay empty views the split finders skip.
constexpr size_t k_u8_chunk = 256;

// How a row's strip byte becomes a histogram cell address.
enum class CellMode : uint8_t
{
    uniform,  // arithmetic over a node's padded u8 arena, no base load
    dense,    // one base per strip column, the whole block selected
    gathered, // column sampling: a selection lookup per add
};

// Where one row block's adds land: the slice being filled, its mirror block,
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

// Accumulates rows [k0, k1) of one node into `target`, reading the row's bins
// as one contiguous strip of the slice's mirror block and grad/hess once per
// row.
void fill_rows(SplitInput const &node, size_t k0, size_t k1, floats_view grad,
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
    // lookahead reads the node's row list rather than this block's, so a block
    // boundary costs no dead zone (docs/architecture/7-parallel.md). Reads
    // only, so results are bit-identical.
    constexpr size_t k_ahead = 16;
    // The node's last rows run unprefetched, peeled out so the hot loop
    // carries no per-row bound test.
    size_t const n_rows = node.rows.size();
    size_t const kp     = n_rows > k_ahead ? std::clamp(n_rows - k_ahead, k0, k1) : k0;
    // A fully selected block indexes the strip directly (the common case);
    // the general loop pays a selection lookup per add to support column
    // sampling. Direct fills into a u8 node compute the cell address
    // arithmetically over the node's padded arena (chunk stride 256), with no
    // per-feature base load at all.
    HistCell *const a0       = target.mode == CellMode::uniform ? base_ptr[0] : nullptr;
    auto            run_rows = [&](auto cell_at)
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
                size_t const         r  = rows[k];
                uint8_t const *const rb = rm_ptr + (r * width);
                float const          g  = grad[r];
                float const          h  = hess[r];
                for (size_t s = 0; s < n_sel_b; ++s)
                {
                    HistCell &c = cell_at(s, rb);
                    c.sum_grad += g;
                    c.sum_hess += h;
                }
            }
        };
        walk(std::true_type{}, k0, kp);
        walk(std::false_type{}, kp, k1);
    };
    if (target.mode == CellMode::uniform)
    {
        run_rows([&](size_t s, uint8_t const *rb) -> HistCell &
                 { return a0[(s * k_u8_chunk) + rb[s]]; });
    }
    else if (target.mode == CellMode::dense)
    {
        run_rows([&](size_t s, uint8_t const *rb) -> HistCell &
                 { return base_ptr[s][rb[s]]; });
    }
    else
    {
        run_rows([&](size_t s, uint8_t const *rb) -> HistCell &
                 { return base_ptr[s][rb[sel_ptr[s0 + s] - fid0]]; });
    }
}

constexpr size_t no_thread = static_cast<size_t>(-1);

// One (node, row-block) work item of the buffer-and-reduce fill.
struct Stripe
{
    size_t node;
    size_t k0, k1;
};

// A sparse node's place in the stripe partition. The lowest thread touching
// it accumulates straight into its arena; every higher thread owns one
// partial at slot0 + (thread - first_thread - 1).
struct ReduceNode
{
    size_t first_thread = no_thread;
    size_t last_thread  = no_thread;
    size_t slot0        = 0;
};

// A level's stripe partition: the flat work list, each node's thread run,
// and the nodes that need a reduce at all.
struct ReducePlan
{
    std::vector<Stripe>     stripes;
    std::vector<ReduceNode> nodes;
    std::vector<size_t>     reduce_nodes;
    size_t                  n_threads = 0;
    size_t                  n_slots   = 0;

    // Thread t owns stripes [begin(t), begin(t + 1)).
    size_t begin(size_t t) const
    {
        return t * stripes.size() / n_threads;
    }
};

// Uniform static striping: node-major fixed-grain blocks split into
// contiguous per-thread ranges, which keeps a thread's live scatter target to
// one node histogram, bounds the partials at n_threads - 1, and fixes the
// summation order (docs/architecture/7-parallel.md).
ReducePlan const &plan_reduce(split_input_refs nodes, size_t grain)
{
    static thread_local ReducePlan plan;
    plan.stripes.clear();
    plan.reduce_nodes.clear();
    plan.nodes.assign(nodes.size(), ReduceNode{});
    for (size_t i = 0; i < nodes.size(); ++i)
    {
        size_t const n = nodes[i].get().rows.size();
        if (n == 0)
        {
            continue;
        }
        size_t const n_blocks = (n + grain - 1) / grain;
        for (size_t b = 0; b < n_blocks; ++b)
        {
            plan.stripes.push_back({i, b * n / n_blocks, (b + 1) * n / n_blocks});
        }
    }
    plan.n_threads = static_cast<size_t>(parallel::n_threads());
    for (size_t t = 0; t < plan.n_threads; ++t)
    {
        for (size_t i = plan.begin(t); i < plan.begin(t + 1); ++i)
        {
            ReduceNode &rn = plan.nodes[plan.stripes[i].node];
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

// Fills every sparse node for one mirror slice through the static stripe
// partition, then sums each node's partials into its arena in ascending
// thread order. A partial is zeroed on first touch, so an untouched
// (thread, node) pair costs neither a zero fill nor a reduce.
void run_fill_reduce(ReducePlan const &plan, split_input_refs nodes, Dataset const &ds,
                     floats_view grad, floats_view hess,
                     std::span<feature_id_t const> selected,
                     SelectedOffsets const &offsets, MirrorSlice const &sl)
{
    static thread_local std::vector<uint8_t> touched;
    std::span<HistCell> const slab = partials_slab(plan.n_slots * sl.cells);
    touched.assign(plan.n_slots, 0);
    // Capture raw pointers: naming a thread_local inside the parallel region
    // would resolve to each worker's own (empty) container.
    HistCell *const           parts     = slab.data();
    uint8_t *const            used      = touched.data();
    size_t const             *off_ptr   = offsets.cells.data();
    feature_id_t const *const sel_ptr   = selected.data();
    uint8_t const *const      rm_ptr    = ds.row_major_bins().data() + sl.rm_base;
    Stripe const *const       strip_ptr = plan.stripes.data();
    ReduceNode const *const   rn_ptr    = plan.nodes.data();
    std::reference_wrapper<SplitInput> const *node_ptr  = nodes.data();
    size_t const                              n_sel_b   = sl.n_selected();
    bool const                                dense_sel = n_sel_b == sl.rm_width;
    CellMode const partial_mode = dense_sel ? CellMode::dense : CellMode::gathered;
    // Only a direct fill addresses a node's padded u8 arena arithmetically;
    // partials are packed to the slice's cells.
    CellMode const direct_mode =
        dense_sel && ds.bins_are_u8() ? CellMode::uniform : partial_mode;
    // One index per stripe-partition slot, not per worker: buffers are keyed
    // by the index, so which worker picks it up changes nothing.
    parallel::for_each_index(
        plan.n_threads,
        [&, parts, used, off_ptr, sel_ptr, rm_ptr, strip_ptr, rn_ptr,
         node_ptr](size_t t)
        {
            static thread_local std::vector<HistCell *> bases;
            bases.resize(n_sel_b);
            size_t last_node = nodes.size();
            for (size_t i = plan.begin(t); i < plan.begin(t + 1); ++i)
            {
                Stripe const     &st     = strip_ptr[i];
                ReduceNode const &rn     = rn_ptr[st.node];
                bool const        direct = t == rn.first_thread;
                size_t const slot = direct ? 0 : rn.slot0 + (t - rn.first_thread - 1);
                if (!direct && used[slot] == 0)
                {
                    std::fill_n(parts + (slot * sl.cells), sl.cells, HistCell{});
                    used[slot] = 1;
                }
                if (st.node != last_node)
                {
                    SplitInput &node = node_ptr[st.node];
                    for (size_t s = sl.s0; s < sl.s1; ++s)
                    {
                        bases[s - sl.s0] =
                            direct
                                ? node.hists[sel_ptr[s]].cells().data()
                                : parts + (slot * sl.cells) + (off_ptr[s] - sl.cell0);
                    }
                    last_node = st.node;
                }
                FillTarget const target{.slice    = sl,
                                        .rm       = rm_ptr,
                                        .bases    = bases.data(),
                                        .selected = sel_ptr,
                                        .mode = direct ? direct_mode : partial_mode};
                fill_rows(node_ptr[st.node], st.k0, st.k1, grad, hess, target);
            }
        });
    size_t const *const red_ptr = plan.reduce_nodes.data();
    parallel::for_each_index(
        plan.reduce_nodes.size() * n_sel_b,
        [&, parts, used, off_ptr, sel_ptr, rn_ptr, node_ptr, red_ptr](size_t j)
        {
            size_t const      ni   = red_ptr[j / n_sel_b];
            ReduceNode const &rn   = rn_ptr[ni];
            size_t const      s    = sl.s0 + (j % n_sel_b);
            size_t const      cell = off_ptr[s] - sl.cell0;
            Histogram        &h    = node_ptr[ni].get().hists[sel_ptr[s]];
            for (size_t t = rn.first_thread + 1; t <= rn.last_thread; ++t)
            {
                size_t const slot = rn.slot0 + (t - rn.first_thread - 1);
                if (used[slot] != 0)
                {
                    h.add_cells({parts + (slot * sl.cells) + cell, h.size()});
                }
            }
        });
}

// Row-block grain of the stripe partition; measured, not tunable
// (docs/architecture/7-parallel.md).
constexpr size_t k_reduce_grain = 1024;

// Sparse nodes fill through one static stripe partition per level, one pass
// per mirror slice.
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
// fill, sparser ones to the row-wise units; measured, not tunable
// (docs/architecture/7-parallel.md).
constexpr size_t k_col_fill_den = 4;

void emplace_placeholders(Dataset const &ds, SplitInput &node,
                          std::span<feature_id_t const> selected)
{
    bool const u8    = ds.bins_are_u8();
    size_t     total = 0;
    for (feature_id_t const fid : selected)
    {
        // A u8 chunk holds the feature's whole histogram; a wider one would
        // overlap the next feature's view of the arena.
        assert(!u8 || ds.n_bins(fid) <= k_u8_chunk);
        total += u8 ? k_u8_chunk : ds.n_bins(fid);
    }
    node.arena.assign(total, HistCell{});
    node.hists.reserve(ds.n_features());
    size_t j   = 0;
    size_t off = 0;
    for (feature_id_t fid = 0; fid < ds.n_features(); ++fid)
    {
        bool const sel = j < selected.size() && selected[j] == fid;
        if (!sel)
        {
            node.hists.emplace_back(std::span<HistCell>{});
            continue;
        }
        node.hists.emplace_back(
            std::span<HistCell>{node.arena.data() + off, ds.n_bins(fid)});
        off += u8 ? k_u8_chunk : ds.n_bins(fid);
        ++j;
    }
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
// sums reproducible at a fixed thread count; single-block nodes and the u16
// feature-parallel path stay bit-identical at any thread count
// (docs/architecture/7-parallel.md).
void CpuHistogramEngine::populate_many(Dataset const &ds, floats_view grad,
                                       floats_view hess, split_input_refs nodes,
                                       std::span<feature_id_t const> selected)
{
    // Placeholder construction is one arena allocation plus a ~256KB zero
    // fill per node; run serially it is an Amdahl chunk that caps the level
    // fill near half its thread efficiency. Nodes are independent, so
    // workers build them concurrently; the pool's mutex serializes only the
    // per-node arena take.
    parallel::for_each_index(nodes.size(), [&](size_t i)
                             { emplace_placeholders(ds, nodes[i], selected); });
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
