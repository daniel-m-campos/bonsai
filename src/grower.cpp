#include "bonsai/grower.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/parallel.hpp"
#include "bonsai/split.hpp"
#include "bonsai/types.hpp"
#include "grower_impl.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <print>
#include <span>
#include <vector>

namespace bonsai
{

namespace
{

// Feature-parallel fill for u16 (high max_bin) data: one thread owns one
// feature's histogram and fills it in row order, so results are
// bit-identical at any thread count. visit_bins monomorphizes the fill per
// bin width.
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

constexpr size_t direct_fill = static_cast<size_t>(-1);

// One row segment of one node, owned by one stripe: fills either the node's
// own histogram cells (single-stripe nodes) or a private partial slab merged
// afterwards.
struct FillUnit
{
    std::reference_wrapper<SplitInput> node;
    size_t                             k0, k1;
    size_t   partial_off; // cell offset into the partials slab, or direct_fill
    uint32_t stripe = 0;
};

struct MergeJob
{
    std::reference_wrapper<SplitInput> node;
    size_t                             partial_off;
    size_t                             n_blocks;
};

// A level's fill schedule plus the partial-slab size its boundary nodes
// need.
struct FillPlan
{
    std::vector<FillUnit> units;
    std::vector<MergeJob> merges;
    size_t                partial_cells = 0;
    size_t                n_stripes     = 1;
};

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

// Cuts the batch's concatenated rows into one contiguous stripe per thread,
// row-weighted so the stripes carry equal work regardless of node-size skew.
// A node whose rows sit inside one stripe fills its cells directly (no
// partials, no zeroing, no merge); only nodes crossing a stripe boundary pay
// a partial per stripe touched, merged in stripe order. Stripe boundaries
// depend on the batch and the configured thread count only, never
// scheduling (docs/architecture/7-parallel.md).
FillPlan const &plan_fill(split_input_refs nodes, size_t n_sel, size_t total_sel_bins)
{
    static thread_local FillPlan plan;
    plan.units.clear();
    plan.merges.clear();
    plan.partial_cells = 0;
    size_t total_rows  = 0;
    for (SplitInput const &node : nodes)
    {
        total_rows += node.rows.size();
    }
    if (total_rows == 0)
    {
        plan.n_stripes = 1;
        return plan;
    }
    // Stripe count scales with the batch's fill work so a stripe's work
    // dwarfs its partial's zero+merge cost (>= ~16x); small batches stay
    // single-stripe and keep the serial order bit-identically.
    plan.n_stripes  = std::clamp(total_rows * n_sel / (16 * total_sel_bins), size_t{1},
                                 static_cast<size_t>(parallel::n_threads()));
    size_t const nt = plan.n_stripes;
    auto         stripe_of_row = [&](size_t g)
    { return std::min(nt - 1, g * nt / total_rows); };
    size_t offset = 0;
    for (SplitInput &node : nodes)
    {
        size_t const n = node.rows.size();
        if (n == 0)
        {
            continue;
        }
        size_t const s_first = stripe_of_row(offset);
        size_t const s_last  = stripe_of_row(offset + n - 1);
        if (s_first == s_last)
        {
            plan.units.push_back(
                {node, 0, n, direct_fill, static_cast<uint32_t>(s_first)});
            offset += n;
            continue;
        }
        size_t const first_off = plan.partial_cells;
        size_t       blocks    = 0;
        for (size_t s = s_first; s <= s_last; ++s)
        {
            size_t const g0 = std::max(offset, s * total_rows / nt);
            size_t const g1 = std::min(
                offset + n, s + 1 == nt ? total_rows : (s + 1) * total_rows / nt);
            if (g1 <= g0)
            {
                continue;
            }
            plan.units.push_back({node, g0 - offset, g1 - offset, plan.partial_cells,
                                  static_cast<uint32_t>(s)});
            plan.partial_cells += total_sel_bins;
            ++blocks;
        }
        plan.merges.push_back({node, first_off, blocks});
        offset += n;
    }
    return plan;
}

// Uninitialized slab: fill units zero their own partials, so pages are
// first-touched by the worker that accumulates into them — a vector's
// main-thread value-init would home every page on one NUMA node and make
// remote workers RMW across the interconnect.
std::span<HistCell> partials_slab(size_t n_cells)
{
    // NOLINTBEGIN(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    static thread_local std::unique_ptr<HistCell[]> slab;
    static thread_local size_t                      cap = 0;
    if (n_cells > cap)
    {
        slab = std::make_unique_for_overwrite<HistCell[]>(n_cells);
        cap  = n_cells;
    }
    // NOLINTEND(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
    return {slab.get(), n_cells};
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

// Stripe start of a unit's partials compacted to this slice: plan offsets
// are exact multiples of the FULL selection footprint, so the division is
// lossless.
size_t stripe_of(size_t partial_off, size_t total_cells, MirrorSlice const &sl)
{
    return partial_off / total_cells * sl.cells;
}

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

// Runs the plan's units over one mirror slice in one parallel section: each
// accumulates its row block, reading the row's bins as one contiguous strip
// of the slice's mirror block and grad/hess once per row.
void run_fill(FillPlan const &plan, Dataset const &ds, floats_view grad,
              floats_view hess, std::span<feature_id_t const> selected,
              SelectedOffsets const &offsets, std::span<HistCell> partials,
              MirrorSlice const &sl)
{
    HistCell *const parts = partials.data();
    // Capture raw pointers: naming a thread_local inside the parallel region
    // would resolve to each worker's own (empty) container.
    size_t const             *off_ptr   = offsets.cells.data();
    feature_id_t const *const sel_ptr   = selected.data();
    uint8_t const *const      rm_ptr    = ds.row_major_bins().data() + sl.rm_base;
    size_t const              width     = sl.rm_width;
    size_t const              fid0      = sl.fid0;
    size_t const              total     = offsets.total_cells;
    FillUnit const *const     units_ptr = plan.units.data();
    parallel::for_each_index(
        plan.n_stripes,
        [&, parts, off_ptr, sel_ptr, rm_ptr, units_ptr](size_t sid)
        {
            for (size_t u = 0; u < plan.units.size(); ++u)
            {
                FillUnit const &unit = units_ptr[u];
                if (unit.stripe != sid)
                {
                    continue;
                }
                size_t const slab = unit.partial_off == direct_fill
                                        ? 0
                                        : stripe_of(unit.partial_off, total, sl);
                if (unit.partial_off != direct_fill)
                {
                    std::fill_n(parts + slab, sl.cells, HistCell{});
                }
                static thread_local std::vector<HistCell *> bases;
                bases.resize(sl.n_selected());
                for (size_t s = sl.s0; s < sl.s1; ++s)
                {
                    bases[s - sl.s0] =
                        unit.partial_off == direct_fill
                            ? unit.node.get().hists[sel_ptr[s]].cells().data()
                            : parts + slab + (off_ptr[s] - sl.cell0);
                }
                HistCell **const base_ptr = bases.data();
                row_id_t const  *rows     = unit.node.get().rows.data();
                // Software prefetch: below the root, a node's rows are an
                // ascending SUBSET, so successive mirror strips sit at
                // irregular strides the hardware prefetcher cannot follow —
                // the populate ledger showed the row loop DRAM-latency-bound
                // at depth. Pull the strip and the grad/hess pair a fixed
                // distance ahead; the tail rows run in a branch-free loop of
                // their own instead of testing bounds every iteration. Reads
                // only, so results are bit-identical.
                constexpr size_t k_ahead  = 16;
                size_t const     n_sel_b  = sl.n_selected();
                auto             fill_row = [&](size_t k)
                {
                    size_t const         r  = rows[k];
                    uint8_t const *const rb = rm_ptr + (r * width);
                    float const          g  = grad[r];
                    float const          h  = hess[r];
                    for (size_t s = 0; s < n_sel_b; ++s)
                    {
                        HistCell &c = base_ptr[s][rb[sel_ptr[sl.s0 + s] - fid0]];
                        c.sum_grad += g;
                        c.sum_hess += h;
                    }
                };
                size_t const main_end =
                    unit.k1 - unit.k0 > k_ahead ? unit.k1 - k_ahead : unit.k0;
                for (size_t k = unit.k0; k < main_end; ++k)
                {
                    size_t const rp = rows[k + k_ahead];
                    __builtin_prefetch(rm_ptr + (rp * width), 0, 0);
                    __builtin_prefetch(rm_ptr + (rp * width) + 64, 0, 0);
                    __builtin_prefetch(&grad[rp], 0, 0);
                    __builtin_prefetch(&hess[rp], 0, 0);
                    fill_row(k);
                }
                for (size_t k = main_end; k < unit.k1; ++k)
                {
                    fill_row(k);
                }
            }
        });
}

// Adds multi-block nodes' partials into their histograms in ascending block
// order, one (node, feature) pair per index, for one mirror slice.
void merge_partials(FillPlan const &plan, std::span<feature_id_t const> selected,
                    SelectedOffsets const &offsets, std::span<HistCell const> partials,
                    MirrorSlice const &sl)
{
    HistCell const *const     parts     = partials.data();
    size_t const              n_sel_b   = sl.n_selected();
    size_t const              total     = offsets.total_cells;
    size_t const             *off_ptr   = offsets.cells.data();
    feature_id_t const *const sel_ptr   = selected.data();
    MergeJob const *const     merge_ptr = plan.merges.data();
    parallel::for_each_index(
        plan.merges.size() * n_sel_b,
        [&, parts, off_ptr, sel_ptr, merge_ptr](size_t ms)
        {
            MergeJob const &m      = merge_ptr[ms / n_sel_b];
            size_t const    s      = sl.s0 + (ms % n_sel_b);
            size_t const    stripe = stripe_of(m.partial_off, total, sl);
            Histogram      &h      = m.node.get().hists[sel_ptr[s]];
            for (size_t b = 0; b < m.n_blocks; ++b)
            {
                h.add_cells({parts + stripe + (b * sl.cells) + (off_ptr[s] - sl.cell0),
                             h.size()});
            }
        });
}

// Dense nodes route to the column fill: per-feature sequential scans with an
// L1-resident 2KB target, no partials and no merge, bit-identical at any
// thread count. Sparse nodes keep the row-wise units, whose 128B strips
// amortize the fetch at any sparsity. The denominator sets the density
// cutoff (rows >= n/den); the env override exists for the admission A/B and
// dies with it.
size_t col_fill_den()
{
    static size_t const v = []
    {
        char const *e = std::getenv("BONSAI_HIST_COLFILL_DEN");
        return e != nullptr ? static_cast<size_t>(std::atoi(e)) : size_t{4};
    }();
    return v;
}

void emplace_placeholders(Dataset const &ds, SplitInput &node,
                          std::span<feature_id_t const> selected)
{
    node.hists.reserve(ds.n_features());
    size_t j = 0;
    for (feature_id_t fid = 0; fid < ds.n_features(); ++fid)
    {
        bool const sel = j < selected.size() && selected[j] == fid;
        node.hists.emplace_back(sel ? ds.n_bins(fid) : 0);
        j += sel ? 1 : 0;
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
    for (SplitInput &node : nodes)
    {
        emplace_placeholders(ds, node, selected);
    }
    if (selected.empty())
    {
        return;
    }
    auto &prof = grower_detail::GrowProfiler::instance();
    // Provable strategy engagement: an equality A/B cannot detect an inert
    // toggle, so the session asserts this line per arm.
    static bool const announced = [&prof]
    {
        if (prof.enabled)
        {
            std::println(stderr, "hist-fill: colfill_den={}", col_fill_den());
        }
        return true;
    }();
    (void) announced;
    for (SplitInput const &node : nodes)
    {
        prof.populate_adds += static_cast<double>(node.rows.size()) *
                              static_cast<double>(selected.size());
    }
    if (!ds.bins_are_u8())
    {
        for (SplitInput &node : nodes)
        {
            fill_feature_parallel(ds, grad, hess, node, selected);
        }
        return;
    }
    grower_detail::GrowProfiler::Lap row_lap;
    size_t const                     den = col_fill_den();
    static thread_local std::vector<std::reference_wrapper<SplitInput>> sparse_nodes;
    sparse_nodes.clear();
    for (SplitInput &node : nodes)
    {
        if (den != 0 && node.rows.size() * den >= ds.n_rows())
        {
            fill_feature_parallel(ds, grad, hess, node, selected);
        }
        else
        {
            sparse_nodes.push_back(node);
        }
    }
    if (!sparse_nodes.empty())
    {
        SelectedOffsets const offsets = selected_offsets(ds, selected);
        // Tiles outer, rows inner: one fill pass per mirror block keeps the
        // live scatter target at one block's histograms (cache-resident by
        // construction) at any selection width, while reads stay sequential
        // inside each block. One block at narrow widths degenerates to the
        // classic single-pass fill. Measured against the per-width strategy
        // pair it replaced in benchmarks/wide-cpu-hist-2026-07.md (issue
        // #217).
        FillPlan const &plan =
            plan_fill(sparse_nodes, selected.size(), offsets.total_cells);
        size_t const n_partial_groups = plan.partial_cells / offsets.total_cells;
        for (MirrorSlice const &sl : mirror_slices(ds, selected, offsets))
        {
            std::span<HistCell> const partials =
                partials_slab(n_partial_groups * sl.cells);
            run_fill(plan, ds, grad, hess, selected, offsets, partials, sl);
            merge_partials(plan, selected, offsets, partials, sl);
        }
    }
    row_lap(prof.populate_row_s);
}

template class DepthwiseGrower<CpuHistogramEngine, HistogramNodeSplitFinder>;
template class ObliviousGrower<CpuHistogramEngine, HistogramLevelSplitFinder>;
template class LeafwiseGrower<CpuHistogramEngine, HistogramNodeSplitFinder>;

} // namespace bonsai
