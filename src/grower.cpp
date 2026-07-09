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
#include <functional>
#include <memory>
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

// One row block of one node: fills either the node's own histogram cells
// (single-block nodes) or a private partial slab merged afterwards.
struct FillUnit
{
    std::reference_wrapper<SplitInput> node;
    size_t                             k0, k1;
    size_t partial_off; // cell offset into the partials slab, or direct_fill
};

struct MergeJob
{
    std::reference_wrapper<SplitInput> node;
    size_t                             partial_off;
    size_t                             n_blocks;
};

// A level's fill schedule plus the partial-slab size its multi-block nodes
// need.
struct FillPlan
{
    std::vector<FillUnit> units;
    std::vector<MergeJob> merges;
    size_t                partial_cells = 0;
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

// Splits each node into row blocks: enough that a block's fill work dwarfs
// its partial's zero+merge cost (>= ~16x), capped at 4x the thread count.
// Single-block nodes write their cells directly and need no partials; block
// counts depend on node size, selection width, and the configured thread
// count only, never scheduling (docs/architecture/7-parallel.md).
FillPlan const &plan_fill(split_input_refs nodes, size_t n_sel, size_t total_sel_bins)
{
    static thread_local FillPlan plan;
    plan.units.clear();
    plan.merges.clear();
    plan.partial_cells      = 0;
    size_t const max_blocks = 4 * static_cast<size_t>(parallel::n_threads());
    for (SplitInput &node : nodes)
    {
        size_t const n = node.rows.size();
        if (n == 0)
        {
            continue;
        }
        size_t const n_blocks =
            std::clamp(n * n_sel / (16 * total_sel_bins), size_t{1}, max_blocks);
        if (n_blocks == 1)
        {
            plan.units.push_back({node, 0, n, direct_fill});
            continue;
        }
        plan.merges.push_back({node, plan.partial_cells, n_blocks});
        for (size_t b = 0; b < n_blocks; ++b)
        {
            plan.units.push_back(
                {node, b * n / n_blocks, (b + 1) * n / n_blocks, plan.partial_cells});
            plan.partial_cells += total_sel_bins;
        }
    }
    return plan;
}

// Uninitialized slab: fill units zero their own partials, so pages are
// first-touched by the worker that accumulates into them — a vector's
// main-thread value-init would home every page on one NUMA node and make
// remote workers RMW across the interconnect.
HistCell *partials_slab(size_t n_cells)
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
    return slab.get();
}

// Runs the plan's units in one parallel section: each accumulates its row
// block, reading the row's bins as one contiguous mirror strip and grad/hess
// once per row.
void run_fill(FillPlan const &plan, Dataset const &ds, floats_view grad,
              floats_view hess, std::span<feature_id_t const> selected,
              SelectedOffsets const &offsets, HistCell *parts)
{
    size_t const n_sel      = selected.size();
    size_t const n_features = ds.n_features();
    // Capture raw pointers: naming a thread_local inside the parallel region
    // would resolve to each worker's own (empty) container.
    size_t const             *off_ptr   = offsets.cells.data();
    feature_id_t const *const sel_ptr   = selected.data();
    uint8_t const *const      rm_ptr    = ds.row_major_bins().data();
    FillUnit const *const     units_ptr = plan.units.data();
    parallel::for_each_index(
        plan.units.size(),
        [&, parts, off_ptr, sel_ptr, rm_ptr, units_ptr](size_t u)
        {
            FillUnit const &unit = units_ptr[u];
            if (unit.partial_off != direct_fill)
            {
                std::fill_n(parts + unit.partial_off, offsets.total_cells, HistCell{});
            }
            static thread_local std::vector<HistCell *> bases;
            bases.resize(n_sel);
            for (size_t s = 0; s < n_sel; ++s)
            {
                bases[s] = unit.partial_off == direct_fill
                               ? unit.node.get().hists[sel_ptr[s]].cells().data()
                               : parts + unit.partial_off + off_ptr[s];
            }
            HistCell **const base_ptr = bases.data();
            row_id_t const  *rows     = unit.node.get().rows.data();
            for (size_t k = unit.k0; k < unit.k1; ++k)
            {
                size_t const         r  = rows[k];
                uint8_t const *const rb = rm_ptr + (r * n_features);
                double const         g  = grad[r];
                double const         h  = hess[r];
                for (size_t s = 0; s < n_sel; ++s)
                {
                    HistCell &c = base_ptr[s][rb[sel_ptr[s]]];
                    c.sum_grad += g;
                    c.sum_hess += h;
                }
            }
        });
}

// Adds multi-block nodes' partials into their histograms in ascending block
// order, one (node, feature) pair per index.
void merge_partials(FillPlan const &plan, std::span<feature_id_t const> selected,
                    SelectedOffsets const &offsets, HistCell const *parts)
{
    size_t const              n_sel     = selected.size();
    size_t const             *off_ptr   = offsets.cells.data();
    feature_id_t const *const sel_ptr   = selected.data();
    MergeJob const *const     merge_ptr = plan.merges.data();
    parallel::for_each_index(
        plan.merges.size() * n_sel,
        [&, parts, off_ptr, sel_ptr, merge_ptr](size_t ms)
        {
            MergeJob const &m = merge_ptr[ms / n_sel];
            size_t const    s = ms % n_sel;
            Histogram      &h = m.node.get().hists[sel_ptr[s]];
            for (size_t b = 0; b < m.n_blocks; ++b)
            {
                h.add_cells(
                    {parts + m.partial_off + (b * offsets.total_cells) + off_ptr[s],
                     h.size()});
            }
        });
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
    SelectedOffsets const            offsets = selected_offsets(ds, selected);
    FillPlan const &plan  = plan_fill(nodes, selected.size(), offsets.total_cells);
    HistCell *const parts = partials_slab(plan.partial_cells);
    run_fill(plan, ds, grad, hess, selected, offsets, parts);
    merge_partials(plan, selected, offsets, parts);
    row_lap(prof.populate_row_s);
}

template class DepthwiseGrower<CpuHistogramEngine, HistogramNodeSplitFinder>;
template class ObliviousGrower<CpuHistogramEngine, HistogramLevelSplitFinder>;
template class LeafwiseGrower<CpuHistogramEngine, HistogramNodeSplitFinder>;

} // namespace bonsai
