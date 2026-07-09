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
    SplitInput *node;
    size_t      k0, k1;
    size_t      partial_off; // cell offset into the partials slab, or direct_fill
};

struct MergeJob
{
    SplitInput *node;
    size_t      partial_off;
    size_t      n_blocks;
};

} // namespace

void CpuHistogramEngine::populate(Dataset const &ds, floats_view grad, floats_view hess,
                                  SplitInput                   &split_input,
                                  std::span<feature_id_t const> selected)
{
    std::array one = {std::ref(split_input)};
    populate_many(ds, grad, hess, one, selected);
}

// Builds histograms for `selected` features only; unselected slots stay
// zero-binned placeholders the split finders skip.
//
// u8 data fills row-wise: each work unit walks a row block of the row-major
// mirror once, reading each row's bins as one contiguous strip regardless of
// how sparse the node is (the per-feature column gather misses cache on
// nearly every access for deep nodes). Rows, grad, and hess are streamed
// once total instead of once per feature. Nodes large enough to amortize a
// merge split into up to 4x-thread-count blocks with private partial
// histograms merged in fixed block order; block counts are a function of
// node size, selection width, and the configured thread count only, so sums
// are reproducible at a fixed thread count (docs/architecture/7-parallel.md).
// Single-block nodes accumulate straight into their cells in row order —
// bit-identical to the feature-parallel fill.
void CpuHistogramEngine::populate_many(Dataset const &ds, floats_view grad,
                                       floats_view hess, split_input_refs nodes,
                                       std::span<feature_id_t const> selected)
{
    size_t const n_features = ds.n_features();
    for (SplitInput &node : nodes)
    {
        node.hists.reserve(n_features);
        size_t j = 0;
        for (feature_id_t fid = 0; fid < n_features; ++fid)
        {
            bool const sel = j < selected.size() && selected[j] == fid;
            node.hists.emplace_back(sel ? ds.n_bins(fid) : 0);
            j += sel ? 1 : 0;
        }
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

    size_t const                            n_sel = selected.size();
    static thread_local std::vector<size_t> offsets;
    offsets.resize(n_sel);
    size_t total_sel_bins = 0;
    for (size_t s = 0; s < n_sel; ++s)
    {
        offsets[s] = total_sel_bins;
        total_sel_bins += ds.n_bins(selected[s]);
    }
    auto const   n_threads  = static_cast<size_t>(parallel::n_threads());
    size_t const max_blocks = 4 * n_threads;

    static thread_local std::vector<FillUnit> units;
    static thread_local std::vector<MergeJob> merges;
    units.clear();
    merges.clear();
    size_t partial_cells = 0;
    for (SplitInput &node : nodes)
    {
        size_t const n = node.rows.size();
        if (n == 0)
        {
            continue;
        }
        // A block's fill work should dwarf its partial's zero+merge cost;
        // below that, one block fills the node's cells directly.
        size_t const n_blocks =
            std::clamp(n * n_sel / (16 * total_sel_bins), size_t{1}, max_blocks);
        if (n_blocks == 1)
        {
            units.push_back({&node, 0, n, direct_fill});
            continue;
        }
        merges.push_back({&node, partial_cells, n_blocks});
        for (size_t b = 0; b < n_blocks; ++b)
        {
            units.push_back(
                {&node, b * n / n_blocks, (b + 1) * n / n_blocks, partial_cells});
            partial_cells += total_sel_bins;
        }
    }
    static thread_local std::vector<HistCell> partials;
    partials.assign(partial_cells, HistCell{});

    // Capture raw pointers: naming a thread_local inside the parallel
    // regions would resolve to each worker's own (empty) container.
    HistCell *const           parts     = partials.data();
    size_t const             *off_ptr   = offsets.data();
    feature_id_t const *const sel_ptr   = selected.data();
    uint8_t const *const      rm_ptr    = ds.row_major_bins().data();
    FillUnit const *const     units_ptr = units.data();
    MergeJob const *const     merge_ptr = merges.data();
    parallel::for_each_index(
        units.size(),
        [&, parts, off_ptr, sel_ptr, rm_ptr, units_ptr](size_t u)
        {
            FillUnit const                             &unit = units_ptr[u];
            static thread_local std::vector<HistCell *> bases;
            bases.resize(n_sel);
            for (size_t s = 0; s < n_sel; ++s)
            {
                bases[s] = unit.partial_off == direct_fill
                               ? unit.node->hists[sel_ptr[s]].cells().data()
                               : parts + unit.partial_off + off_ptr[s];
            }
            HistCell **const base_ptr = bases.data();
            row_id_t const  *rows     = unit.node->rows.data();
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
    parallel::for_each_index(
        merges.size() * n_sel,
        [&, parts, off_ptr, sel_ptr, merge_ptr](size_t ms)
        {
            MergeJob const &m = merge_ptr[ms / n_sel];
            size_t const    s = ms % n_sel;
            Histogram      &h = m.node->hists[sel_ptr[s]];
            for (size_t b = 0; b < m.n_blocks; ++b)
            {
                h.add_cells({parts + m.partial_off + (b * total_sel_bins) + off_ptr[s],
                             h.size()});
            }
        });
    row_lap(prof.populate_row_s);
}

template class DepthwiseGrower<CpuHistogramEngine, HistogramNodeSplitFinder>;
template class ObliviousGrower<CpuHistogramEngine, HistogramLevelSplitFinder>;
template class LeafwiseGrower<CpuHistogramEngine, HistogramNodeSplitFinder>;

} // namespace bonsai
