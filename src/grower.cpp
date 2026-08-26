#include "bonsai/grower.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/parallel.hpp"
#include "bonsai/split.hpp"
#include "bonsai/types.hpp"
#include "fill/columns.hpp"
#include "fill/lone.hpp"
#include "fill/plan.hpp"
#include "fill/rows.hpp"
#include "grower_impl.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <functional>
#include <span>
#include <vector>

namespace bonsai
{

namespace fd = fill_detail;

namespace
{

// Density cutoff: a node holding rows >= n_rows / den routes to the column
// fill, sparser ones to the row-chunk fill; measured, not tunable
// (docs/invariants.md).
constexpr size_t k_col_fill_den = 4;

} // namespace

// One node's fill, the level plane's way: the node is a level of one, so it
// takes the same batched path a frontier does.
void CpuHistogramEngine::populate(Dataset const &ds, floats_view grad, floats_view hess,
                                  SplitInput                   &split_input,
                                  std::span<feature_id_t const> selected)
{
    std::array one = {std::ref(split_input)};
    populate_many(ds, grad, hess, one, selected);
}

// The leaf plane's per-split fill. Its node has no frontier to spread the work
// over, so the fill cuts the node itself: a sparse one into feature ranges by
// row blocks over the mirror, a dense or u16 one column by column. The carve
// and the subtraction from `sibling` (the larger child, holding the parent's
// histograms) ride the same cut, so a split's whole histogram step costs one
// parallel region, two when row blocks leave partials to reduce. Returns
// whether the subtraction rode the fill: a node this path declines falls back
// to the level plane's fill, which leaves the sibling to the caller.
bool CpuHistogramEngine::populate_lone(Dataset const &ds, floats_view grad,
                                       floats_view hess, SplitInput &split_input,
                                       std::span<feature_id_t const> selected,
                                       NodeHistograms               &sibling)
{
    hess = fd::fill_hess(hess);
    // The root is the one lone node the level growers populate too, and it
    // has no sibling to subtract from: it keeps their path.
    if (split_input.id == 0 || selected.empty() || split_input.rows.empty())
    {
        populate(ds, grad, hess, split_input, selected);
        return false;
    }
    fd::SelectionPlan const &sp = fd::selection_plan(ds, selected);
    split_input.hists.carve_storage(sp.layout, ds.n_features());
    if (ds.bins_are_u8() && split_input.rows.size() * k_col_fill_den < ds.n_rows())
    {
        // Decomposed to the node and the mirror row's width.
        fd::FillPlan const plan = fd::plan_lone_fill(
            split_input.rows.size(), sp, static_cast<size_t>(parallel::n_threads()));
        fd::fill_lone(ds, split_input, selected, sp, sp.layout, sibling, plan.ranges,
                      plan.blocks, fd::ordered_gh(split_input.rows, grad, hess));
    }
    else
    {
        fd::fill_columns_lone(ds, grad, hess, split_input, selected, sp.layout,
                              sibling);
    }
    // The carve rides the fill's decomposition, so a work list that missed a
    // run would read cells whose lifetime never started.
    assert(split_input.hists.all_runs_carved(sp.layout, selected));
    return true;
}

// Builds histograms for `selected` features only; unselected slots stay
// zero-binned placeholders the split finders skip. u8 data fills row-wise
// over the row-major mirror — cache-friendly at any node sparsity, with
// sums reproducible at a fixed thread count; single-chunk nodes and the u16
// feature-parallel path stay bit-identical at any thread count
// (docs/invariants.md).
void CpuHistogramEngine::populate_many(Dataset const &ds, floats_view grad,
                                       floats_view hess, split_input_refs nodes,
                                       std::span<feature_id_t const> selected)
{
    hess = fd::fill_hess(hess);
    // One layout for the tree: every node's arena packs the same way.
    fd::SelectionPlan const &sp = fd::selection_plan(ds, selected);
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
            fd::fill_columns(ds, grad, hess, node, selected);
        }
        return;
    }
    static thread_local std::vector<std::reference_wrapper<SplitInput>> sparse_nodes;
    sparse_nodes.clear();
    for (SplitInput &node : nodes)
    {
        if (node.rows.size() * k_col_fill_den >= ds.n_rows())
        {
            fd::fill_columns(ds, grad, hess, node, selected);
        }
        else
        {
            sparse_nodes.emplace_back(node);
        }
    }
    if (!sparse_nodes.empty())
    {
        fd::fill_sparse(ds, grad, hess, sparse_nodes, selected, sp);
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
    fd::plan_cache()           = {};
    fd::plan_cache().hess      = hess.data();
    fd::plan_cache().unit_hess = unit;
}

template class DepthwiseGrower<CpuHistogramEngine, HistogramNodeSplitFinder>;
template class ObliviousGrower<CpuHistogramEngine, HistogramLevelSplitFinder>;
template class LeafwiseGrower<CpuHistogramEngine, HistogramNodeSplitFinder>;

} // namespace bonsai
