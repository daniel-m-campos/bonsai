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

// perf: Density cutoff: a node holding rows >= n_rows / den routes to the column
// fill, sparser ones to the row-chunk fill; measured, not tunable.
constexpr size_t k_col_fill_den = 4;

} // namespace

void CpuHistogramEngine::populate(Dataset const &ds, floats_view grad, floats_view hess,
                                  SplitInput                   &split_input,
                                  std::span<feature_id_t const> selected)
{
    std::array one = {std::ref(split_input)};
    populate_many(ds, grad, hess, one, selected);
} // namespace fd

bool CpuHistogramEngine::populate_lone(Dataset const &ds, floats_view grad,
                                       floats_view hess, SplitInput &split_input,
                                       std::span<feature_id_t const> selected,
                                       NodeHistograms               &sibling)
{
    hess = fd::fill_hess(hess);
    if (split_input.id == 0 || selected.empty() || split_input.rows.empty())
    {
        populate(ds, grad, hess, split_input, selected);
        return false;
    }
    fd::SelectionPlan const &sp = fd::selection_plan(ds, selected);
    split_input.hists.carve_storage(sp.layout, ds.n_features());
    if (ds.bins_are_u8() &&
        split_input.rows.size() * k_col_fill_den < ds.plane_n_rows())
    {
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
    assert(split_input.hists.all_runs_carved(sp.layout, selected));
    return true;
}

void CpuHistogramEngine::populate_many(Dataset const &ds, floats_view grad,
                                       floats_view hess, split_input_refs nodes,
                                       std::span<feature_id_t const> selected)
{
    hess                           = fd::fill_hess(hess);
    fd::SelectionPlan const &sp    = fd::selection_plan(ds, selected);
    bool const               alone = nodes.size() == 1;
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
        if (node.rows.size() * k_col_fill_den >= ds.plane_n_rows())
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

void CpuHistogramEngine::begin_tree(Dataset const & /*ds*/, floats_view /*grad*/,
                                    floats_view hess)
{
    bool const unit =
        !hess.empty() && std::ranges::all_of(hess, [](float h) { return h == 1.0F; });
    fd::plan_cache()           = {};
    fd::plan_cache().hess      = hess.data();
    fd::plan_cache().unit_hess = unit;
}

template bool GrowerHost<CpuHistogramEngine>::eval_accumulate<DenseTree>(
    DenseTree const &, Dataset const &, float, std::span<float>,
    std::optional<float> &);
template bool GrowerHost<CpuHistogramEngine>::eval_accumulate<ObliviousTree>(
    ObliviousTree const &, Dataset const &, float, std::span<float>,
    std::optional<float> &);
template class DepthwiseGrower<CpuHistogramEngine, HistogramNodeSplitFinder>;
template class ObliviousGrower<CpuHistogramEngine, HistogramLevelSplitFinder>;
template class LeafwiseGrower<CpuHistogramEngine, HistogramNodeSplitFinder>;

} // namespace bonsai
