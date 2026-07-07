#include "bonsai/grower.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/parallel.hpp"
#include "bonsai/split.hpp"
#include "bonsai/types.hpp"
#include "grower_impl.hpp"
#include <cstddef>
#include <span>
#include <vector>

namespace bonsai
{

// Builds histograms for `selected` features only; unselected slots stay
// zero-binned placeholders the split finders skip.
void CpuHistogramBuilder::populate(Dataset const &ds, floats_view grad,
                                   floats_view hess, SplitInput &node,
                                   std::span<feature_id_t const> selected)
{
    // Gather grad/hess into node-row order once, so every feature's scan
    // below reads them sequentially instead of re-walking the full arrays
    // with scattered indices (n_features x full-array traffic otherwise).
    static thread_local std::vector<float> ordered_grad;
    static thread_local std::vector<float> ordered_hess;
    size_t const n = node.rows.size();
    ordered_grad.resize(n);
    ordered_hess.resize(n);
    // Capture raw pointers: naming the thread_local inside the parallel
    // region would resolve to each worker's own (empty) vector.
    float *const og = ordered_grad.data();
    float *const oh = ordered_hess.data();
    parallel::for_each_index(n,
                             [&, og, oh](size_t k)
                             {
                                 row_id_t const r = node.rows[k];
                                 og[k]            = grad[r];
                                 oh[k]            = hess[r];
                             });

    node.hists.reserve(ds.n_features());
    size_t j = 0;
    for (feature_id_t fid = 0; fid < ds.n_features(); ++fid)
    {
        bool const sel = j < selected.size() && selected[j] == fid;
        node.hists.emplace_back(sel ? ds.n_bins(fid) : 0);
        j += sel ? 1 : 0;
    }
    // Feature-parallel: each feature's histogram is owned by one thread and
    // filled in row order, so results are bit-identical at any thread count.
    parallel::for_each_index(selected.size(),
                             [&, og, oh](size_t s)
                             {
                                 feature_id_t const fid  = selected[s];
                                 Histogram         &h    = node.hists[fid];
                                 auto const        &bins = ds.feature_bins(fid);
                                 for (size_t k = 0; k < n; ++k)
                                 {
                                     h.add(bins[node.rows[k]], og[k], oh[k]);
                                 }
                             });
}

template class DepthwiseGrower<HistogramNodeSplitFinder, CpuHistogramBuilder>;
template class ObliviousGrower<HistogramLevelSplitFinder, CpuHistogramBuilder>;
template class LeafwiseGrower<HistogramNodeSplitFinder, CpuHistogramBuilder>;

} // namespace bonsai
