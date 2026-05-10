#include "bonsai/split.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/types.hpp"

namespace bonsai
{

namespace
{

inline double score(double g, double h, double lambda)
{
    return (g * g) / (h + lambda);
}

inline void update_best_for_feature(feature_id_t fid, Histogram const &hist,
                                    HistogramSplitFinder::Params const &params,
                                    Split &best)
{
    double total_grad        = 0.0;
    double total_hess        = 0.0;
    auto const &missing_cell = hist.missing();
    for (auto const &cell : hist.sweep_cells())
    {
        total_grad += cell.sum_grad;
        total_hess += cell.sum_hess;
    }
    double left_grad = 0.0;
    double left_hess = 0.0;
    bin_id_t b       = 0;
    for (auto const &cell : hist.cut_cells())
    {
        left_grad += cell.sum_grad;
        left_hess += cell.sum_hess;
        double const right_grad = total_grad - left_grad;
        double const right_hess = total_hess - left_hess;

        for (bool const default_left : {true, false})
        {
            double const g_left =
                left_grad + (default_left ? missing_cell.sum_grad : 0.0);
            double const h_left =
                left_hess + (default_left ? missing_cell.sum_hess : 0.0);
            double const g_right =
                right_grad + (!default_left ? missing_cell.sum_grad : 0.0);
            double const h_right =
                right_hess + (!default_left ? missing_cell.sum_hess : 0.0);
            if (h_left < params.min_child_hess || h_right < params.min_child_hess)
            {
                continue;
            }
            double const gain = score(g_left, h_left, params.lambda_l2) +
                                score(g_right, h_right, params.lambda_l2) -
                                params.parent_score;
            if (gain > best.gain)
            {
                best = {.gain         = gain,
                        .feature_id   = fid,
                        .bin_id       = b,
                        .default_left = default_left,
                        .valid        = true};
            }
        }
        ++b;
    }
}

} // namespace

Split HistogramSplitFinder::find(histogram_view_t hists, Params params)
{
    Split best;
    for (feature_id_t fid = 0; fid < hists.size(); ++fid)
    {
        update_best_for_feature(fid, hists[fid], params, best);
    }
    return best;
}

} // namespace bonsai
