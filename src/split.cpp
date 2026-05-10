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

} // namespace

Split HistogramSplitFinder::find(histogram_view_t hists, Params params)
{
    Split best_split;
    for (feature_id_t fid = 0; fid < hists.size(); ++fid)
    {
        auto const &hist         = hists[fid];
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
            double right_grad = total_grad - left_grad;
            double right_hess = total_hess - left_hess;

            for (bool const default_left : {true, false})
            {
                double g_left =
                    left_grad + (default_left ? missing_cell.sum_grad : 0.0);
                double hess_left =
                    left_hess + (default_left ? missing_cell.sum_hess : 0.0);
                double g_right =
                    right_grad + (!default_left ? missing_cell.sum_grad : 0.0);
                double h_right =
                    right_hess + (!default_left ? missing_cell.sum_hess : 0.0);
                if (hess_left < params.min_child_hess ||
                    h_right < params.min_child_hess)
                {
                    continue;
                }
                double const gain = score(g_left, hess_left, params.lambda_l2) +
                                    score(g_right, h_right, params.lambda_l2) -
                                    params.parent_score;
                if (gain > best_split.gain)
                {
                    best_split = {.gain         = gain,
                                  .feature_id   = fid,
                                  .bin_id       = b,
                                  .default_left = default_left,
                                  .valid        = true};
                }
            }
            ++b;
        }
    }
    return best_split;
}

} // namespace bonsai
