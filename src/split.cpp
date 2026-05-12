#include "bonsai/split.hpp"
#include "bonsai/config/tree_config.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/types.hpp"
#include <cstddef>

namespace bonsai
{

namespace
{

inline void update_best_for_feature(SplitNode const &node, feature_id_t fid,
                                    TreeConfig const &config, Split &best)
{
    auto const &hist         = node.hists[fid];
    auto const &missing_cell = hist.missing();
    double const real_grad   = node.grad - missing_cell.sum_grad;
    double const real_hess   = node.hess - missing_cell.sum_hess;
    double const node_score  = score(node.grad, node.hess, config.lambda_l2);
    double left_grad         = 0.0;
    double left_hess         = 0.0;
    bin_id_t b               = 0;
    for (auto const &cell : hist.cut_cells())
    {
        left_grad += cell.sum_grad;
        left_hess += cell.sum_hess;
        double const right_grad = real_grad - left_grad;
        double const right_hess = real_hess - left_hess;

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
            if (h_left < config.min_child_hess || h_right < config.min_child_hess)
            {
                continue;
            }
            double const gain = score(g_left, h_left, config.lambda_l2) +
                                score(g_right, h_right, config.lambda_l2) - node_score;
            if (gain > best.gain && gain >= config.min_gain_to_split)
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

Split HistogramSplitFinder::find(SplitNode const &node, TreeConfig const &config)
{
    if (node.rows.size() < 2 * size_t{config.min_data_in_leaf})
    {
        return {};
    }
    Split best;
    auto const &hists = node.hists;
    for (feature_id_t fid = 0; fid < hists.size(); ++fid)
    {
        update_best_for_feature(node, fid, config, best);
    }
    return best;
}

} // namespace bonsai
