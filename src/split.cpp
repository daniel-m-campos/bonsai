#include "bonsai/split.hpp"
#include "bonsai/config/tree_config.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/types.hpp"
#include <cstddef>

namespace bonsai
{

namespace
{

struct SplitSums
{
    double gL; // left  grad with missing routed by default_left
    double hL; // left  hess
    double gR; // right grad
    double hR; // right hess
};

// Decompose a candidate cut at `left_prefix` (cumulative sum over
// cut_cells up to and including bin b) into per-side sums for gain
// scoring. `real_grad` / `real_hess` are the node's totals
// EXCLUDING the missing bin (= parent_totals - missing); the caller
// hoists that subtraction once per feature/parent so the inner loop
// doesn't recompute it per bin/default_left. Single source of truth
// for missing-routing semantics; shared by node and level finders.
inline SplitSums split_sums_at(HistCell const &left_prefix, HistCell const &missing,
                               double real_grad, double real_hess, bool default_left)
{
    double const right_grad = real_grad - left_prefix.sum_grad;
    double const right_hess = real_hess - left_prefix.sum_hess;
    return {
        .gL = left_prefix.sum_grad + (default_left ? missing.sum_grad : 0.0),
        .hL = left_prefix.sum_hess + (default_left ? missing.sum_hess : 0.0),
        .gR = right_grad + (!default_left ? missing.sum_grad : 0.0),
        .hR = right_hess + (!default_left ? missing.sum_hess : 0.0),
    };
}

inline void update_best_for_feature_for_node(SplitInput const &input, feature_id_t fid,
                                             TreeConfig const &config,
                                             SplitOutput &best)
{
    auto const &hist         = input.hists[fid];
    auto const &missing_cell = hist.missing();
    double const node_score =
        score(hist.total_grad(), hist.total_hess(), config.lambda_l2);
    double const real_grad = hist.total_grad() - missing_cell.sum_grad;
    double const real_hess = hist.total_hess() - missing_cell.sum_hess;

    HistCell left_prefix{};
    bin_id_t b = 0;
    for (auto const &cell : hist.cut_cells())
    {
        left_prefix.sum_grad += cell.sum_grad;
        left_prefix.sum_hess += cell.sum_hess;
        for (bool const default_left : {true, false})
        {
            auto const s = split_sums_at(left_prefix, missing_cell, real_grad,
                                         real_hess, default_left);
            if (s.hL < config.min_child_hess || s.hR < config.min_child_hess)
            {
                continue;
            }
            double const gain = score(s.gL, s.hL, config.lambda_l2) +
                                score(s.gR, s.hR, config.lambda_l2) - node_score;
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

SplitOutput HistogramNodeSplitFinder::find(SplitInput const &input,
                                           TreeConfig const &config)
{
    if (input.rows.size() < 2 * size_t{config.min_data_in_leaf})
    {
        return {};
    }
    SplitOutput best;
    auto const &hists = input.hists;
    for (feature_id_t fid = 0; fid < hists.size(); ++fid)
    {
        update_best_for_feature_for_node(input, fid, config, best);
    }
    return best;
}

namespace
{
[[maybe_unused]] inline void
update_best_for_feature_for_level([[maybe_unused]] SplitInput const &input,
                                  [[maybe_unused]] feature_id_t fid,
                                  [[maybe_unused]] TreeConfig const &config,
                                  [[maybe_unused]] SplitOutput &best)
{
}

} // namespace

SplitOutput HistogramLevelSplitFinder::find([[maybe_unused]] FrontierInput frontier,
                                            [[maybe_unused]] TreeConfig const &config)
{
    SplitOutput best;

    return best;
}

} // namespace bonsai
