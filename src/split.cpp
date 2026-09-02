#include "bonsai/split.hpp"
#include "bonsai/config/tree_config.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/parallel.hpp"
#include "bonsai/types.hpp"
#include <algorithm>
#include <cstddef>
#include <mdspan>
#include <span>
#include <vector>

namespace bonsai
{

namespace
{

struct SplitSums
{
    double gL;
    double hL;
    double gR;
    double hR;
};

inline SplitSums split_sums_at(double left_grad, double left_hess,
                               HistCell const &missing, double real_grad,
                               double real_hess, bool default_left)
{
    double const right_grad = real_grad - left_grad;
    double const right_hess = real_hess - left_hess;
    return {
        .gL = left_grad + (default_left ? missing.sum_grad : 0.0),
        .hL = left_hess + (default_left ? missing.sum_hess : 0.0),
        .gR = right_grad + (!default_left ? missing.sum_grad : 0.0),
        .hR = right_hess + (!default_left ? missing.sum_hess : 0.0),
    };
}

struct CandidateScore
{
    SplitSums s;
    double    children_score = 0.0;
    bool      feasible       = false;
};

inline CandidateScore score_candidate(double left_grad, double left_hess,
                                      HistCell const &missing, double real_grad,
                                      double real_hess, bool default_left,
                                      TreeConfig const &config)
{
    auto const s = split_sums_at(left_grad, left_hess, missing, real_grad, real_hess,
                                 default_left);
    if (s.hL < config.min_child_hess || s.hR < config.min_child_hess)
    {
        return {.s = s};
    }
    return {.s              = s,
            .children_score = score(s.gL, s.hL, config.lambda_l1, config.lambda_l2) +
                              score(s.gR, s.hR, config.lambda_l1, config.lambda_l2),
            .feasible = true};
}

inline bool missing_empty(HistCell const &missing)
{
    return missing.sum_grad == 0.0F && missing.sum_hess == 0.0F;
}

inline void update_best(SplitOutput &best, double gain, feature_id_t fid, bin_id_t bin,
                        bool default_left, TreeConfig const &config)
{
    if (gain > best.gain && gain >= config.min_gain_to_split)
    {
        best = {.gain         = gain,
                .feature_id   = fid,
                .bin_id       = bin,
                .default_left = default_left,
                .valid        = true};
    }
}

inline void update_best_for_feature_for_node(SplitInput const &input, feature_id_t fid,
                                             HistCell const   &node_totals,
                                             TreeConfig const &config,
                                             SplitOutput      &best)
{
    auto const &hist = input.hists[fid];
    if (hist.size() == 0)
    {
        return;
    }
    if (!input.allowed.empty() && input.allowed[fid] == 0)
    {
        return;
    }
    auto const  &missing_cell = hist.missing();
    double const node_score   = score(node_totals.sum_grad, node_totals.sum_hess,
                                      config.lambda_l1, config.lambda_l2);
    double const real_grad    = node_totals.sum_grad - missing_cell.sum_grad;
    double const real_hess    = node_totals.sum_hess - missing_cell.sum_hess;

    int const mc = monotone_constraint_of(config, fid);

    size_t const n_dirs = missing_empty(missing_cell) ? 1 : 2;

    double   left_grad = 0.0;
    double   left_hess = 0.0;
    bin_id_t b         = 0;
    for (auto const &cell : hist.cut_cells())
    {
        left_grad += cell.sum_grad;
        left_hess += cell.sum_hess;
        for (size_t d = 0; d < n_dirs; ++d)
        {
            bool const default_left = d == 0;
            auto const c = score_candidate(left_grad, left_hess, missing_cell,
                                           real_grad, real_hess, default_left, config);
            if (!c.feasible)
            {
                continue;
            }
            if (mc != 0)
            {
                double const w_left =
                    bounded_leaf_weight(c.s.gL, c.s.hL, config, input.lo, input.hi);
                double const w_right =
                    bounded_leaf_weight(c.s.gR, c.s.hR, config, input.lo, input.hi);
                if (static_cast<double>(mc) * (w_right - w_left) < 0.0)
                {
                    continue;
                }
            }
            update_best(best, c.children_score - node_score, fid, b, default_left,
                        config);
        }
        ++b;
    }
}

inline void update_best_for_feature_for_level(FrontierInput frontier, feature_id_t fid,
                                              std::vector<HistCell> const &node_totals,
                                              TreeConfig const            &config,
                                              SplitOutput                 &best)
{
    size_t const n_parents = frontier.size();
    size_t const n_bins    = frontier.front().hists[fid].prefix_size();
    if (n_bins == 0)
    {
        return;
    }

    static thread_local std::vector<HistCell> prefix_storage;
    static thread_local std::vector<double>   real_grad;
    static thread_local std::vector<double>   real_hess;
    prefix_storage.resize(n_parents * n_bins);
    real_grad.resize(n_parents);
    real_hess.resize(n_parents);
    auto prefix = std::mdspan<HistCell, std::dextents<size_t, 2>>(prefix_storage.data(),
                                                                  n_parents, n_bins);

    static thread_local std::vector<double> parent_score;
    parent_score.resize(n_parents);
    double sum_parent_score  = 0.0;
    bool   all_missing_empty = true;
    for (size_t p = 0; p < n_parents; ++p)
    {
        auto const &hist    = frontier[p].hists[fid];
        auto const &missing = hist.missing();
        hist.fill_prefix(std::span(&prefix[p, 0], n_bins));
        parent_score[p] = score(node_totals[p].sum_grad, node_totals[p].sum_hess,
                                config.lambda_l1, config.lambda_l2);
        sum_parent_score += parent_score[p];
        real_grad[p]      = node_totals[p].sum_grad - missing.sum_grad;
        real_hess[p]      = node_totals[p].sum_hess - missing.sum_hess;
        all_missing_empty = all_missing_empty && missing_empty(missing);
    }
    size_t const n_dirs = all_missing_empty ? 1 : 2;

    for (size_t b = 0; b < n_bins; ++b)
    {
        for (size_t d = 0; d < n_dirs; ++d)
        {
            bool const default_left       = d == 0;
            double     sum_children_score = 0.0;
            for (size_t p = 0; p < n_parents; ++p)
            {
                auto const     &hist = frontier[p].hists[fid];
                HistCell const &lp   = prefix[p, b];
                auto const      c =
                    score_candidate(lp.sum_grad, lp.sum_hess, hist.missing(),
                                    real_grad[p], real_hess[p], default_left, config);
                // perf: A node whose children would fall under min_child_hess no
                // longer vetoes the whole candidate (at depth >= 5
                // some frontier node is always near-empty, so every good cut
                // was rejected and levelwise trailed catboost by 3-26%). It
                // contributes its parent score instead (zero gain) and the
                // broadcast split still applies to it; empty children are
                // first-class (invariants: zero-cover-branches-are-real).
                sum_children_score += c.feasible ? c.children_score : parent_score[p];
            }
            update_best(best, sum_children_score - sum_parent_score, fid,
                        static_cast<bin_id_t>(b), default_left, config);
        }
    }
}

SplitOutput reduce_in_feature_order(std::span<SplitOutput const> per_feature)
{
    SplitOutput best;
    for (auto const &cand : per_feature)
    {
        if (cand.valid && cand.gain > best.gain)
        {
            best = cand;
        }
    }
    return best;
}

bool cannot_split(SplitInput const &input, TreeConfig const &config)
{
    return input.hists.empty() ||
           input.rows.size() < 2 * size_t{config.min_data_in_leaf};
}

size_t scan_ranges(size_t n_features)
{
    size_t const units = static_cast<size_t>(parallel::n_threads()) * 4;
    return std::max<size_t>(1, std::min(n_features, units));
}

} // namespace

SplitOutput HistogramNodeSplitFinder::find(SplitInput const &input,
                                           TreeConfig const &config)
{
    if (cannot_split(input, config))
    {
        return {};
    }
    feature_id_t const n_features  = input.hists.size();
    HistCell const     node_totals = input.totals();
    SplitOutput        best;
    for (feature_id_t fid = 0; fid < n_features; ++fid)
    {
        SplitOutput cand;
        update_best_for_feature_for_node(input, fid, node_totals, config, cand);
        if (cand.valid && cand.gain > best.gain)
        {
            best = cand;
        }
    }
    return best;
}

void HistogramNodeSplitFinder::find_parallel(std::span<SplitInput const> nodes,
                                             TreeConfig const           &config,
                                             std::span<SplitOutput>      out)
{
    static thread_local std::vector<HistCell>    totals;
    static thread_local std::vector<SplitOutput> partials;
    size_t const                                 n = nodes.size();
    size_t const ranges = scan_ranges(nodes.empty() ? 0 : nodes.front().hists.size());
    totals.assign(n, HistCell{});
    partials.assign(n * ranges, SplitOutput{});
    for (size_t i = 0; i < n; ++i)
    {
        if (!cannot_split(nodes[i], config))
        {
            totals[i] = nodes[i].totals();
        }
    }
    HistCell const *const tot_ptr  = totals.data();
    SplitOutput *const    part_ptr = partials.data();
    parallel::for_each_index(
        n * ranges,
        [&, tot_ptr, part_ptr](size_t u)
        {
            SplitInput const &input = nodes[u / ranges];
            if (cannot_split(input, config))
            {
                return;
            }
            size_t const nf = input.hists.size();
            size_t const r  = u % ranges;
            for (size_t fid = r * nf / ranges; fid < (r + 1) * nf / ranges; ++fid)
            {
                update_best_for_feature_for_node(input, static_cast<feature_id_t>(fid),
                                                 tot_ptr[u / ranges], config,
                                                 part_ptr[u]);
            }
        });
    for (size_t i = 0; i < n; ++i)
    {
        out[i] = reduce_in_feature_order({part_ptr + (i * ranges), ranges});
    }
}

SplitOutput HistogramLevelSplitFinder::find(FrontierInput     frontier,
                                            TreeConfig const &config)
{
    if (frontier.empty())
    {
        return {};
    }
    feature_id_t const    n_features = frontier.front().hists.size();
    std::vector<HistCell> node_totals(frontier.size());
    for (size_t p = 0; p < frontier.size(); ++p)
    {
        node_totals[p] = frontier[p].totals();
    }
    std::vector<SplitOutput> per_feature(n_features);
    parallel::for_each_index(n_features,
                             [&](size_t fid)
                             {
                                 update_best_for_feature_for_level(
                                     frontier, static_cast<feature_id_t>(fid),
                                     node_totals, config, per_feature[fid]);
                             });
    return reduce_in_feature_order(per_feature);
}

} // namespace bonsai
