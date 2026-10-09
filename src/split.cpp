#include "bonsai/split.hpp"
#include "bonsai/config/tree_config.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/parallel.hpp"
#include "bonsai/types.hpp"
#include <algorithm>
#include <array>
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

inline HistCell cell_totals(SplitInput const &input)
{
    NodeTotals const t = input.totals();
    return {.sum_grad = static_cast<float>(t.sum_grad),
            .sum_hess = static_cast<float>(t.sum_hess)};
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
                // perf: An infeasible node contributes its parent score instead
                // of vetoing the candidate; at depth >= 5 some frontier node is
                // always near-empty, and a veto cost levelwise 3-26% against
                // catboost (invariants: infeasible-node-scores-its-parent,
                // zero-cover-branches-are-real).
                sum_children_score += c.feasible ? c.children_score : parent_score[p];
            }
            update_best(best, sum_children_score - sum_parent_score, fid,
                        static_cast<bin_id_t>(b), default_left, config);
        }
    }
}

// perf: Eight features scan side by side so the two divisions per cell
// vectorize across features; same-pod EPYC 4564P hot find 3.20 -> 1.76
// ns/cell at one thread and 4.71 -> 2.69 at twelve, M2 4.29 -> 3.14.
inline constexpr size_t k_scan_lanes = 8;

inline constexpr HistCell k_idle_cell{};

struct LaneGroup
{
    std::array<HistCell const *, k_scan_lanes> cells{};
    std::array<size_t, k_scan_lanes>           cuts{};
    std::array<feature_id_t, k_scan_lanes>     fids{};
    std::array<double, k_scan_lanes>           real_grad{};
    std::array<double, k_scan_lanes>           real_hess{};
    std::array<double, k_scan_lanes>           miss_grad{};
    std::array<double, k_scan_lanes>           miss_hess{};
    size_t                                     n        = 0;
    size_t                                     max_cuts = 0;
    bool                                       two_dirs = false;

    void clear()
    {
        cells.fill(&k_idle_cell);
        cuts.fill(0);
        n        = 0;
        max_cuts = 0;
        two_dirs = false;
    }

    bool full() const
    {
        return n == k_scan_lanes;
    }

    void add(feature_id_t fid, Histogram const &hist, HistCell const &node_totals)
    {
        HistCell const &missing = hist.missing();
        cells[n]                = hist.cut_cells().data();
        cuts[n]                 = hist.prefix_size();
        fids[n]                 = fid;
        real_grad[n]            = node_totals.sum_grad - missing.sum_grad;
        real_hess[n]            = node_totals.sum_hess - missing.sum_hess;
        miss_grad[n]            = missing.sum_grad;
        miss_hess[n]            = missing.sum_hess;
        max_cuts                = std::max(max_cuts, cuts[n]);
        two_dirs                = two_dirs || !missing_empty(missing);
        ++n;
    }
};

using lane_d = double __attribute__((ext_vector_type(k_scan_lanes)));
using lane_i = int64_t __attribute__((ext_vector_type(k_scan_lanes)));

struct LaneBest
{
    lane_d gain = 0.0;
    lane_i bin  = 0;
    lane_i left = 0;

    [[gnu::always_inline]] void offer(lane_i const &feasible, lane_d const &candidate,
                                      int64_t b, int64_t default_left, double min_gain)
    {
        lane_i const better = feasible & (candidate > gain) & (candidate >= min_gain);
        gain                = better ? candidate : gain;
        bin                 = better ? lane_i(b) : bin;
        left                = better ? lane_i(default_left) : left;
    }
};

[[gnu::always_inline]] inline void lane_l1_thresholded(lane_d const &g, double l1,
                                                       lane_d &t)
{
    t = g > l1 ? g - l1 : (g < -l1 ? g + l1 : lane_d(0.0));
}

[[gnu::always_inline]] inline void lane_score(lane_d const &g, lane_d const &h,
                                              double l1, double l2, lane_d &out)
{
    lane_d t;
    lane_l1_thresholded(g, l1, t);
    lane_d const d = h + l2;
    out            = d > 0.0 ? (t * t) / d : lane_d(0.0);
}

[[gnu::always_inline]] inline void
lane_children_score(lane_d const &grad_left, lane_d const &hess_left,
                    lane_d const &grad_right, lane_d const &hess_right, double l1,
                    double l2, lane_d &out)
{
    lane_d left;
    lane_d right;
    lane_score(grad_left, hess_left, l1, l2, left);
    lane_score(grad_right, hess_right, l1, l2, right);
    out = left + right;
}

struct LaneSums
{
    lane_d real_grad;
    lane_d real_hess;
    lane_d miss_grad;
    lane_d miss_hess;
};

[[gnu::always_inline]] inline void lane_sums(LaneGroup const &g, LaneSums &s)
{
    for (size_t l = 0; l < k_scan_lanes; ++l)
    {
        s.real_grad[l] = g.real_grad[l];
        s.real_hess[l] = g.real_hess[l];
        s.miss_grad[l] = g.miss_grad[l];
        s.miss_hess[l] = g.miss_hess[l];
    }
}

struct LaneRule
{
    double l1;
    double l2;
    double min_hess;
    double min_gain;
    double node_score;
};

inline LaneRule lane_rule(TreeConfig const &config, double node_score)
{
    return {.l1         = config.lambda_l1,
            .l2         = config.lambda_l2,
            .min_hess   = config.min_child_hess,
            .min_gain   = config.min_gain_to_split,
            .node_score = node_score};
}

struct LanePrefix
{
    lane_d left_grad;
    lane_d left_hess;
    lane_d right_grad;
    lane_d right_hess;
    lane_i active;
};

template <bool DefaultLeft>
[[gnu::always_inline]] inline void offer_direction(LaneBest &best, LaneSums const &s,
                                                   LanePrefix const &p,
                                                   LaneRule const &rule, int64_t b)
{
    lane_d const none       = 0.0;
    lane_d const grad_left  = p.left_grad + (DefaultLeft ? s.miss_grad : none);
    lane_d const hess_left  = p.left_hess + (DefaultLeft ? s.miss_hess : none);
    lane_d const grad_right = p.right_grad + (DefaultLeft ? none : s.miss_grad);
    lane_d const hess_right = p.right_hess + (DefaultLeft ? none : s.miss_hess);
    lane_i const feasible =
        p.active & ~((hess_left < rule.min_hess) | (hess_right < rule.min_hess));
    lane_d children;
    lane_children_score(grad_left, hess_left, grad_right, hess_right, rule.l1, rule.l2,
                        children);
    best.offer(feasible, children - rule.node_score, b, DefaultLeft ? 1 : 0,
               rule.min_gain);
}

template <bool TwoDirs>
[[gnu::always_inline]] inline void scan_lanes_body(LaneGroup const &g,
                                                   LaneRule const &rule, LaneBest &best)
{
    LaneSums s;
    lane_sums(g, s);
    LanePrefix p{};
    for (size_t b = 0; b < g.max_cuts; ++b)
    {
        lane_d cell_grad;
        lane_d cell_hess;
        for (size_t l = 0; l < k_scan_lanes; ++l)
        {
            bool const     in   = b < g.cuts[l];
            HistCell const cell = g.cells[l][in ? b : 0];
            cell_grad[l]        = in ? cell.sum_grad : 0.0F;
            cell_hess[l]        = in ? cell.sum_hess : 0.0F;
            p.active[l]         = in ? -1 : 0;
        }
        p.left_grad += cell_grad;
        p.left_hess += cell_hess;
        p.right_grad = s.real_grad - p.left_grad;
        p.right_hess = s.real_hess - p.left_hess;
        offer_direction<true>(best, s, p, rule, static_cast<int64_t>(b));
        if constexpr (TwoDirs)
        {
            offer_direction<false>(best, s, p, rule, static_cast<int64_t>(b));
        }
    }
}

#ifdef __x86_64__
inline bool lanes_available()
{
    static bool const avx2 = []
    {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2");
    }();
    return avx2;
}

template <bool TwoDirs>
__attribute__((target("avx2"))) void scan_lanes(LaneGroup const &g,
                                                LaneRule const &rule, LaneBest &best)
{
    scan_lanes_body<TwoDirs>(g, rule, best);
}
#else
inline bool lanes_available()
{
    return true;
}

template <bool TwoDirs>
void scan_lanes(LaneGroup const &g, LaneRule const &rule, LaneBest &best)
{
    scan_lanes_body<TwoDirs>(g, rule, best);
}
#endif

inline void flush_lanes(LaneGroup &g, LaneRule const &rule, SplitOutput &best)
{
    if (g.n == 0)
    {
        return;
    }
    LaneBest lanes;
    if (g.two_dirs)
    {
        scan_lanes<true>(g, rule, lanes);
    }
    else
    {
        scan_lanes<false>(g, rule, lanes);
    }
    for (size_t l = 0; l < g.n; ++l)
    {
        if (lanes.gain[l] > best.gain)
        {
            best = {.gain         = lanes.gain[l],
                    .feature_id   = g.fids[l],
                    .bin_id       = static_cast<bin_id_t>(lanes.bin[l]),
                    .default_left = lanes.left[l] != 0,
                    .valid        = true};
        }
    }
    g.clear();
}

inline bool scans_in_lanes(SplitInput const &input, feature_id_t fid,
                           TreeConfig const &config)
{
    return lanes_available() && monotone_constraint_of(config, fid) == 0 &&
           input.hists[fid].size() != 0 &&
           (input.allowed.empty() || input.allowed[fid] != 0);
}

SplitOutput scan_features(SplitInput const &input, feature_id_t f0, feature_id_t f1,
                          HistCell const &node_totals, TreeConfig const &config)
{
    LaneRule const rule =
        lane_rule(config, score(node_totals.sum_grad, node_totals.sum_hess,
                                config.lambda_l1, config.lambda_l2));
    SplitOutput best;
    LaneGroup   group;
    group.clear();
    for (feature_id_t fid = f0; fid < f1; ++fid)
    {
        if (scans_in_lanes(input, fid, config))
        {
            group.add(fid, input.hists[fid], node_totals);
            if (group.full())
            {
                flush_lanes(group, rule, best);
            }
            continue;
        }
        flush_lanes(group, rule, best);
        update_best_for_feature_for_node(input, fid, node_totals, config, best);
    }
    flush_lanes(group, rule, best);
    return best;
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

constexpr size_t k_scan_ranges_per_thread = 4;

size_t scan_ranges(size_t n_features)
{
    size_t const units =
        static_cast<size_t>(parallel::n_threads()) * k_scan_ranges_per_thread;
    return std::max<size_t>(1, std::min(n_features / k_scan_lanes, units));
}

} // namespace

SplitOutput HistogramNodeSplitFinder::find(SplitInput const &input,
                                           TreeConfig const &config)
{
    if (cannot_split(input, config))
    {
        return {};
    }
    feature_id_t const n_features = input.hists.size();
    return scan_features(input, 0, n_features, cell_totals(input), config);
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
            totals[i] = cell_totals(nodes[i]);
        }
    }
    HistCell const *const tot_ptr  = totals.data();
    SplitOutput *const    part_ptr = partials.data();
    parallel::for_each_index(n * ranges,
                             [&, tot_ptr, part_ptr](size_t u)
                             {
                                 SplitInput const &input = nodes[u / ranges];
                                 if (cannot_split(input, config))
                                 {
                                     return;
                                 }
                                 size_t const nf = input.hists.size();
                                 size_t const r  = u % ranges;
                                 part_ptr[u]     = scan_features(
                                     input, static_cast<feature_id_t>(r * nf / ranges),
                                     static_cast<feature_id_t>((r + 1) * nf / ranges),
                                     tot_ptr[u / ranges], config);
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
        node_totals[p] = cell_totals(frontier[p]);
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
