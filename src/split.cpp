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

struct LevelParents
{
    FrontierInput             frontier;
    std::span<size_t const>   populated;
    std::span<HistCell const> totals;

    SplitInput const &front() const
    {
        return frontier[populated.front()];
    }
};

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

inline bool node_scans(SplitInput const &input, feature_id_t fid)
{
    return input.hists[fid].size() != 0 &&
           (input.allowed.empty() || input.allowed[fid] != 0);
}

inline void update_best_for_feature_for_node(SplitInput const &input, feature_id_t fid,
                                             HistCell const   &node_totals,
                                             TreeConfig const &config,
                                             SplitOutput      &best)
{
    if (!node_scans(input, fid))
    {
        return;
    }
    auto const  &hist         = input.hists[fid];
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

inline void update_best_for_feature_for_level(LevelParents const &lp, feature_id_t fid,
                                              TreeConfig const &config,
                                              SplitOutput      &best)
{
    size_t const n_parents = lp.frontier.size();
    size_t const n_bins    = lp.front().hists[fid].prefix_size();
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
    for (size_t const p : lp.populated)
    {
        auto const &hist    = lp.frontier[p].hists[fid];
        auto const &missing = hist.missing();
        hist.fill_prefix(std::span(&prefix[p, 0], n_bins));
        parent_score[p] = score(lp.totals[p].sum_grad, lp.totals[p].sum_hess,
                                config.lambda_l1, config.lambda_l2);
        sum_parent_score += parent_score[p];
        real_grad[p]      = lp.totals[p].sum_grad - missing.sum_grad;
        real_hess[p]      = lp.totals[p].sum_hess - missing.sum_hess;
        all_missing_empty = all_missing_empty && missing_empty(missing);
    }
    size_t const n_dirs = all_missing_empty ? 1 : 2;

    for (size_t b = 0; b < n_bins; ++b)
    {
        for (size_t d = 0; d < n_dirs; ++d)
        {
            bool const default_left       = d == 0;
            double     sum_children_score = 0.0;
            for (size_t const p : lp.populated)
            {
                auto const     &hist = lp.frontier[p].hists[fid];
                HistCell const &left = prefix[p, b];
                auto const      c =
                    score_candidate(left.sum_grad, left.sum_hess, hist.missing(),
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

using lane_d = double __attribute__((ext_vector_type(k_scan_lanes)));
using lane_i = int64_t __attribute__((ext_vector_type(k_scan_lanes)));

struct LaneSums
{
    lane_d real_grad;
    lane_d real_hess;
    lane_d miss_grad;
    lane_d miss_hess;
};

struct LaneGroup
{
    std::array<HistCell const *, k_scan_lanes> cells{};
    std::array<size_t, k_scan_lanes>           cuts{};
    std::array<feature_id_t, k_scan_lanes>     fids{};
    LaneSums                                   sums{};
    size_t                                     n        = 0;
    size_t                                     max_cuts = 0;
    bool                                       two_dirs = false;

    void clear()
    {
        cells.fill(&k_idle_cell);
        cuts.fill(0);
        sums     = {};
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
        sums.real_grad[n]       = node_totals.sum_grad - missing.sum_grad;
        sums.real_hess[n]       = node_totals.sum_hess - missing.sum_hess;
        sums.miss_grad[n]       = missing.sum_grad;
        sums.miss_hess[n]       = missing.sum_hess;
        max_cuts                = std::max(max_cuts, cuts[n]);
        two_dirs                = two_dirs || !missing_empty(missing);
        ++n;
    }
};

struct LaneBest
{
    lane_d gain = 0.0;
    lane_i bin  = 0;
    lane_i left = 0;

    [[gnu::always_inline]] void offer(lane_i const &feasible, lane_d const &candidate,
                                      int64_t b, bool default_left, double min_gain)
    {
        lane_i const better = feasible & (candidate > gain) & (candidate >= min_gain);
        gain                = better ? candidate : gain;
        bin                 = better ? lane_i(b) : bin;
        left                = better ? lane_i(default_left ? 1 : 0) : left;
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

struct LaneCandidate
{
    lane_i feasible;
    lane_d children;
};

template <bool DefaultLeft>
[[gnu::always_inline]] inline void
lane_candidate(LaneSums const &s, LanePrefix const &p, LaneRule const &rule,
               LaneCandidate &c)
{
    lane_d const none       = 0.0;
    lane_d const grad_left  = p.left_grad + (DefaultLeft ? s.miss_grad : none);
    lane_d const hess_left  = p.left_hess + (DefaultLeft ? s.miss_hess : none);
    lane_d const grad_right = p.right_grad + (DefaultLeft ? none : s.miss_grad);
    lane_d const hess_right = p.right_hess + (DefaultLeft ? none : s.miss_hess);
    c.feasible =
        p.active & ~((hess_left < rule.min_hess) | (hess_right < rule.min_hess));
    lane_children_score(grad_left, hess_left, grad_right, hess_right, rule.l1, rule.l2,
                        c.children);
}

template <bool DefaultLeft>
[[gnu::always_inline]] inline void offer_direction(LaneBest &best, LaneSums const &s,
                                                   LanePrefix const &p,
                                                   LaneRule const &rule, int64_t b)
{
    LaneCandidate c;
    lane_candidate<DefaultLeft>(s, p, rule, c);
    best.offer(c.feasible, c.children - rule.node_score, b, DefaultLeft, rule.min_gain);
}

[[gnu::always_inline]] inline void gather_cells(LaneGroup const &g, size_t b,
                                                lane_d &cell_grad, lane_d &cell_hess,
                                                lane_i &active)
{
    for (size_t l = 0; l < k_scan_lanes; ++l)
    {
        bool const     in   = b < g.cuts[l];
        HistCell const cell = g.cells[l][in ? b : 0];
        cell_grad[l]        = in ? cell.sum_grad : 0.0F;
        cell_hess[l]        = in ? cell.sum_hess : 0.0F;
        active[l]           = in ? -1 : 0;
    }
}

template <bool TwoDirs>
[[gnu::always_inline]] inline void scan_lanes_body(LaneGroup const &g,
                                                   LaneRule const &rule, LaneBest &best)
{
    LaneSums const &s = g.sums;
    LanePrefix      p{};
    for (size_t b = 0; b < g.max_cuts; ++b)
    {
        lane_d cell_grad;
        lane_d cell_hess;
        gather_cells(g, b, cell_grad, cell_hess, p.active);
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

using lane_f = float __attribute__((ext_vector_type(k_scan_lanes)));

[[gnu::always_inline]] inline void narrow_lanes(lane_d const &wide, lane_d &out)
{
    out = __builtin_convertvector(__builtin_convertvector(wide, lane_f), lane_d);
}

template <bool DefaultLeft>
[[gnu::always_inline]] inline void
accumulate_direction(lane_d &acc, LaneSums const &s, LanePrefix const &p,
                     LaneRule const &rule, double parent_score)
{
    LaneCandidate c;
    lane_candidate<DefaultLeft>(s, p, rule, c);
    acc += c.feasible ? c.children : lane_d(parent_score);
}

template <bool TwoDirs>
[[gnu::always_inline]] inline void
accumulate_lanes_body(LaneGroup const &g, LaneRule const &rule, double parent_score,
                      lane_d *acc)
{
    constexpr size_t n_dirs = TwoDirs ? 2 : 1;
    LaneSums const  &s      = g.sums;
    LanePrefix       p{};
    lane_d           sum_grad = 0.0;
    lane_d           sum_hess = 0.0;
    for (size_t b = 0; b < g.max_cuts; ++b)
    {
        lane_d cell_grad;
        lane_d cell_hess;
        gather_cells(g, b, cell_grad, cell_hess, p.active);
        sum_grad += cell_grad;
        sum_hess += cell_hess;
        narrow_lanes(sum_grad, p.left_grad);
        narrow_lanes(sum_hess, p.left_hess);
        p.right_grad = s.real_grad - p.left_grad;
        p.right_hess = s.real_hess - p.left_hess;
        accumulate_direction<true>(acc[b * n_dirs], s, p, rule, parent_score);
        if constexpr (TwoDirs)
        {
            accumulate_direction<false>(acc[(b * n_dirs) + 1], s, p, rule,
                                        parent_score);
        }
    }
}

#ifdef __x86_64__
#define BONSAI_LANE_TARGET __attribute__((target("avx2")))
#else
#define BONSAI_LANE_TARGET
#endif

inline bool lanes_available()
{
#ifdef __x86_64__
    static bool const avx2 = []
    {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2");
    }();
    return avx2;
#else
    return true;
#endif
}

template <bool TwoDirs>
BONSAI_LANE_TARGET void scan_lanes(LaneGroup const &g, LaneRule const &rule,
                                   LaneBest &best)
{
    scan_lanes_body<TwoDirs>(g, rule, best);
}

inline void merge_lanes(LaneBest const &lanes, std::span<feature_id_t const> fids,
                        SplitOutput &best)
{
    for (size_t l = 0; l < fids.size(); ++l)
    {
        if (lanes.gain[l] > best.gain)
        {
            best = {.gain         = lanes.gain[l],
                    .feature_id   = fids[l],
                    .bin_id       = static_cast<bin_id_t>(lanes.bin[l]),
                    .default_left = lanes.left[l] != 0,
                    .valid        = true};
        }
    }
}

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
    merge_lanes(lanes, std::span(g.fids).first(g.n), best);
    g.clear();
}

struct LevelGroup
{
    std::array<feature_id_t, k_scan_lanes> fids{};
    std::array<size_t, k_scan_lanes>       cuts{};
    size_t                                 n        = 0;
    size_t                                 max_cuts = 0;
    bool                                   two_dirs = false;
};

inline bool missing_anywhere(LevelParents const &lp, feature_id_t fid)
{
    return std::ranges::any_of(
        lp.populated, [&lp, fid](size_t p)
        { return !missing_empty(lp.frontier[p].hists[fid].missing()); });
}

inline LevelGroup level_group(LevelParents const &lp, feature_id_t f0, feature_id_t f1)
{
    LevelGroup g;
    for (feature_id_t fid = f0; fid < f1; ++fid)
    {
        g.fids[g.n] = fid;
        g.cuts[g.n] = lp.front().hists[fid].prefix_size();
        g.max_cuts  = std::max(g.max_cuts, g.cuts[g.n]);
        g.two_dirs  = g.two_dirs || missing_anywhere(lp, fid);
        ++g.n;
    }
    return g;
}

template <bool TwoDirs>
BONSAI_LANE_TARGET double accumulate_level(LevelParents const &lp, LevelGroup const &lg,
                                           LaneRule const &rule, lane_d *acc)
{
    double sum_parent_score = 0.0;
    for (size_t const p : lp.populated)
    {
        LaneGroup g;
        g.clear();
        for (size_t l = 0; l < lg.n; ++l)
        {
            g.add(lg.fids[l], lp.frontier[p].hists[lg.fids[l]], lp.totals[p]);
        }
        double const parent_score =
            score(lp.totals[p].sum_grad, lp.totals[p].sum_hess, rule.l1, rule.l2);
        sum_parent_score += parent_score;
        accumulate_lanes_body<TwoDirs>(g, rule, parent_score, acc);
    }
    return sum_parent_score;
}

template <bool TwoDirs>
BONSAI_LANE_TARGET void offer_level(LevelGroup const &lg, lane_d const *acc,
                                    double sum_parent_score, LaneRule const &rule,
                                    LaneBest &best)
{
    constexpr size_t n_dirs = TwoDirs ? 2 : 1;
    for (size_t b = 0; b < lg.max_cuts; ++b)
    {
        lane_i active;
        for (size_t l = 0; l < k_scan_lanes; ++l)
        {
            active[l] = b < lg.cuts[l] ? -1 : 0;
        }
        auto const bin = static_cast<int64_t>(b);
        best.offer(active, acc[b * n_dirs] - sum_parent_score, bin, true,
                   rule.min_gain);
        if constexpr (TwoDirs)
        {
            best.offer(active, acc[(b * n_dirs) + 1] - sum_parent_score, bin, false,
                       rule.min_gain);
        }
    }
}

template <bool TwoDirs>
inline void scan_level_lanes(LevelParents const &lp, LevelGroup const &lg,
                             LaneRule const &rule, LaneBest &best)
{
    static thread_local std::vector<lane_d> acc;
    acc.assign(lg.max_cuts * (TwoDirs ? 2 : 1), lane_d(0.0));
    double const sum_parent_score = accumulate_level<TwoDirs>(lp, lg, rule, acc.data());
    offer_level<TwoDirs>(lg, acc.data(), sum_parent_score, rule, best);
}

SplitOutput scan_level_group(LevelParents const &lp, feature_id_t f0, feature_id_t f1,
                             TreeConfig const &config)
{
    LevelGroup const lg = level_group(lp, f0, f1);
    SplitOutput      best;
    if (lg.max_cuts == 0)
    {
        return best;
    }
    LaneRule const rule = lane_rule(config, 0.0);
    LaneBest       lanes;
    if (lg.two_dirs)
    {
        scan_level_lanes<true>(lp, lg, rule, lanes);
    }
    else
    {
        scan_level_lanes<false>(lp, lg, rule, lanes);
    }
    merge_lanes(lanes, std::span(lg.fids).first(lg.n), best);
    return best;
}

inline bool scans_in_lanes(SplitInput const &input, feature_id_t fid,
                           TreeConfig const &config)
{
    return lanes_available() && monotone_constraint_of(config, fid) == 0 &&
           node_scans(input, fid);
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
    std::vector<size_t>   populated;
    std::vector<HistCell> node_totals(frontier.size());
    for (size_t p = 0; p < frontier.size(); ++p)
    {
        if (!frontier[p].hists.empty())
        {
            populated.push_back(p);
            node_totals[p] = cell_totals(frontier[p]);
        }
    }
    if (populated.empty())
    {
        return {};
    }
    LevelParents const lp{
        .frontier = frontier, .populated = populated, .totals = node_totals};
    feature_id_t const n_features = lp.front().hists.size();
    if (lanes_available())
    {
        size_t const groups = (n_features + k_scan_lanes - 1) / k_scan_lanes;
        std::vector<SplitOutput> per_group(groups);
        parallel::for_each_index(
            groups,
            [&](size_t g)
            {
                auto const f0 = static_cast<feature_id_t>(g * k_scan_lanes);
                auto const f1 = static_cast<feature_id_t>(
                    std::min<size_t>((g + 1) * k_scan_lanes, n_features));
                per_group[g] = scan_level_group(lp, f0, f1, config);
            });
        return reduce_in_feature_order(per_group);
    }
    std::vector<SplitOutput> per_feature(n_features);
    parallel::for_each_index(n_features,
                             [&](size_t fid)
                             {
                                 update_best_for_feature_for_level(
                                     lp, static_cast<feature_id_t>(fid), config,
                                     per_feature[fid]);
                             });
    return reduce_in_feature_order(per_feature);
}

} // namespace bonsai
