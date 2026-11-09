#include "bonsai/split.hpp"
#include "bonsai/config/tree_config.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/parallel.hpp"
#include "bonsai/types.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
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

struct LevelParents
{
    FrontierInput           frontier;
    std::span<size_t const> populated;

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
        auto const    &hist    = lp.frontier[p].hists[fid];
        auto const    &missing = hist.missing();
        HistCell const totals  = hist.fill_prefix(std::span(&prefix[p, 0], n_bins));
        parent_score[p] =
            score(totals.sum_grad, totals.sum_hess, config.lambda_l1, config.lambda_l2);
        sum_parent_score += parent_score[p];
        real_grad[p]      = totals.sum_grad - missing.sum_grad;
        real_hess[p]      = totals.sum_hess - missing.sum_hess;
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
inline constexpr int      k_unconstrained = 0;

using lane_d = double __attribute__((ext_vector_type(k_scan_lanes)));
using lane_i = int64_t __attribute__((ext_vector_type(k_scan_lanes)));

struct LaneRule
{
    double l1;
    double l2;
    double min_hess;
    double min_gain;
    double lo = -std::numeric_limits<double>::infinity();
    double hi = std::numeric_limits<double>::infinity();
};

inline LaneRule lane_rule(TreeConfig const &config)
{
    return {.l1       = config.lambda_l1,
            .l2       = config.lambda_l2,
            .min_hess = config.min_child_hess,
            .min_gain = config.min_gain_to_split};
}

inline LaneRule node_rule(SplitInput const &input, TreeConfig const &config)
{
    LaneRule rule = lane_rule(config);
    rule.lo       = input.lo;
    rule.hi       = input.hi;
    return rule;
}

struct LaneSums
{
    lane_d real_grad;
    lane_d real_hess;
    lane_d miss_grad;
    lane_d miss_hess;
    lane_d node_score;
};

struct LaneGroup
{
    std::array<HistCell const *, k_scan_lanes> cells{};
    std::array<cell_view_t, k_scan_lanes>      after_cuts{};
    std::array<size_t, k_scan_lanes>           cuts{};
    std::array<feature_id_t, k_scan_lanes>     fids{};
    lane_d                                     direction = 0.0;
    size_t                                     n         = 0;
    size_t                                     max_cuts  = 0;
    bool                                       two_dirs  = false;
    bool                                       bounded   = false;

    void clear()
    {
        cells.fill(&k_idle_cell);
        after_cuts.fill({});
        cuts.fill(0);
        direction = 0.0;
        n         = 0;
        max_cuts  = 0;
        two_dirs  = false;
        bounded   = false;
    }

    bool full() const
    {
        return n == k_scan_lanes;
    }

    size_t shared_cuts() const
    {
        return *std::ranges::min_element(cuts);
    }

    void add(feature_id_t fid, Histogram const &hist, int monotone)
    {
        cell_view_t const all = hist.all_cells();
        cells[n]              = all.data();
        after_cuts[n]         = all.subspan(hist.prefix_size());
        cuts[n]               = hist.prefix_size();
        fids[n]               = fid;
        max_cuts              = std::max(max_cuts, cuts[n]);
        two_dirs              = two_dirs || !missing_empty(hist.missing());
        direction[n]          = monotone;
        bounded               = bounded || monotone != 0;
        ++n;
    }
};

// perf: The running left sums are stored per cut so a feature's totals fall
// out of the pass that scores it; summing its cells in a pass of their own
// first cost the M2 find +20 to +23% at 20k x 2000.
struct LeftSums
{
    lane_d grad;
    lane_d hess;
};

inline LeftSums *left_sums_buffer(size_t cuts)
{
    static thread_local std::vector<LeftSums> buffer;
    if (buffer.size() < cuts)
    {
        buffer.resize(cuts);
    }
    return buffer.data();
}

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
    lane_d grad_left;
    lane_d hess_left;
    lane_d grad_right;
    lane_d hess_right;
    lane_i feasible;
    lane_d children;
};

template <bool DefaultLeft>
[[gnu::always_inline]] inline void
lane_candidate(LaneSums const &s, LanePrefix const &p, LaneRule const &rule,
               LaneCandidate &c)
{
    lane_d const none = 0.0;
    c.grad_left       = p.left_grad + (DefaultLeft ? s.miss_grad : none);
    c.hess_left       = p.left_hess + (DefaultLeft ? s.miss_hess : none);
    c.grad_right      = p.right_grad + (DefaultLeft ? none : s.miss_grad);
    c.hess_right      = p.right_hess + (DefaultLeft ? none : s.miss_hess);
    c.feasible =
        p.active & ~((c.hess_left < rule.min_hess) | (c.hess_right < rule.min_hess));
    lane_children_score(c.grad_left, c.hess_left, c.grad_right, c.hess_right, rule.l1,
                        rule.l2, c.children);
}

[[gnu::always_inline]] inline void lane_bounded_weight(lane_d const &g, lane_d const &h,
                                                       LaneRule const &rule, lane_d &w)
{
    lane_d t;
    lane_l1_thresholded(g, rule.l1, t);
    lane_d const d    = h + rule.l2;
    lane_d const free = d > 0.0 ? -t / d : lane_d(0.0);
    w = free < rule.lo ? lane_d(rule.lo) : (rule.hi < free ? lane_d(rule.hi) : free);
}

[[gnu::always_inline]] inline void
keep_direction(LaneCandidate &c, lane_d const &direction, LaneRule const &rule)
{
    lane_d w_left;
    lane_d w_right;
    lane_bounded_weight(c.grad_left, c.hess_left, rule, w_left);
    lane_bounded_weight(c.grad_right, c.hess_right, rule, w_right);
    c.feasible &= ~(direction * (w_right - w_left) < 0.0);
}

template <bool DefaultLeft, bool Bounded>
[[gnu::always_inline]] inline void
offer_direction(LaneBest &best, LaneGroup const &g, LaneSums const &s,
                LanePrefix const &p, LaneRule const &rule, int64_t b)
{
    LaneCandidate c;
    lane_candidate<DefaultLeft>(s, p, rule, c);
    if constexpr (Bounded)
    {
        keep_direction(c, g.direction, rule);
    }
    best.offer(c.feasible, c.children - s.node_score, b, DefaultLeft, rule.min_gain);
}

template <bool AllLanes>
[[gnu::always_inline]] inline void gather_cells(LaneGroup const &g, size_t b,
                                                lane_d &cell_grad, lane_d &cell_hess)
{
    for (size_t l = 0; l < k_scan_lanes; ++l)
    {
        bool const     in   = AllLanes || b < g.cuts[l];
        HistCell const cell = g.cells[l][in ? b : 0];
        cell_grad[l]        = in ? cell.sum_grad : 0.0F;
        cell_hess[l]        = in ? cell.sum_hess : 0.0F;
    }
}

template <bool AllLanes>
[[gnu::always_inline]] inline void
lanes_with_cut(std::array<size_t, k_scan_lanes> const &cuts, size_t b, lane_i &active)
{
    for (size_t l = 0; l < k_scan_lanes; ++l)
    {
        active[l] = AllLanes || b < cuts[l] ? -1 : 0;
    }
}

struct CutRange
{
    size_t begin;
    size_t end;
};

template <bool AllLanes>
[[gnu::always_inline]] inline void add_left_sums(LaneGroup const &g, CutRange range,
                                                 LeftSums *left, LeftSums &run)
{
    for (size_t b = range.begin; b < range.end; ++b)
    {
        lane_d cell_grad;
        lane_d cell_hess;
        gather_cells<AllLanes>(g, b, cell_grad, cell_hess);
        run.grad += cell_grad;
        run.hess += cell_hess;
        left[b] = run;
    }
}

// perf: Cuts every lane holds gather with no per-lane test; the tested gather
// compiles to a branch per lane on x86, 87 to 125 instructions a cut.
[[gnu::always_inline]] inline void store_left_sums(LaneGroup const &g, LeftSums *left,
                                                   LeftSums &all_cuts)
{
    size_t const shared = g.shared_cuts();
    all_cuts            = {};
    add_left_sums<true>(g, {.begin = 0, .end = shared}, left, all_cuts);
    add_left_sums<false>(g, {.begin = shared, .end = g.max_cuts}, left, all_cuts);
}

inline void lane_sums(LaneGroup const &g, LeftSums const &all_cuts,
                      LaneRule const &rule, LaneSums &s)
{
    s = {};
    for (size_t l = 0; l < g.n; ++l)
    {
        double grad = all_cuts.grad[l];
        double hess = all_cuts.hess[l];
        for (HistCell const &cell : g.after_cuts[l])
        {
            grad += cell.sum_grad;
            hess += cell.sum_hess;
        }
        HistCell const  totals{.sum_grad = static_cast<float>(grad),
                               .sum_hess = static_cast<float>(hess)};
        HistCell const &missing = g.after_cuts[l].back();
        s.real_grad[l]          = totals.sum_grad - missing.sum_grad;
        s.real_hess[l]          = totals.sum_hess - missing.sum_hess;
        s.miss_grad[l]          = missing.sum_grad;
        s.miss_hess[l]          = missing.sum_hess;
        s.node_score[l] = score(totals.sum_grad, totals.sum_hess, rule.l1, rule.l2);
    }
}

[[gnu::always_inline]] inline void sum_lanes(LaneGroup const &g, LaneRule const &rule,
                                             LeftSums *left, LaneSums &s)
{
    LeftSums all_cuts;
    store_left_sums(g, left, all_cuts);
    lane_sums(g, all_cuts, rule, s);
}

template <bool TwoDirs, bool Bounded, bool AllLanes>
[[gnu::always_inline]] inline void
offer_cuts(LaneGroup const &g, LaneSums const &s, LeftSums const *left,
           LaneRule const &rule, CutRange range, LaneBest &best)
{
    LanePrefix p;
    for (size_t b = range.begin; b < range.end; ++b)
    {
        lanes_with_cut<AllLanes>(g.cuts, b, p.active);
        p.left_grad  = left[b].grad;
        p.left_hess  = left[b].hess;
        p.right_grad = s.real_grad - p.left_grad;
        p.right_hess = s.real_hess - p.left_hess;
        offer_direction<true, Bounded>(best, g, s, p, rule, static_cast<int64_t>(b));
        if constexpr (TwoDirs)
        {
            offer_direction<false, Bounded>(best, g, s, p, rule,
                                            static_cast<int64_t>(b));
        }
    }
}

template <bool TwoDirs, bool Bounded>
[[gnu::always_inline]] inline void scan_lanes_body(LaneGroup const &g,
                                                   LaneRule const &rule, LaneBest &best)
{
    LeftSums *const left = left_sums_buffer(g.max_cuts);
    LaneSums        s;
    sum_lanes(g, rule, left, s);
    size_t const shared = g.shared_cuts();
    offer_cuts<TwoDirs, Bounded, true>(g, s, left, rule, {.begin = 0, .end = shared},
                                       best);
    offer_cuts<TwoDirs, Bounded, false>(g, s, left, rule,
                                        {.begin = shared, .end = g.max_cuts}, best);
}

using lane_f = float __attribute__((ext_vector_type(k_scan_lanes)));

[[gnu::always_inline]] inline void narrow_lanes(lane_d const &wide, lane_d &out)
{
    out = __builtin_convertvector(__builtin_convertvector(wide, lane_f), lane_d);
}

template <bool DefaultLeft>
[[gnu::always_inline]] inline void accumulate_direction(lane_d &acc, LaneSums const &s,
                                                        LanePrefix const &p,
                                                        LaneRule const   &rule)
{
    LaneCandidate c;
    lane_candidate<DefaultLeft>(s, p, rule, c);
    acc += c.feasible ? c.children : s.node_score;
}

template <bool TwoDirs, bool AllLanes>
[[gnu::always_inline]] inline void
accumulate_cuts(LaneGroup const &g, LaneSums const &s, LeftSums const *left,
                LaneRule const &rule, CutRange range, lane_d *acc)
{
    constexpr size_t n_dirs = TwoDirs ? 2 : 1;
    LanePrefix       p;
    for (size_t b = range.begin; b < range.end; ++b)
    {
        lanes_with_cut<AllLanes>(g.cuts, b, p.active);
        narrow_lanes(left[b].grad, p.left_grad);
        narrow_lanes(left[b].hess, p.left_hess);
        p.right_grad = s.real_grad - p.left_grad;
        p.right_hess = s.real_hess - p.left_hess;
        accumulate_direction<true>(acc[b * n_dirs], s, p, rule);
        if constexpr (TwoDirs)
        {
            accumulate_direction<false>(acc[(b * n_dirs) + 1], s, p, rule);
        }
    }
}

template <bool TwoDirs>
[[gnu::always_inline]] inline void
accumulate_lanes_body(LaneGroup const &g, LaneSums const &s, LeftSums const *left,
                      LaneRule const &rule, lane_d *acc)
{
    size_t const shared = g.shared_cuts();
    accumulate_cuts<TwoDirs, true>(g, s, left, rule, {.begin = 0, .end = shared}, acc);
    accumulate_cuts<TwoDirs, false>(g, s, left, rule,
                                    {.begin = shared, .end = g.max_cuts}, acc);
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

template <bool TwoDirs, bool Bounded>
BONSAI_LANE_TARGET void scan_lanes(LaneGroup const &g, LaneRule const &rule,
                                   LaneBest &best)
{
    scan_lanes_body<TwoDirs, Bounded>(g, rule, best);
}

template <bool Bounded>
inline void scan_group(LaneGroup const &g, LaneRule const &rule, LaneBest &best)
{
    if (g.two_dirs)
    {
        scan_lanes<true, Bounded>(g, rule, best);
    }
    else
    {
        scan_lanes<false, Bounded>(g, rule, best);
    }
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
    if (g.bounded)
    {
        scan_group<true>(g, rule, lanes);
    }
    else
    {
        scan_group<false>(g, rule, lanes);
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
        if (lp.front().hists[fid].size() == 0)
        {
            continue;
        }
        g.fids[g.n] = fid;
        g.cuts[g.n] = lp.front().hists[fid].prefix_size();
        g.max_cuts  = std::max(g.max_cuts, g.cuts[g.n]);
        g.two_dirs  = g.two_dirs || missing_anywhere(lp, fid);
        ++g.n;
    }
    return g;
}

template <bool TwoDirs>
BONSAI_LANE_TARGET void accumulate_level(LevelParents const &lp, LevelGroup const &lg,
                                         LaneRule const &rule, lane_d *acc,
                                         lane_d &sum_parent_score)
{
    sum_parent_score     = 0.0;
    LeftSums *const left = left_sums_buffer(lg.max_cuts);
    for (size_t const p : lp.populated)
    {
        LaneGroup g;
        g.clear();
        for (size_t l = 0; l < lg.n; ++l)
        {
            g.add(lg.fids[l], lp.frontier[p].hists[lg.fids[l]], k_unconstrained);
        }
        LaneSums s;
        sum_lanes(g, rule, left, s);
        sum_parent_score += s.node_score;
        accumulate_lanes_body<TwoDirs>(g, s, left, rule, acc);
    }
}

template <bool TwoDirs>
BONSAI_LANE_TARGET void offer_level(LevelGroup const &lg, lane_d const *acc,
                                    lane_d const   &sum_parent_score,
                                    LaneRule const &rule, LaneBest &best)
{
    constexpr size_t n_dirs = TwoDirs ? 2 : 1;
    for (size_t b = 0; b < lg.max_cuts; ++b)
    {
        lane_i active;
        lanes_with_cut<false>(lg.cuts, b, active);
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
    lane_d sum_parent_score;
    accumulate_level<TwoDirs>(lp, lg, rule, acc.data(), sum_parent_score);
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
    LaneRule const rule = lane_rule(config);
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

inline void update_best_for_feature_for_node(SplitInput const &input, feature_id_t fid,
                                             TreeConfig const &config,
                                             SplitOutput      &best)
{
    if (!node_scans(input, fid))
    {
        return;
    }
    auto const    &hist         = input.hists[fid];
    HistCell const node_totals  = hist.totals();
    auto const    &missing_cell = hist.missing();
    double const   node_score   = score(node_totals.sum_grad, node_totals.sum_hess,
                                        config.lambda_l1, config.lambda_l2);
    double const   real_grad    = node_totals.sum_grad - missing_cell.sum_grad;
    double const   real_hess    = node_totals.sum_hess - missing_cell.sum_hess;

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

SplitOutput scan_features(SplitInput const &input, feature_id_t f0, feature_id_t f1,
                          TreeConfig const &config)
{
    SplitOutput best;
    if (!lanes_available())
    {
        for (feature_id_t fid = f0; fid < f1; ++fid)
        {
            update_best_for_feature_for_node(input, fid, config, best);
        }
        return best;
    }
    LaneRule const rule = node_rule(input, config);
    LaneGroup      group;
    group.clear();
    for (feature_id_t fid = f0; fid < f1; ++fid)
    {
        if (!node_scans(input, fid))
        {
            continue;
        }
        group.add(fid, input.hists[fid], monotone_constraint_of(config, fid));
        if (group.full())
        {
            flush_lanes(group, rule, best);
        }
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

size_t lane_groups(size_t n_features)
{
    return (n_features + k_scan_lanes - 1) / k_scan_lanes;
}

size_t scan_ranges(size_t n_features)
{
    size_t const units =
        static_cast<size_t>(parallel::n_threads()) * k_scan_ranges_per_thread;
    return std::max<size_t>(1, std::min(lane_groups(n_features), units));
}

// perf: A range starts on a lane group, so only a node's last range ends in a
// partly filled one; 2000 features in 48 even ranges scanned 288 groups
// against 250.
feature_id_t range_begin(size_t r, size_t ranges, size_t n_features)
{
    size_t const group = r * lane_groups(n_features) / ranges;
    return static_cast<feature_id_t>(std::min(n_features, group * k_scan_lanes));
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
    return scan_features(input, 0, n_features, config);
}

void HistogramNodeSplitFinder::find_parallel(std::span<SplitInput const> nodes,
                                             TreeConfig const           &config,
                                             std::span<SplitOutput>      out)
{
    static thread_local std::vector<SplitOutput> partials;
    size_t const                                 n = nodes.size();
    size_t const ranges = scan_ranges(nodes.empty() ? 0 : nodes.front().hists.size());
    partials.assign(n * ranges, SplitOutput{});
    SplitOutput *const part_ptr = partials.data();
    parallel::for_each_index(n * ranges,
                             [&, part_ptr](size_t u)
                             {
                                 SplitInput const &input = nodes[u / ranges];
                                 if (cannot_split(input, config))
                                 {
                                     return;
                                 }
                                 size_t const nf = input.hists.size();
                                 size_t const r  = u % ranges;
                                 part_ptr[u]     = scan_features(
                                     input, range_begin(r, ranges, nf),
                                     range_begin(r + 1, ranges, nf), config);
                             });
    for (size_t i = 0; i < n; ++i)
    {
        out[i] = reduce_in_feature_order({part_ptr + (i * ranges), ranges});
    }
}

SplitOutput HistogramLevelSplitFinder::find(FrontierInput     frontier,
                                            TreeConfig const &config)
{
    std::vector<size_t> populated;
    for (size_t p = 0; p < frontier.size(); ++p)
    {
        if (!frontier[p].hists.empty())
        {
            populated.push_back(p);
        }
    }
    if (populated.empty())
    {
        return {};
    }
    LevelParents const lp{.frontier = frontier, .populated = populated};
    feature_id_t const n_features = lp.front().hists.size();
    if (lanes_available())
    {
        size_t const             groups = lane_groups(n_features);
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
