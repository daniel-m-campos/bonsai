#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <random>
#include <utility>
#include <vector>

#include "bonsai/config/tree_config.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/split.hpp"
#include "bonsai/types.hpp"
#include "test_split_helpers.hpp"

using namespace bonsai;       // NOLINT
using namespace bonsai::test; // NOLINT

namespace
{

struct ParentScan
{
    std::vector<HistCell> prefix;
    HistCell              missing;
    double                real_grad;
    double                real_hess;
    double                parent_score;
};

ParentScan parent_scan(SplitInput const &parent, feature_id_t fid,
                       HistCell const &totals, TreeConfig const &cfg)
{
    Histogram const &hist    = parent.hists[fid];
    HistCell const  &missing = hist.missing();
    ParentScan       scan{.prefix       = std::vector<HistCell>(hist.prefix_size()),
                          .missing      = missing,
                          .real_grad    = totals.sum_grad - missing.sum_grad,
                          .real_hess    = totals.sum_hess - missing.sum_hess,
                          .parent_score = score(totals.sum_grad, totals.sum_hess,
                                                cfg.lambda_l1, cfg.lambda_l2)};
    hist.fill_prefix(scan.prefix);
    return scan;
}

double children_or_parent(ParentScan const &p, size_t b, bool default_left,
                          TreeConfig const &cfg)
{
    double const left_grad  = p.prefix[b].sum_grad;
    double const left_hess  = p.prefix[b].sum_hess;
    double const right_grad = p.real_grad - left_grad;
    double const right_hess = p.real_hess - left_hess;
    double const grad_left  = left_grad + (default_left ? p.missing.sum_grad : 0.0);
    double const hess_left  = left_hess + (default_left ? p.missing.sum_hess : 0.0);
    double const grad_right = right_grad + (!default_left ? p.missing.sum_grad : 0.0);
    double const hess_right = right_hess + (!default_left ? p.missing.sum_hess : 0.0);
    if (hess_left < cfg.min_child_hess || hess_right < cfg.min_child_hess)
    {
        return p.parent_score;
    }
    return score(grad_left, hess_left, cfg.lambda_l1, cfg.lambda_l2) +
           score(grad_right, hess_right, cfg.lambda_l1, cfg.lambda_l2);
}

SplitOutput scalar_level_feature(FrontierInput frontier, feature_id_t fid,
                                 std::vector<HistCell> const &totals,
                                 TreeConfig const            &cfg)
{
    SplitOutput  best;
    size_t const n_bins = frontier.front().hists[fid].prefix_size();
    if (n_bins == 0)
    {
        return best;
    }
    std::vector<ParentScan> scans;
    double                  sum_parent_score = 0.0;
    bool                    two_dirs         = false;
    for (size_t p = 0; p < frontier.size(); ++p)
    {
        scans.push_back(parent_scan(frontier[p], fid, totals[p], cfg));
        sum_parent_score += scans.back().parent_score;
        HistCell const &m = scans.back().missing;
        two_dirs          = two_dirs || m.sum_grad != 0.0F || m.sum_hess != 0.0F;
    }
    for (size_t b = 0; b < n_bins; ++b)
    {
        for (size_t d = 0; d < (two_dirs ? 2U : 1U); ++d)
        {
            double sum_children_score = 0.0;
            for (ParentScan const &scan : scans)
            {
                sum_children_score += children_or_parent(scan, b, d == 0, cfg);
            }
            offer_split(best, sum_children_score - sum_parent_score, fid,
                        static_cast<bin_id_t>(b), d == 0, cfg);
        }
    }
    return best;
}

SplitOutput scalar_level_scan(FrontierInput frontier, TreeConfig const &cfg)
{
    std::vector<HistCell> totals(frontier.size());
    for (size_t p = 0; p < frontier.size(); ++p)
    {
        totals[p] = cell_totals(frontier[p]);
    }
    SplitOutput best;
    for (feature_id_t fid = 0; fid < frontier.front().hists.size(); ++fid)
    {
        SplitOutput const mine = scalar_level_feature(frontier, fid, totals, cfg);
        if (mine.valid && mine.gain > best.gain)
        {
            best = mine;
        }
    }
    return best;
}

std::vector<size_t> random_bins(std::mt19937 &rng, size_t n_features)
{
    std::vector<size_t> bins(n_features);
    for (size_t &n : bins)
    {
        n = rng() % 8 == 0 ? 1 : std::uniform_int_distribution<size_t>(2, 256)(rng);
    }
    return bins;
}

SplitInput random_parent(std::mt19937 &rng, std::vector<size_t> const &bins,
                         bool with_missing)
{
    size_t const  rows = std::uniform_int_distribution<size_t>(0, 200)(rng);
    SplitInput    node{.hists = {}, .rows = {}};
    HistogramDraw draw{rng, rows, with_missing};
    for (size_t const n_bins : bins)
    {
        node.hists.push_back(draw(n_bins));
    }
    return node;
}

SplitInput zero_parent(std::vector<size_t> const &bins)
{
    SplitInput node{.hists = {}, .rows = {}};
    for (size_t const n_bins : bins)
    {
        node.hists.push_back(Histogram{n_bins});
    }
    return node;
}

std::vector<SplitInput> random_frontier(std::mt19937 &rng)
{
    size_t const n_parents    = std::uniform_int_distribution<size_t>(1, 24)(rng);
    size_t const n_features   = std::uniform_int_distribution<size_t>(1, 20)(rng);
    auto const   bins         = random_bins(rng, n_features);
    bool const   with_missing = rng() % 2 == 0;
    std::vector<SplitInput> frontier;
    for (size_t p = 0; p < n_parents; ++p)
    {
        frontier.push_back(random_parent(rng, bins, with_missing));
    }
    return frontier;
}

} // namespace

TEST_CASE("HistogramLevelSplitFinder: single-parent frontier reproduces node finder",
          "[split][level][single-parent]")
{
    auto                node = obvious_cut_node();
    FrontierInput const span{&node, 1};

    TreeConfig cfg{.lambda_l2 = 1.0F, .min_data_in_leaf = 0};

    SplitOutput const node_s  = HistogramNodeSplitFinder::find(node, cfg);
    SplitOutput const level_s = HistogramLevelSplitFinder::find(span, cfg);

    REQUIRE(node_s.valid);
    REQUIRE(level_s.valid);
    CHECK(level_s.feature_id == node_s.feature_id);
    CHECK(level_s.bin_id == node_s.bin_id);
    CHECK(level_s.default_left == node_s.default_left);
    CHECK(level_s.gain == node_s.gain);
}

TEST_CASE("HistogramLevelSplitFinder: sums per-parent gains across the frontier",
          "[split][level][sum]")
{
    // Two identical parents; each has the obvious cut at bin 0, gain 1.0.
    // Summed level gain = 2.0.
    auto const frontier_arr  = std::array{obvious_cut_node(), obvious_cut_node()};
    FrontierInput const span = frontier_arr;

    TreeConfig cfg{.min_child_hess = 0.0F, .lambda_l2 = 1.0F, .min_data_in_leaf = 0};

    SplitOutput const s = HistogramLevelSplitFinder::find(span, cfg);

    REQUIRE(s.valid);
    CHECK(s.feature_id == feature_id_t{0});
    CHECK(s.bin_id == bin_id_t{0});

    double const per_parent_gain = score(-1.0, 1.0, cfg.lambda_l2) +
                                   score(+1.0, 1.0, cfg.lambda_l2) -
                                   score(0.0, 2.0, cfg.lambda_l2);
    CHECK(s.gain == 2.0 * per_parent_gain);
}

TEST_CASE(
    "HistogramLevelSplitFinder: picks the summed-gain argmax, not per-parent picks",
    "[split][level][argmax]")
{
    // 4-bin histograms: bins 0,1 are cut positions; bin 2 last-real; bin 3 missing.
    //
    // Parent A favors cut at bin 0:
    //   bin0=(-2,1) bin1=(0,1) bin2=(0,1) missing=(0,0)
    //   gain@0 = score(-2,1,1)+score(0,2,1) - score(-2,3,1) = 2 + 0 - 1 = 1
    //   gain@1 = score(-2,2,1)+score(0,1,1) - 1               = 4/3 + 0 - 1 = 1/3
    //
    // Parent B strongly favors cut at bin 1:
    //   bin0=(0,1) bin1=(0,1) bin2=(-4,1) missing=(0,0)
    //   gain@0 = score(0,1,1)+score(-4,2,1) - score(-4,3,1)   = 0 + 16/3 - 4 = 4/3
    //   gain@1 = score(0,2,1)+score(-4,1,1) - 4                = 0 + 8 - 4 = 4
    //
    // Summed: bin0 = 1 + 4/3 = 7/3; bin1 = 1/3 + 4 = 13/3.
    // Level finder must pick bin 1 (sum-argmax), not bin 0 (parent A's pick).

    auto make_a = []
    {
        Histogram h{4};
        h.add(0, -2.0, 1.0);
        h.add(1, 0.0, 1.0);
        h.add(2, 0.0, 1.0);
        return node_of(std::move(h));
    };
    auto make_b = []
    {
        Histogram h{4};
        h.add(0, 0.0, 1.0);
        h.add(1, 0.0, 1.0);
        h.add(2, -4.0, 1.0);
        return node_of(std::move(h));
    };

    auto const          frontier_arr = std::array{make_a(), make_b()};
    FrontierInput const span         = frontier_arr;

    TreeConfig cfg{.min_child_hess = 0.0F, .lambda_l2 = 1.0F, .min_data_in_leaf = 0};

    SplitOutput const s = HistogramLevelSplitFinder::find(span, cfg);

    REQUIRE(s.valid);
    CHECK(s.feature_id == feature_id_t{0});
    CHECK(s.bin_id == bin_id_t{1});

    // bin=1: Parent A children (gL=-2, hL=2), (gR=0, hR=1); B children (gL=0, hL=2),
    // (gR=-4, hR=1). 13/3 mathematically; Approx because the level finder's
    // accumulation order differs.
    double const expected_gain =
        (score(-2.0, 2.0, cfg.lambda_l2) + score(0.0, 1.0, cfg.lambda_l2) -
         score(-2.0, 3.0, cfg.lambda_l2)) +
        (score(0.0, 2.0, cfg.lambda_l2) + score(-4.0, 1.0, cfg.lambda_l2) -
         score(-4.0, 3.0, cfg.lambda_l2));
    CHECK(s.gain == Catch::Approx(expected_gain).epsilon(1e-12));
}

// INVARIANT: infeasible-node-scores-its-parent
// A frontier node whose children would fall under min_child_hess contributes
// its parent score (zero gain) to a level candidate instead of vetoing it. At
// real depths some frontier node is always near-empty, so a veto rejected
// every good deep cut. The device level finder holds the same rule, pinned
// by its own parity case in test_cuda_grower.cpp.
TEST_CASE("HistogramLevelSplitFinder: an infeasible parent scores zero, not a veto",
          "[split][level][min_child_hess][invariant]")
{
    // Parent A and B both have the obvious cut at bin 0, but B's children
    // hess (0.5) is below min_child_hess (0.8): B contributes its parent
    // score and A's gain carries the cut.
    auto make_b = []
    {
        Histogram h{3};
        h.add(0, -1.0, 0.5);
        h.add(1, +1.0, 0.5);
        return node_of(std::move(h));
    };

    auto const          frontier_arr = std::array{obvious_cut_node(), make_b()};
    FrontierInput const span         = frontier_arr;

    TreeConfig cfg{.min_child_hess = 0.8F, .lambda_l2 = 1.0F, .min_data_in_leaf = 0};

    SplitOutput const s = HistogramLevelSplitFinder::find(span, cfg);

    CHECK(s.valid);
    CHECK(s.feature_id == 0);
    CHECK(s.bin_id == 0);
    // Gain = A's children score minus A's parent score only (B nets zero).
    auto const        only_a = std::array{obvious_cut_node()};
    SplitOutput const a_alone =
        HistogramLevelSplitFinder::find(FrontierInput{only_a}, cfg);
    CHECK(s.gain == Catch::Approx(a_alone.gain));
}

TEST_CASE(
    "HistogramLevelSplitFinder: min_gain_to_split rejects sub-threshold summed gain",
    "[split][level][min_gain_to_split]")
{
    // Two identical parents, summed gain = 2.0. Threshold 3.0 rejects.
    auto const frontier_arr  = std::array{obvious_cut_node(), obvious_cut_node()};
    FrontierInput const span = frontier_arr;

    TreeConfig cfg{.min_child_hess    = 0.0F,
                   .min_gain_to_split = 3.0F,
                   .lambda_l2         = 1.0F,
                   .min_data_in_leaf  = 0};

    SplitOutput const s = HistogramLevelSplitFinder::find(span, cfg);

    CHECK_FALSE(s.valid);
    CHECK(s.gain == 0.0);
}

TEST_CASE("HistogramLevelSplitFinder: empty frontier returns invalid",
          "[split][level][edge]")
{
    TreeConfig        cfg{.lambda_l2 = 1.0F, .min_data_in_leaf = 0};
    SplitOutput const s = HistogramLevelSplitFinder::find(FrontierInput{}, cfg);

    CHECK_FALSE(s.valid);
    CHECK(s.gain == 0.0);
}

TEST_CASE("HistogramLevelSplitFinder: degenerate prefix_size == 0 returns invalid",
          "[split][level][edge]")
{
    // n_bins=1 → prefix_size()=0; n_bins guard in the per-feature loop
    // makes this a no-op. No best is ever recorded.
    auto make_parent = []
    {
        Histogram h{1};
        h.add(0, 1.0, 1.0);
        return node_of(std::move(h));
    };

    auto const          frontier_arr = std::array{make_parent(), make_parent()};
    FrontierInput const span         = frontier_arr;

    TreeConfig        cfg{.lambda_l2 = 1.0F, .min_data_in_leaf = 0};
    SplitOutput const s = HistogramLevelSplitFinder::find(span, cfg);

    CHECK_FALSE(s.valid);
    CHECK(s.gain == 0.0);
}

TEST_CASE("gain and leaf math survive an empty node side at lambda_l2 = 0",
          "[split][empty-leaf]")
{
    // An oblivious level mints empty children structurally (the level's
    // shared split applies to a node whose rows all route one way), so
    // G = H = 0 reaches the leaf math; with lambda_l2 = 0 the denominator
    // is 0 and the quotient was 0/0. No evidence means 0, not NaN.
    CHECK(score(0.0, 0.0, 0.0) == 0.0);
    CHECK(score(0.0, 0.0, 0.0, 0.0) == 0.0);
    CHECK(bounded_leaf_weight(0.0, 0.0, 0.0, 0.0, -1e30, 1e30) == 0.0);

    // lambda_l1 cannot resurrect the quotient: an empty leaf's G = 0 lies
    // inside every soft-threshold band, so the numerator is 0 for any l1.
    CHECK(score(0.0, 0.0, 0.5, 0.0) == 0.0);
    CHECK(bounded_leaf_weight(0.0, 0.0, 0.5, 0.0, -1e30, 1e30) == 0.0);
    // A populated leaf whose hessians underflowed to zero (saturated
    // logloss rows) gets 0 rather than the old +-inf step.
    CHECK(bounded_leaf_weight(3.0, 0.0, 0.0, 0.0, -1e30, 1e30) == 0.0);

    // The guard is inert whenever the denominator is positive: identical
    // quotients, so lambda_l2 > 0 models cannot move (the hash gates hold).
    CHECK(score(3.0, 2.0, 1.0) == Catch::Approx((3.0 * 3.0) / 3.0));
    CHECK(bounded_leaf_weight(3.0, 2.0, 0.0, 1.0, -1e30, 1e30) == Catch::Approx(-1.0));
    // An empty side under a monotone bound still lands inside the bound.
    CHECK(bounded_leaf_weight(0.0, 0.0, 0.0, 0.0, 0.5, 2.0) == 0.5);
}

// INVARIANT: zero-parent-moves-no-level-split
// An oblivious level's shared split applies to every frontier node, so a
// node whose rows all routed one way leaves an empty child in the next
// frontier. That child contributes nothing to any candidate: its parent
// score and both children's scores are zero, so the level split chosen
// with it in the frontier is the split chosen without it, gain bit for bit.
TEST_CASE("HistogramLevelSplitFinder: a parent of zeros moves no level split",
          "[split][level][edge]")
{
    std::mt19937 rng(2027);
    size_t       valid = 0;
    for (int trial = 0; trial < 200; ++trial)
    {
        std::vector<SplitInput> frontier = random_frontier(rng);
        TreeConfig const        cfg      = random_rule(rng);
        SplitOutput const       want = HistogramLevelSplitFinder::find(frontier, cfg);
        std::vector<size_t>     bins;
        for (Histogram const &h : frontier.front().hists)
        {
            bins.push_back(h.size());
        }
        frontier.push_back(zero_parent(bins));
        REQUIRE(same_split(HistogramLevelSplitFinder::find(frontier, cfg), want));
        valid += want.valid ? 1 : 0;
    }
    CHECK(valid > 120);
}

// INVARIANT: lane-level-scan-matches-scalar-scan
TEST_CASE("HistogramLevelSplitFinder: lanes match the scalar level scan to the bit",
          "[split][level][edge]")
{
    std::mt19937 rng(2026);
    size_t       valid = 0;
    for (int trial = 0; trial < 300; ++trial)
    {
        std::vector<SplitInput> const frontier = random_frontier(rng);
        TreeConfig const              cfg      = random_rule(rng);
        SplitOutput const             want     = scalar_level_scan(frontier, cfg);
        SplitOutput const got = HistogramLevelSplitFinder::find(frontier, cfg);
        REQUIRE(same_split(got, want));
        valid += want.valid ? 1 : 0;
    }
    CHECK(valid > 200);
}
