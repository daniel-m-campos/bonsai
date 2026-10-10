#include <algorithm>
#include <array>
#include <bit>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include "bonsai/config/tree_config.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/split.hpp"
#include "bonsai/types.hpp"

using namespace bonsai; // NOLINT

TEST_CASE("HistogramNodeSplitFinder: picks the obvious cut on a single feature",
          "[split][basic]")
{
    // 3 cells: bins 0,1 real; bin 2 missing (zero).
    Histogram h{3};
    h.add(0, -1.0, 1.0);
    h.add(1, +1.0, 1.0);

    TreeConfig cfg{.lambda_l2 = 1.0F, .min_data_in_leaf = 0};
    SplitInput node{.hists = {}, .rows = {}};
    node.hists.push_back(std::move(h));
    SplitOutput const s = HistogramNodeSplitFinder::find(node, cfg);

    REQUIRE(s.valid);
    CHECK(s.feature_id == feature_id_t{0});
    CHECK(s.bin_id == bin_id_t{0});

    NodeTotals const total    = node.totals();
    double const     expected = score(-1.0, 1.0, cfg.lambda_l2) +
                            score(+1.0, 1.0, cfg.lambda_l2) -
                            score(total.sum_grad, total.sum_hess, cfg.lambda_l2);
    CHECK(s.gain == expected);
}

TEST_CASE("HistogramNodeSplitFinder: picks the best feature across two features",
          "[split][feature]")
{
    // Feature 0: weak split (small gradient swing).
    Histogram h0{3};
    h0.add(0, -0.25, 1.0);
    h0.add(1, +0.25, 1.0);

    // Feature 1: strong split (large gradient swing).
    Histogram h1{3};
    h1.add(0, -2.0, 1.0);
    h1.add(1, +2.0, 1.0);

    TreeConfig cfg{.lambda_l2 = 1.0F, .min_data_in_leaf = 0};
    SplitInput node{.hists = {}, .rows = {}};
    node.hists.push_back(std::move(h0));
    node.hists.push_back(std::move(h1));
    SplitOutput const s = HistogramNodeSplitFinder::find(node, cfg);

    REQUIRE(s.valid);
    CHECK(s.feature_id == feature_id_t{1});
    CHECK(s.bin_id == bin_id_t{0});
}

TEST_CASE("HistogramNodeSplitFinder: missing cell prefers default_left when its grad "
          "pulls left",
          "[split][missing]")
{
    // Real bins symmetric so the cut alone has zero net pull;
    // missing cell's grad matches the left bin's sign, so sending it
    // left (default_left=true) increases |g_left| and hence gain.
    Histogram h{3};
    h.add(0, -1.0, 1.0); // real
    h.add(1, +1.0, 1.0); // real
    h.add(2, -1.0, 1.0); // missing (last)

    TreeConfig cfg{.lambda_l2 = 1.0F, .min_data_in_leaf = 0};
    SplitInput node{.hists = {}, .rows = {}};
    node.hists.push_back(std::move(h));
    SplitOutput const s = HistogramNodeSplitFinder::find(node, cfg);

    REQUIRE(s.valid);
    CHECK(s.default_left == true);
    CHECK(s.bin_id == bin_id_t{0});

    // Expected: g_left = -1 + (-1) = -2, h_left = 2; g_right = +1, h_right = 1.
    NodeTotals const total    = node.totals();
    double const     expected = score(-2.0, 2.0, cfg.lambda_l2) +
                            score(+1.0, 1.0, cfg.lambda_l2) -
                            score(total.sum_grad, total.sum_hess, cfg.lambda_l2);
    CHECK(s.gain == expected);
}

TEST_CASE("HistogramNodeSplitFinder: missing cell prefers default_right when its grad "
          "pulls right",
          "[split][missing]")
{
    // Mirror of the prior test: missing grad matches right bin sign.
    Histogram h{3};
    h.add(0, -1.0, 1.0);
    h.add(1, +1.0, 1.0);
    h.add(2, +1.0, 1.0); // missing

    TreeConfig cfg{.lambda_l2 = 1.0F, .min_data_in_leaf = 0};
    SplitInput node{.hists = {}, .rows = {}};
    node.hists.push_back(std::move(h));
    SplitOutput const s = HistogramNodeSplitFinder::find(node, cfg);

    REQUIRE(s.valid);
    CHECK(s.default_left == false);
    CHECK(s.bin_id == bin_id_t{0});

    // g_left = -1, h_left = 1; g_right = +1 + (+1) = +2, h_right = 2.
    NodeTotals const total    = node.totals();
    double const     expected = score(-1.0, 1.0, cfg.lambda_l2) +
                            score(+2.0, 2.0, cfg.lambda_l2) -
                            score(total.sum_grad, total.sum_hess, cfg.lambda_l2);
    CHECK(s.gain == expected);
}

TEST_CASE(
    "HistogramNodeSplitFinder: returns invalid when no positive-gain split exists",
    "[split][invalid]")
{
    // Two real bins with identical (grad, hess) and missing zero.
    // Best achievable child-score sum equals the unsplit score, so
    // setting node_score to that value drives net gain to zero —
    // never strictly greater than best_split.gain (default 0.0).
    Histogram h{3};
    h.add(0, +0.5, 1.0);
    h.add(1, +0.5, 1.0);

    constexpr float lambda = 1.0F;
    TreeConfig      cfg{.lambda_l2 = lambda, .min_data_in_leaf = 0};
    SplitInput      node{.hists = {}, .rows = {}};
    node.hists.push_back(std::move(h));

    SplitOutput const s = HistogramNodeSplitFinder::find(node, cfg);

    CHECK_FALSE(s.valid);
    CHECK(s.gain == 0.0);
    CHECK(s.feature_id == feature_id_t{0});
    CHECK(s.bin_id == bin_id_t{0});
    CHECK(s.default_left == true);
}

TEST_CASE(
    "HistogramNodeSplitFinder: empty histogram view returns invalid default Split",
    "[split][edge]")
{
    SplitOutput const s =
        HistogramNodeSplitFinder::find(SplitInput{}, TreeConfig{.min_data_in_leaf = 0});

    CHECK_FALSE(s.valid);
    CHECK(s.gain == 0.0);
    CHECK(s.feature_id == feature_id_t{0});
    CHECK(s.bin_id == bin_id_t{0});
    CHECK(s.default_left == true);
}

TEST_CASE("HistogramNodeSplitFinder: skips the degenerate all-real-on-left cut",
          "[split][missing][regression]")
{
    // Regression: pre-fix, the cut after the last real bin with
    // default_left=false sent every real row left and only the
    // missing rows right. With a tiny-hess, big-grad missing cell,
    // the right side score (g_m^2 / (h_m + lambda)) blew up and beat
    // the real cut. The genuine cut at b=0 has gain ~0 (g_left=+1
    // and g_right=-1 give symmetric scores ~ 0.5 + 0.5 = 1.0). The
    // pre-fix degenerate cut yielded ~25 / 1.01 ≈ 24.75. So if the
    // bug returns, default_left=false on bin_id=1 wins.
    Histogram h{3};
    h.add(0, +1.0, 1.0);
    h.add(1, -1.0, 1.0);
    h.add(2, +5.0, 0.01); // missing: tiny hess, big grad

    TreeConfig cfg{.lambda_l2 = 1.0F, .min_data_in_leaf = 0};
    SplitInput node{.hists = {}, .rows = {}};
    node.hists.push_back(std::move(h));
    SplitOutput const s = HistogramNodeSplitFinder::find(node, cfg);

    REQUIRE(s.valid);
    CHECK(s.bin_id == bin_id_t{0});
}

TEST_CASE(
    "HistogramNodeSplitFinder: min_child_hess rejects splits with too-light children",
    "[split][min_child_hess]")
{
    // All real bins zero, only the missing cell carries mass.
    // Without a min-hess guard, default_left=true sends the missing
    // cell left, scores g_m^2 / (h_m + lambda) > 0, and registers a
    // "split" that puts every row on one side. The default
    // min_child_hess = 1.0 should reject this: right child has hess
    // 0 < 1.0.
    // Real bins cleanly split (opposing grad). Bin 0's hess is tiny so
    // the only candidate cut puts a sub-1.0-hess child on the left.
    // min_child_hess=1.0 rejects; disabling the guard accepts the same
    // genuine cut.
    Histogram h{4};
    h.add(0, -1.0, 0.5); // real, light
    h.add(1, +1.0, 2.0); // real, heavy
    // bin 2: zero, missing: zero

    SplitInput node{.hists = {}, .rows = {}};
    node.hists.push_back(std::move(h));

    SplitOutput const guarded =
        HistogramNodeSplitFinder::find(node, TreeConfig{.min_data_in_leaf = 0});
    CHECK_FALSE(guarded.valid);

    SplitOutput const unguarded = HistogramNodeSplitFinder::find(
        node,
        TreeConfig{.min_child_hess = 0.0F, .lambda_l2 = 1.0F, .min_data_in_leaf = 0});
    REQUIRE(unguarded.valid);
    CHECK(unguarded.bin_id == bin_id_t{0});
}

TEST_CASE("HistogramNodeSplitFinder: lambda_l2 changes the chosen cut",
          "[split][regularization]")
{
    // Three real bins, one missing (zero). Two candidate cuts:
    //   bin_id=0 — left side has tiny hessian, big gradient; high
    //              unregularized score, sensitive to lambda.
    //   bin_id=1 — left side balanced, less sensitive to lambda.
    // Tuned so cut 0 wins at lambda=0 and cut 1 wins at lambda=10.
    Histogram h{4};
    h.add(0, +2.0, 0.25); // bin 0: tiny hess, big grad
    h.add(1, +1.0, 4.0);  // bin 1
    h.add(2, -3.0, 4.0);  // bin 2
    // bin 3 missing, zero

    SplitInput node{.hists = {}, .rows = {}};
    node.hists.push_back(std::move(h));

    SplitOutput const lo = HistogramNodeSplitFinder::find(
        node,
        TreeConfig{.min_child_hess = 0.0F, .lambda_l2 = 0.0F, .min_data_in_leaf = 0});
    SplitOutput const hi = HistogramNodeSplitFinder::find(
        node,
        TreeConfig{.min_child_hess = 0.0F, .lambda_l2 = 10.0F, .min_data_in_leaf = 0});

    REQUIRE(lo.valid);
    REQUIRE(hi.valid);
    CHECK(lo.bin_id == bin_id_t{0});
    CHECK(hi.bin_id == bin_id_t{1});
}

// The "obvious cut" fixture: 3-bin histogram, opposing grads at bins 0 and 1,
// missing bin zero. Only one valid cut (bin_id=0); its gain equals
//   score(-1, 1, 1) + score(+1, 1, 1) - score(0, 2, 1) = 0.5 + 0.5 - 0 = 1.0.
namespace
{

SplitInput make_obvious_node()
{
    Histogram h{3};
    h.add(0, -1.0, 1.0);
    h.add(1, +1.0, 1.0);
    SplitInput node{.hists = {}, .rows = {}};
    node.hists.push_back(std::move(h));
    return node;
}

} // namespace

TEST_CASE("HistogramNodeSplitFinder: min_gain_to_split rejects sub-threshold cuts",
          "[split][min_gain_to_split]")
{
    auto       node = make_obvious_node();
    TreeConfig cfg{.min_child_hess    = 0.0F,
                   .min_gain_to_split = 1.5F,
                   .lambda_l2         = 1.0F,
                   .min_data_in_leaf  = 0};

    SplitOutput const s = HistogramNodeSplitFinder::find(node, cfg);

    // Only candidate has gain 1.0 < 1.5 → splitter should reject.
    CHECK_FALSE(s.valid);
    CHECK(s.gain == 0.0);
}

TEST_CASE(
    "HistogramNodeSplitFinder: min_gain_to_split accepts at-or-above-threshold cuts",
    "[split][min_gain_to_split]")
{
    auto       node = make_obvious_node();
    TreeConfig cfg{.min_child_hess    = 0.0F,
                   .min_gain_to_split = 0.5F,
                   .lambda_l2         = 1.0F,
                   .min_data_in_leaf  = 0};

    SplitOutput const s = HistogramNodeSplitFinder::find(node, cfg);

    REQUIRE(s.valid);
    CHECK(s.bin_id == bin_id_t{0});
    CHECK(s.gain == Catch::Approx(1.0).epsilon(1e-9));
}

TEST_CASE("HistogramNodeSplitFinder: min_data_in_leaf rejects when parent has "
          "<2*threshold rows",
          "[split][min_data_in_leaf]")
{
    auto node = make_obvious_node();
    node.rows = std::vector<row_id_t>(9, 0); // size only; values don't matter

    TreeConfig cfg{
        .min_child_hess = 0.0F, .lambda_l2 = 1.0F, .min_data_in_leaf = 5}; // 9 < 2*5

    SplitOutput const s = HistogramNodeSplitFinder::find(node, cfg);

    CHECK_FALSE(s.valid);
    CHECK(s.gain == 0.0);
}

TEST_CASE("HistogramNodeSplitFinder: min_data_in_leaf accepts when parent has "
          ">=2*threshold rows",
          "[split][min_data_in_leaf]")
{
    auto node = make_obvious_node();
    node.rows = std::vector<row_id_t>(100, 0);

    TreeConfig cfg{
        .min_child_hess = 0.0F, .lambda_l2 = 1.0F, .min_data_in_leaf = 5}; // 100 >= 2*5

    SplitOutput const s = HistogramNodeSplitFinder::find(node, cfg);

    REQUIRE(s.valid);
    CHECK(s.bin_id == bin_id_t{0});
}

namespace
{

struct ScalarPrefix
{
    double   left_grad;
    double   left_hess;
    double   real_grad;
    double   real_hess;
    HistCell missing;
};

struct ScalarChildren
{
    double grad_left;
    double hess_left;
    double grad_right;
    double hess_right;
};

ScalarChildren scalar_children(ScalarPrefix const &p, bool default_left)
{
    return {.grad_left = p.left_grad + (default_left ? p.missing.sum_grad : 0.0),
            .hess_left = p.left_hess + (default_left ? p.missing.sum_hess : 0.0),
            .grad_right =
                (p.real_grad - p.left_grad) + (default_left ? 0.0 : p.missing.sum_grad),
            .hess_right = (p.real_hess - p.left_hess) +
                          (default_left ? 0.0 : p.missing.sum_hess)};
}

void scalar_offer(SplitOutput &best, ScalarChildren const &c, feature_id_t fid,
                  bin_id_t b, bool default_left, TreeConfig const &cfg,
                  double node_score)
{
    if (c.hess_left < cfg.min_child_hess || c.hess_right < cfg.min_child_hess)
    {
        return;
    }
    double const gain =
        score(c.grad_left, c.hess_left, cfg.lambda_l1, cfg.lambda_l2) +
        score(c.grad_right, c.hess_right, cfg.lambda_l1, cfg.lambda_l2) - node_score;
    if (gain > best.gain && gain >= cfg.min_gain_to_split)
    {
        best = {.gain         = gain,
                .feature_id   = fid,
                .bin_id       = b,
                .default_left = default_left,
                .valid        = true};
    }
}

bool scalar_scannable(SplitInput const &node, feature_id_t fid)
{
    return node.hists[fid].size() != 0 &&
           (node.allowed.empty() || node.allowed[fid] != 0);
}

void scalar_scan_feature(SplitInput const &node, feature_id_t fid,
                         HistCell const &totals, TreeConfig const &cfg,
                         double node_score, SplitOutput &best)
{
    Histogram const &hist       = node.hists[fid];
    HistCell const  &missing    = hist.missing();
    bool const       no_missing = missing.sum_grad == 0.0F && missing.sum_hess == 0.0F;
    ScalarPrefix     p{.left_grad = 0.0,
                       .left_hess = 0.0,
                       .real_grad = totals.sum_grad - missing.sum_grad,
                       .real_hess = totals.sum_hess - missing.sum_hess,
                       .missing   = missing};
    bin_id_t         b = 0;
    for (HistCell const &cell : hist.cut_cells())
    {
        p.left_grad += cell.sum_grad;
        p.left_hess += cell.sum_hess;
        scalar_offer(best, scalar_children(p, true), fid, b, true, cfg, node_score);
        if (!no_missing)
        {
            scalar_offer(best, scalar_children(p, false), fid, b, false, cfg,
                         node_score);
        }
        ++b;
    }
}

SplitOutput scalar_scan(SplitInput const &node, TreeConfig const &cfg)
{
    NodeTotals const t = node.totals();
    HistCell const   totals{.sum_grad = static_cast<float>(t.sum_grad),
                            .sum_hess = static_cast<float>(t.sum_hess)};
    double const     node_score =
        score(totals.sum_grad, totals.sum_hess, cfg.lambda_l1, cfg.lambda_l2);
    SplitOutput best;
    for (feature_id_t fid = 0; fid < node.hists.size(); ++fid)
    {
        if (scalar_scannable(node, fid))
        {
            scalar_scan_feature(node, fid, totals, cfg, node_score, best);
        }
    }
    return best;
}

SplitInput random_node(std::mt19937 &rng)
{
    size_t const n_features   = std::uniform_int_distribution<size_t>(1, 40)(rng);
    size_t const rows         = std::uniform_int_distribution<size_t>(1, 400)(rng);
    bool const   with_missing = rng() % 2 == 0;
    SplitInput   node{.hists = {}, .rows = {}};
    std::normal_distribution<float> grad(0.0F, 1.0F);
    for (size_t f = 0; f < n_features; ++f)
    {
        if (rng() % 8 == 0)
        {
            node.hists.push_back(Histogram{});
            continue;
        }
        size_t const n_bins = std::uniform_int_distribution<size_t>(2, 256)(rng);
        Histogram    h{n_bins};
        for (size_t r = 0; r < rows; ++r)
        {
            size_t bin = rng() % 4 == 0 ? std::min<size_t>(rng() % 3, n_bins - 2)
                                        : rng() % (n_bins - 1);
            if (with_missing && rng() % 5 == 0)
            {
                bin = n_bins - 1;
            }
            float const hess = rng() % 4 == 0 ? 0.0F : 1.0F;
            h.add(static_cast<bin_id_t>(bin), grad(rng) * (rng() % 2 ? 1.0F : 100.0F),
                  hess);
        }
        node.hists.push_back(std::move(h));
    }
    if (rng() % 3 == 0)
    {
        node.allowed.resize(n_features);
        for (char &a : node.allowed)
        {
            a = static_cast<char>(rng() % 2);
        }
    }
    return node;
}

TreeConfig random_rule(std::mt19937 &rng)
{
    TreeConfig cfg{.min_data_in_leaf = 0};
    cfg.lambda_l1         = rng() % 2 ? 0.0F : 0.7F;
    cfg.lambda_l2         = rng() % 3 ? 1.0F : 0.0F;
    cfg.min_child_hess    = rng() % 2 ? 0.0F : 2.0F;
    cfg.min_gain_to_split = rng() % 3 ? 0.0F : 0.25F;
    return cfg;
}

bool same_split(SplitOutput const &a, SplitOutput const &b)
{
    return std::bit_cast<uint64_t>(a.gain) == std::bit_cast<uint64_t>(b.gain) &&
           a.feature_id == b.feature_id && a.bin_id == b.bin_id &&
           a.default_left == b.default_left && a.valid == b.valid;
}

} // namespace

// INVARIANT: lane-scan-matches-scalar-scan
TEST_CASE("HistogramNodeSplitFinder: lanes match the scalar scan to the bit",
          "[split][edge]")
{
    std::mt19937 rng(2026);
    size_t       valid = 0;
    for (int trial = 0; trial < 400; ++trial)
    {
        SplitInput const           node = random_node(rng);
        TreeConfig const           cfg  = random_rule(rng);
        SplitOutput const          want = scalar_scan(node, cfg);
        SplitOutput const          got  = HistogramNodeSplitFinder::find(node, cfg);
        std::array<SplitOutput, 1> batched{};
        HistogramNodeSplitFinder::find_parallel({&node, 1}, cfg, batched);
        REQUIRE(same_split(got, want));
        REQUIRE(same_split(batched[0], want));
        valid += want.valid ? 1 : 0;
    }
    CHECK(valid > 300);
}
