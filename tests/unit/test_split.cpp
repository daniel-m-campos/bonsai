#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <utility>
#include <vector>

#include "bonsai/config/tree_config.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/split.hpp"
#include "bonsai/types.hpp"

using namespace bonsai; // NOLINT

TEST_CASE("HistogramSplitFinder: picks the obvious cut on a single feature",
          "[split][basic]")
{
    // 3 cells: bins 0,1 real; bin 2 missing (zero).
    Histogram h{3};
    h.add(0, -1.0, 1.0);
    h.add(1, +1.0, 1.0);

    TreeConfig cfg{.lambda_l2 = 1.0F, .min_data_in_leaf = 0};
    SplitNode node{.hists = {}, .rows = {}, .grad = 0.0, .hess = 2.0};
    node.hists.push_back(std::move(h));
    Split const s = HistogramSplitFinder::find(node, cfg);

    REQUIRE(s.valid);
    CHECK(s.feature_id == feature_id_t{0});
    CHECK(s.bin_id == bin_id_t{0});

    double const expected = score(-1.0, 1.0, cfg.lambda_l2) +
                            score(+1.0, 1.0, cfg.lambda_l2) -
                            score(node.grad, node.hess, cfg.lambda_l2);
    CHECK(s.gain == expected);
}

TEST_CASE("HistogramSplitFinder: picks the best feature across two features",
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
    SplitNode node{.hists = {}, .rows = {}, .grad = 0.0, .hess = 2.0};
    node.hists.push_back(std::move(h0));
    node.hists.push_back(std::move(h1));
    Split const s = HistogramSplitFinder::find(node, cfg);

    REQUIRE(s.valid);
    CHECK(s.feature_id == feature_id_t{1});
    CHECK(s.bin_id == bin_id_t{0});
}

TEST_CASE(
    "HistogramSplitFinder: missing cell prefers default_left when its grad pulls left",
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
    SplitNode node{.hists = {}, .rows = {}, .grad = -1.0, .hess = 3.0};
    node.hists.push_back(std::move(h));
    Split const s = HistogramSplitFinder::find(node, cfg);

    REQUIRE(s.valid);
    CHECK(s.default_left == true);
    CHECK(s.bin_id == bin_id_t{0});

    // Expected: g_left = -1 + (-1) = -2, h_left = 2; g_right = +1, h_right = 1.
    double const expected = score(-2.0, 2.0, cfg.lambda_l2) +
                            score(+1.0, 1.0, cfg.lambda_l2) -
                            score(node.grad, node.hess, cfg.lambda_l2);
    CHECK(s.gain == expected);
}

TEST_CASE("HistogramSplitFinder: missing cell prefers default_right when its grad "
          "pulls right",
          "[split][missing]")
{
    // Mirror of the prior test: missing grad matches right bin sign.
    Histogram h{3};
    h.add(0, -1.0, 1.0);
    h.add(1, +1.0, 1.0);
    h.add(2, +1.0, 1.0); // missing

    TreeConfig cfg{.lambda_l2 = 1.0F, .min_data_in_leaf = 0};
    SplitNode node{.hists = {}, .rows = {}, .grad = +1.0, .hess = 3.0};
    node.hists.push_back(std::move(h));
    Split const s = HistogramSplitFinder::find(node, cfg);

    REQUIRE(s.valid);
    CHECK(s.default_left == false);
    CHECK(s.bin_id == bin_id_t{0});

    // g_left = -1, h_left = 1; g_right = +1 + (+1) = +2, h_right = 2.
    double const expected = score(-1.0, 1.0, cfg.lambda_l2) +
                            score(+2.0, 2.0, cfg.lambda_l2) -
                            score(node.grad, node.hess, cfg.lambda_l2);
    CHECK(s.gain == expected);
}

TEST_CASE("HistogramSplitFinder: returns invalid when no positive-gain split exists",
          "[split][invalid]")
{
    // Two real bins with identical (grad, hess) and missing zero.
    // Best achievable child-score sum equals the unsplit score, so
    // setting node_score to that value drives net gain to zero —
    // never strictly greater than best_split.gain (default 0.0).
    Histogram h{3};
    h.add(0, +0.5, 1.0);
    h.add(1, +0.5, 1.0);

    float constexpr lambda = 1.0F;
    TreeConfig cfg{.lambda_l2 = lambda, .min_data_in_leaf = 0};
    SplitNode node{.hists = {}, .rows = {}, .grad = 1.0, .hess = 2.0};
    node.hists.push_back(std::move(h));

    Split const s = HistogramSplitFinder::find(node, cfg);

    CHECK_FALSE(s.valid);
    CHECK(s.gain == 0.0);
    CHECK(s.feature_id == feature_id_t{0});
    CHECK(s.bin_id == bin_id_t{0});
    CHECK(s.default_left == true);
}

TEST_CASE("HistogramSplitFinder: empty histogram view returns invalid default Split",
          "[split][edge]")
{
    Split const s =
        HistogramSplitFinder::find(SplitNode{}, TreeConfig{.min_data_in_leaf = 0});

    CHECK_FALSE(s.valid);
    CHECK(s.gain == 0.0);
    CHECK(s.feature_id == feature_id_t{0});
    CHECK(s.bin_id == bin_id_t{0});
    CHECK(s.default_left == true);
}

TEST_CASE("HistogramSplitFinder: skips the degenerate all-real-on-left cut",
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
    SplitNode node{.hists = {}, .rows = {}, .grad = +5.0, .hess = 2.01};
    node.hists.push_back(std::move(h));
    Split const s = HistogramSplitFinder::find(node, cfg);

    REQUIRE(s.valid);
    CHECK(s.bin_id == bin_id_t{0});
}

TEST_CASE("HistogramSplitFinder: min_child_hess rejects splits with too-light children",
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

    SplitNode node{.hists = {}, .rows = {}, .grad = 0.0, .hess = 2.5};
    node.hists.push_back(std::move(h));

    Split const guarded =
        HistogramSplitFinder::find(node, TreeConfig{.min_data_in_leaf = 0});
    CHECK_FALSE(guarded.valid);

    Split const unguarded = HistogramSplitFinder::find(
        node,
        TreeConfig{.min_child_hess = 0.0F, .lambda_l2 = 1.0F, .min_data_in_leaf = 0});
    REQUIRE(unguarded.valid);
    CHECK(unguarded.bin_id == bin_id_t{0});
}

TEST_CASE("HistogramSplitFinder: lambda_l2 changes the chosen cut",
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

    SplitNode node{.hists = {}, .rows = {}, .grad = 0.0, .hess = 8.25};
    node.hists.push_back(std::move(h));

    Split const lo = HistogramSplitFinder::find(
        node,
        TreeConfig{.min_child_hess = 0.0F, .lambda_l2 = 0.0F, .min_data_in_leaf = 0});
    Split const hi = HistogramSplitFinder::find(
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

SplitNode make_obvious_node()
{
    Histogram h{3};
    h.add(0, -1.0, 1.0);
    h.add(1, +1.0, 1.0);
    SplitNode node{.hists = {}, .rows = {}, .grad = 0.0, .hess = 2.0};
    node.hists.push_back(std::move(h));
    return node;
}

} // namespace

TEST_CASE("HistogramSplitFinder: min_gain_to_split rejects sub-threshold cuts",
          "[split][min_gain_to_split]")
{
    auto node = make_obvious_node();
    TreeConfig cfg{.min_child_hess    = 0.0F,
                   .min_gain_to_split = 1.5F,
                   .lambda_l2         = 1.0F,
                   .min_data_in_leaf  = 0};

    Split const s = HistogramSplitFinder::find(node, cfg);

    // Only candidate has gain 1.0 < 1.5 → splitter should reject.
    CHECK_FALSE(s.valid);
    CHECK(s.gain == 0.0);
}

TEST_CASE("HistogramSplitFinder: min_gain_to_split accepts at-or-above-threshold cuts",
          "[split][min_gain_to_split]")
{
    auto node = make_obvious_node();
    TreeConfig cfg{.min_child_hess    = 0.0F,
                   .min_gain_to_split = 0.5F,
                   .lambda_l2         = 1.0F,
                   .min_data_in_leaf  = 0};

    Split const s = HistogramSplitFinder::find(node, cfg);

    REQUIRE(s.valid);
    CHECK(s.bin_id == bin_id_t{0});
    CHECK(s.gain == Catch::Approx(1.0).epsilon(1e-9));
}

TEST_CASE(
    "HistogramSplitFinder: min_data_in_leaf rejects when parent has <2*threshold rows",
    "[split][min_data_in_leaf]")
{
    auto node = make_obvious_node();
    node.rows = std::vector<row_id_t>(9, 0); // size only; values don't matter

    TreeConfig cfg{
        .min_child_hess = 0.0F, .lambda_l2 = 1.0F, .min_data_in_leaf = 5}; // 9 < 2*5

    Split const s = HistogramSplitFinder::find(node, cfg);

    CHECK_FALSE(s.valid);
    CHECK(s.gain == 0.0);
}

TEST_CASE(
    "HistogramSplitFinder: min_data_in_leaf accepts when parent has >=2*threshold rows",
    "[split][min_data_in_leaf]")
{
    auto node = make_obvious_node();
    node.rows = std::vector<row_id_t>(100, 0);

    TreeConfig cfg{
        .min_child_hess = 0.0F, .lambda_l2 = 1.0F, .min_data_in_leaf = 5}; // 100 >= 2*5

    Split const s = HistogramSplitFinder::find(node, cfg);

    REQUIRE(s.valid);
    CHECK(s.bin_id == bin_id_t{0});
}
