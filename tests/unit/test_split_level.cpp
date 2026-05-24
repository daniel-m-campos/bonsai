#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <utility>
#include <vector>

#include "bonsai/config/tree_config.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/split.hpp"
#include "bonsai/types.hpp"

using namespace bonsai; // NOLINT

namespace
{

SplitInput make_node_with_one_hist(Histogram h)
{
    SplitInput node{.hists = {}, .rows = {}};
    node.hists.push_back(std::move(h));
    return node;
}

} // namespace

TEST_CASE("HistogramLevelSplitFinder: single-parent frontier reproduces node finder",
          "[split][level][single-parent]")
{
    // The obvious-cut fixture from test_split_node.cpp.
    Histogram h{3};
    h.add(0, -1.0, 1.0);
    h.add(1, +1.0, 1.0);

    auto node                = make_node_with_one_hist(std::move(h));
    auto const frontier_arr  = std::array{node};
    FrontierInput const span = frontier_arr;

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
    // Two identical parents; each has the obvious cut at bin 0.
    // Per-parent gain = score(-1,1,1) + score(+1,1,1) - score(0,2,1) = 1.0.
    // Summed level gain = 2.0.
    auto make_parent = [] {
        Histogram h{3};
        h.add(0, -1.0, 1.0);
        h.add(1, +1.0, 1.0);
        return make_node_with_one_hist(std::move(h));
    };

    auto const frontier_arr = std::array{make_parent(), make_parent()};
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

TEST_CASE("HistogramLevelSplitFinder: picks the summed-gain argmax, not per-parent picks",
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

    auto make_a = [] {
        Histogram h{4};
        h.add(0, -2.0, 1.0);
        h.add(1, 0.0, 1.0);
        h.add(2, 0.0, 1.0);
        return make_node_with_one_hist(std::move(h));
    };
    auto make_b = [] {
        Histogram h{4};
        h.add(0, 0.0, 1.0);
        h.add(1, 0.0, 1.0);
        h.add(2, -4.0, 1.0);
        return make_node_with_one_hist(std::move(h));
    };

    auto const frontier_arr  = std::array{make_a(), make_b()};
    FrontierInput const span = frontier_arr;

    TreeConfig cfg{.min_child_hess = 0.0F, .lambda_l2 = 1.0F, .min_data_in_leaf = 0};

    SplitOutput const s = HistogramLevelSplitFinder::find(span, cfg);

    REQUIRE(s.valid);
    CHECK(s.feature_id == feature_id_t{0});
    CHECK(s.bin_id == bin_id_t{1});

    // bin=1: Parent A children (gL=-2, hL=2), (gR=0, hR=1); B children (gL=0, hL=2), (gR=-4, hR=1).
    // 13/3 mathematically; Approx because the level finder's accumulation order differs.
    double const expected_gain =
        (score(-2.0, 2.0, cfg.lambda_l2) + score(0.0, 1.0, cfg.lambda_l2) -
         score(-2.0, 3.0, cfg.lambda_l2)) +
        (score(0.0, 2.0, cfg.lambda_l2) + score(-4.0, 1.0, cfg.lambda_l2) -
         score(-4.0, 3.0, cfg.lambda_l2));
    CHECK(s.gain == Catch::Approx(expected_gain).epsilon(1e-12));
}

TEST_CASE("HistogramLevelSplitFinder: rejects whole split when any parent violates min_child_hess",
          "[split][level][min_child_hess]")
{
    // Parent A and B both have the obvious cut at bin 0, but B's children
    // hess (0.5) is below min_child_hess (0.8). Whole (fid=0, b=0, *) is
    // rejected. n_bins=3 so prefix_size=1: no other cut to fall back to.
    auto make_a = [] {
        Histogram h{3};
        h.add(0, -1.0, 1.0);
        h.add(1, +1.0, 1.0);
        return make_node_with_one_hist(std::move(h));
    };
    auto make_b = [] {
        Histogram h{3};
        h.add(0, -1.0, 0.5);
        h.add(1, +1.0, 0.5);
        return make_node_with_one_hist(std::move(h));
    };

    auto const frontier_arr  = std::array{make_a(), make_b()};
    FrontierInput const span = frontier_arr;

    TreeConfig cfg{
        .min_child_hess = 0.8F, .lambda_l2 = 1.0F, .min_data_in_leaf = 0};

    SplitOutput const s = HistogramLevelSplitFinder::find(span, cfg);

    CHECK_FALSE(s.valid);
    CHECK(s.gain == 0.0);
}

TEST_CASE("HistogramLevelSplitFinder: min_gain_to_split rejects sub-threshold summed gain",
          "[split][level][min_gain_to_split]")
{
    // Two identical parents → summed gain = 2.0. Threshold 3.0 rejects.
    auto make_parent = [] {
        Histogram h{3};
        h.add(0, -1.0, 1.0);
        h.add(1, +1.0, 1.0);
        return make_node_with_one_hist(std::move(h));
    };

    auto const frontier_arr  = std::array{make_parent(), make_parent()};
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
    TreeConfig cfg{.lambda_l2 = 1.0F, .min_data_in_leaf = 0};
    SplitOutput const s = HistogramLevelSplitFinder::find(FrontierInput{}, cfg);

    CHECK_FALSE(s.valid);
    CHECK(s.gain == 0.0);
}

TEST_CASE("HistogramLevelSplitFinder: degenerate prefix_size == 0 returns invalid",
          "[split][level][edge]")
{
    // n_bins=1 → prefix_size()=0; n_bins guard in the per-feature loop
    // makes this a no-op. No best is ever recorded.
    auto make_parent = [] {
        Histogram h{1};
        h.add(0, 1.0, 1.0);
        return make_node_with_one_hist(std::move(h));
    };

    auto const frontier_arr  = std::array{make_parent(), make_parent()};
    FrontierInput const span = frontier_arr;

    TreeConfig cfg{.lambda_l2 = 1.0F, .min_data_in_leaf = 0};
    SplitOutput const s = HistogramLevelSplitFinder::find(span, cfg);

    CHECK_FALSE(s.valid);
    CHECK(s.gain == 0.0);
}
