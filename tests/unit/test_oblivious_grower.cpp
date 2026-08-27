#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "bonsai/config/errors.hpp"
#include "bonsai/config/tree_config.hpp"
#include "bonsai/detail/column_batch.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/types.hpp"
#include "test_grower_helpers.hpp"

using namespace bonsai;       // NOLINT
using namespace bonsai::test; // NOLINT

TEST_CASE("ObliviousGrower: depth=1 separable yields one split, two leaves",
          "[grower][oblivious][smoke]")
{
    auto              in = separable_4row();
    TreeConfig        cfg{.min_child_hess   = 0.0F,
                          .lambda_l2        = 1.0F,
                          .max_depth        = 1,
                          .min_data_in_leaf = 0};
    ObliviousGrower<> grower{cfg};
    auto [tree, train_leaf_values, tree_lids] =
        grower.grow(in.built.ds, in.grad, in.hess, in.rows);

    CHECK(tree.params().n_leaves == 2);
    CHECK(tree.params().depth == 1);

    // Left rows (grad -1) → leaf value -(-2)/(2+1) = +2/3.
    // Right rows (grad +1) → leaf value -(+2)/(2+1) = -2/3.
    float const left_pred  = predict_one(tree, std::vector<float>{0.0F});
    float const right_pred = predict_one(tree, std::vector<float>{1.0F});
    CHECK(left_pred == Catch::Approx(2.0F / 3.0F).epsilon(1e-5));
    CHECK(right_pred == Catch::Approx(-2.0F / 3.0F).epsilon(1e-5));
}

TEST_CASE("ObliviousGrower: depth=2 separable yields four leaves with correct routing",
          "[grower][oblivious][depth]")
{
    // 2 features, 8 rows arranged so each (f0_low/high, f1_low/high) quadrant
    // has 2 rows. Distinct feature values per row so the binner gives each
    // value its own bin (avoids quantile collapse for tiny datasets).
    detail::ColumnBatch batch{
        .features      = {{0.0F, 0.1F, 0.2F, 0.3F, 1.0F, 1.1F, 1.2F, 1.3F},
                          {0.0F, 0.1F, 1.0F, 1.1F, 0.2F, 0.3F, 1.2F, 1.3F}},
        .labels        = std::vector<float>(8, 0.0F),
        .weights       = {},
        .feature_names = {"a", "b"},
    };
    auto               built = build(std::move(batch));
    std::vector<float> grad{-0.5F, -0.5F, +0.5F, +0.5F, -5.0F, -5.0F, +5.0F, +5.0F};
    std::vector<float> hess(8, 1.0F);
    auto               rows = iota_rows(8);

    TreeConfig        cfg{.min_child_hess   = 0.0F,
                          .lambda_l2        = 1.0F,
                          .max_depth        = 2,
                          .min_data_in_leaf = 0};
    ObliviousGrower<> grower{cfg};
    auto [tree, train_leaf_values, tree_lids] = grower.grow(built.ds, grad, hess, rows);

    CHECK(tree.params().n_leaves == 4);
    CHECK(tree.params().depth == 2);

    float const p_lolo =
        predict_one(tree, std::vector<float>{0.0F, 0.0F}); // grad <0 → +
    float const p_lohi =
        predict_one(tree, std::vector<float>{0.0F, 2.0F}); // grad >0 → -
    float const p_hilo =
        predict_one(tree, std::vector<float>{2.0F, 0.0F}); // grad <0 → +
    float const p_hihi =
        predict_one(tree, std::vector<float>{2.0F, 2.0F}); // grad >0 → -
    CHECK(p_lolo > 0.0F);
    CHECK(p_lohi < 0.0F);
    CHECK(p_hilo > 0.0F);
    CHECK(p_hihi < 0.0F);
}

TEST_CASE("ObliviousGrower: max_depth=0 returns single-leaf tree",
          "[grower][oblivious][edge]")
{
    auto              in = two_value_pair();
    TreeConfig        cfg{.lambda_l2 = 1.0F, .max_depth = 0, .min_data_in_leaf = 0};
    ObliviousGrower<> grower{cfg};
    auto [tree, train_leaf_values, tree_lids] =
        grower.grow(in.built.ds, in.grad, in.hess, in.rows);

    CHECK(tree.params().n_leaves == 1);
    CHECK(tree.params().depth == 0);
    // Sum grad = 0, sum hess = 2 → leaf value = -0/(2+1) = 0.
    CHECK(predict_one(tree, std::vector<float>{0.5F}) == 0.0F);
}

TEST_CASE("ObliviousGrower: no positive-gain split yields single leaf",
          "[grower][oblivious][no_split]")
{
    auto              in = uniform_3row();
    TreeConfig        cfg{.min_child_hess   = 0.0F,
                          .lambda_l2        = 1.0F,
                          .max_depth        = 3,
                          .min_data_in_leaf = 0};
    ObliviousGrower<> grower{cfg};
    auto [tree, train_leaf_values, tree_lids] =
        grower.grow(in.built.ds, in.grad, in.hess, in.rows);

    CHECK(tree.params().n_leaves == 1);
    CHECK(tree.params().depth == 0);
    // Sum grad = 3, sum hess = 3 → leaf value = -3/(3+1) = -0.75.
    CHECK(predict_one(tree, std::vector<float>{0.5F}) == -0.75F);
}

TEST_CASE("ObliviousGrower: NaN predict routes via default_left",
          "[grower][oblivious][missing]")
{
    auto              in = separable_4row();
    TreeConfig        cfg{.min_child_hess   = 0.0F,
                          .lambda_l2        = 1.0F,
                          .max_depth        = 1,
                          .min_data_in_leaf = 0};
    ObliviousGrower<> grower{cfg};
    auto [tree, train_leaf_values, tree_lids] =
        grower.grow(in.built.ds, in.grad, in.hess, in.rows);

    float const nan_pred =
        predict_one(tree, std::vector<float>{std::numeric_limits<float>::quiet_NaN()});
    CHECK(std::isfinite(nan_pred));
    CHECK(nan_pred == Catch::Approx(2.0F / 3.0F).epsilon(1e-5));
}

TEST_CASE("ObliviousGrower: min_child_hess starves all splits → single leaf",
          "[grower][oblivious][min_child_hess]")
{
    auto              in = separable_4row();
    TreeConfig        cfg{.min_child_hess   = 3.0F,
                          .lambda_l2        = 1.0F,
                          .max_depth        = 2,
                          .min_data_in_leaf = 0};
    ObliviousGrower<> grower{cfg};
    auto [tree, train_leaf_values, tree_lids] =
        grower.grow(in.built.ds, in.grad, in.hess, in.rows);

    CHECK(tree.params().n_leaves == 1);
    CHECK(tree.params().depth == 0);
    // Sum grad = 0, sum hess = 4 → leaf value = -0/(4+1) = 0.
    CHECK(predict_one(tree, std::vector<float>{0.5F}) == 0.0F);
}

TEST_CASE("ObliviousGrower: empty row_indices yields zero-valued single leaf",
          "[grower][oblivious][edge]")
{
    auto in = uniform_3row();
    in.rows = {}; // empty
    TreeConfig        cfg{.min_child_hess   = 0.0F,
                          .lambda_l2        = 1.0F,
                          .max_depth        = 3,
                          .min_data_in_leaf = 0};
    ObliviousGrower<> grower{cfg};
    auto [tree, train_leaf_values, tree_lids] =
        grower.grow(in.built.ds, in.grad, in.hess, in.rows);

    CHECK(tree.params().n_leaves == 1);
    CHECK(tree.params().depth == 0);
    CHECK(predict_one(tree, std::vector<float>{0.5F}) == 0.0F);
}

TEST_CASE("ObliviousGrower: level with no gain stops growth before max_depth",
          "[grower][oblivious][early_stop]")
{
    // separable_4row: grad {-1,-1,+1,+1}. The level-1 split separates the two
    // halves; within each child, grads are uniform so no further split has
    // positive gain. With max_depth=3, an oblivious grower must stop at
    // depth=1 (all-or-nothing) — n_leaves stays at 2, not 4 or 8.
    auto              in = separable_4row();
    TreeConfig        cfg{.min_child_hess   = 0.0F,
                          .lambda_l2        = 1.0F,
                          .max_depth        = 3,
                          .min_data_in_leaf = 0};
    ObliviousGrower<> grower{cfg};
    auto [tree, train_leaf_values, tree_lids] =
        grower.grow(in.built.ds, in.grad, in.hess, in.rows);

    CHECK(tree.params().depth == 1);
    CHECK(tree.params().n_leaves == 2);
}

// INVARIANT: levelwise-rejects-interaction-constraints
// The levelwise (oblivious) grower rejects interaction constraints at
// construction rather than silently ignoring them, on both the CPU and CUDA
// engines: the throw is in the shared template, so an engine cannot opt out
// of it. Monotone constraints are honoured rather than rejected, by
// project_monotone; interaction constraints have no equivalent projection,
// because they constrain which features may share a path rather than how
// leaf values are ordered.
TEST_CASE("ObliviousGrower: rejects constraints it cannot honour at construction",
          "[grower][oblivious][ctor][invariant]")
{
    SECTION("a monotone direction is accepted")
    {
        TreeConfig cfg{};
        cfg.monotone_constraints = {0, 1, 0};
        CHECK_NOTHROW(ObliviousGrower<>{cfg});
    }
    SECTION("an all-zero monotone list is not a constraint")
    {
        TreeConfig cfg{};
        cfg.monotone_constraints = {0, 0, 0};
        CHECK_NOTHROW(ObliviousGrower<>{cfg});
    }
    SECTION("an interaction group throws")
    {
        TreeConfig cfg{};
        cfg.interaction_constraints = {"0,1"};
        CHECK_THROWS_AS(ObliviousGrower<>{cfg}, ConfigError);
    }
}

// INVARIANT: levelwise-monotone-holds
// A levelwise fit under a monotone constraint produces predictions ordered by
// that feature, on both the CPU and CUDA engines. The mechanism is a
// projection of the finished leaf table onto the monotone cone
// (project_monotone), not a veto during growth, so the tree's structure is
// whatever the unconstrained search would have chosen.
TEST_CASE("ObliviousGrower: monotone +1 forces non-decreasing predictions",
          "[grower][oblivious][monotone][invariant]")
{
    // Same gradients as the depthwise case: group means swing down then up,
    // so the unconstrained tree is genuinely non-monotone in the feature.
    detail::ColumnBatch batch{
        .features      = {{0.0F, 0.1F, 0.2F, 1.0F, 1.1F, 1.2F, 2.0F, 2.1F, 2.2F}},
        .labels        = std::vector<float>(9, 0.0F),
        .weights       = {},
        .feature_names = {"a"},
    };
    auto               built = build(std::move(batch));
    std::vector<float> grad{-1.0F, -1.0F, -1.0F, +2.0F, +2.0F,
                            +2.0F, -2.0F, -2.0F, -2.0F};
    std::vector<float> hess(9, 1.0F);
    auto               rows = iota_rows(9);

    TreeConfig unconstrained{.min_child_hess   = 0.0F,
                             .lambda_l2        = 1.0F,
                             .max_depth        = 3,
                             .min_data_in_leaf = 0};
    TreeConfig constrained           = unconstrained;
    constrained.monotone_constraints = {+1};

    auto curve_of = [&](ObliviousTree const &tree)
    {
        std::vector<float> out;
        for (float x : {0.0F, 1.05F, 2.1F})
        {
            out.push_back(predict_one(tree, std::vector<float>{x}));
        }
        return out;
    };

    ObliviousGrower<> free_grower{unconstrained};
    auto [free_tree, free_values, free_lids] =
        free_grower.grow(built.ds, grad, hess, rows);
    auto const free_curve = curve_of(free_tree);
    CHECK((free_curve[1] < free_curve[0] || free_curve[2] < free_curve[1]));

    ObliviousGrower<> mono_grower{constrained};
    auto [mono_tree, mono_values, mono_lids] =
        mono_grower.grow(built.ds, grad, hess, rows);
    auto const curve = curve_of(mono_tree);
    CHECK(curve[0] <= curve[1]);
    CHECK(curve[1] <= curve[2]);

    SECTION("the structure is the unconstrained one, only the values move")
    {
        REQUIRE(free_tree.splits().size() == mono_tree.splits().size());
        for (size_t lvl = 0; lvl < free_tree.splits().size(); ++lvl)
        {
            CHECK(free_tree.splits()[lvl].feature_id ==
                  mono_tree.splits()[lvl].feature_id);
            CHECK(free_tree.splits()[lvl].threshold ==
                  mono_tree.splits()[lvl].threshold);
        }
    }
}

TEST_CASE("ObliviousGrower: monotone -1 forces non-increasing predictions",
          "[grower][oblivious][monotone]")
{
    detail::ColumnBatch batch{
        .features      = {{0.0F, 0.1F, 0.2F, 1.0F, 1.1F, 1.2F, 2.0F, 2.1F, 2.2F}},
        .labels        = std::vector<float>(9, 0.0F),
        .weights       = {},
        .feature_names = {"a"},
    };
    auto               built = build(std::move(batch));
    std::vector<float> grad{-1.0F, -1.0F, -1.0F, +2.0F, +2.0F,
                            +2.0F, -2.0F, -2.0F, -2.0F};
    std::vector<float> hess(9, 1.0F);
    auto               rows = iota_rows(9);

    TreeConfig cfg{.min_child_hess   = 0.0F,
                   .lambda_l2        = 1.0F,
                   .max_depth        = 3,
                   .min_data_in_leaf = 0};
    cfg.monotone_constraints = {-1};

    ObliviousGrower<> grower{cfg};
    auto [tree, values, lids] = grower.grow(built.ds, grad, hess, rows);

    float previous = predict_one(tree, std::vector<float>{0.0F});
    for (float x : {1.05F, 2.1F})
    {
        float const next = predict_one(tree, std::vector<float>{x});
        CHECK(next <= previous);
        previous = next;
    }
}
