#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <utility>
#include <vector>

#include "bonsai/config/tree_config.hpp"
#include "bonsai/detail/column_batch.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/types.hpp"
#include "test_grower_helpers.hpp"

using namespace bonsai;       // NOLINT
using namespace bonsai::test; // NOLINT

namespace
{

// 2 features, 8 rows in 2x2 quadrants (same layout as the depthwise depth=2
// test). Root splits on f1; both children then have an equal-gain f0 split,
// so a limited leaf budget exercises the tie-break.
detail::ColumnBatch quadrant_batch()
{
    return detail::ColumnBatch{
        .features      = {{0.0F, 0.1F, 0.2F, 0.3F, 1.0F, 1.1F, 1.2F, 1.3F},
                          {0.0F, 0.1F, 1.0F, 1.1F, 0.2F, 0.3F, 1.2F, 1.3F}},
        .labels        = std::vector<float>(8, 0.0F),
        .weights       = {},
        .feature_names = {"a", "b"},
    };
}

std::vector<float> quadrant_grad()
{
    return {-0.5F, -0.5F, +0.5F, +0.5F, -5.0F, -5.0F, +5.0F, +5.0F};
}

} // namespace

TEST_CASE("LeafwiseGrower: separable yields one split, two leaves",
          "[grower][leafwise][smoke]")
{
    auto             in = separable_4row();
    TreeConfig       cfg{.min_child_hess   = 0.0F,
                         .lambda_l2        = 1.0F,
                         .max_depth        = 1,
                         .min_data_in_leaf = 0,
                         .max_leaves       = 31};
    LeafwiseGrower<> grower{cfg};
    auto [tree, train_leaf_values] =
        grower.grow(in.built.ds, in.grad, in.hess, in.rows);

    CHECK(tree.params().n_leaves == 2);
    CHECK(tree.params().depth == 1);

    // Same leaf values as depthwise: -(-2)/(2+1) = +2/3 and -(+2)/(2+1) = -2/3.
    float const left_pred  = predict_one(tree, std::vector<float>{0.0F});
    float const right_pred = predict_one(tree, std::vector<float>{1.0F});
    CHECK(left_pred == Catch::Approx(2.0F / 3.0F).epsilon(1e-5));
    CHECK(right_pred == Catch::Approx(-2.0F / 3.0F).epsilon(1e-5));
}

TEST_CASE("LeafwiseGrower: max_leaves=1 returns single root leaf",
          "[grower][leafwise][edge]")
{
    auto             in = separable_4row();
    TreeConfig       cfg{.min_child_hess   = 0.0F,
                         .lambda_l2        = 1.0F,
                         .max_depth        = 6,
                         .min_data_in_leaf = 0,
                         .max_leaves       = 1};
    LeafwiseGrower<> grower{cfg};
    auto [tree, train_leaf_values] =
        grower.grow(in.built.ds, in.grad, in.hess, in.rows);

    CHECK(tree.params().n_leaves == 1);
    CHECK(tree.params().depth == 0);
    // Sum grad = 0, sum hess = 4 → leaf value = 0.
    CHECK(predict_one(tree, std::vector<float>{0.5F}) == 0.0F);
}

TEST_CASE("LeafwiseGrower: max_depth=0 returns single-leaf tree",
          "[grower][leafwise][edge]")
{
    auto             in = two_value_pair();
    TreeConfig       cfg{.lambda_l2        = 1.0F,
                         .max_depth        = 0,
                         .min_data_in_leaf = 0,
                         .max_leaves       = 31};
    LeafwiseGrower<> grower{cfg};
    auto [tree, train_leaf_values] =
        grower.grow(in.built.ds, in.grad, in.hess, in.rows);

    CHECK(tree.params().n_leaves == 1);
    CHECK(tree.params().depth == 0);
    CHECK(predict_one(tree, std::vector<float>{0.5F}) == 0.0F);
}

TEST_CASE("LeafwiseGrower: no positive-gain split yields single leaf",
          "[grower][leafwise][no_split]")
{
    auto             in = uniform_3row();
    TreeConfig       cfg{.min_child_hess   = 0.0F,
                         .lambda_l2        = 1.0F,
                         .max_depth        = 3,
                         .min_data_in_leaf = 0,
                         .max_leaves       = 31};
    LeafwiseGrower<> grower{cfg};
    auto [tree, train_leaf_values] =
        grower.grow(in.built.ds, in.grad, in.hess, in.rows);

    CHECK(tree.params().n_leaves == 1);
    CHECK(tree.params().depth == 0);
    // Sum grad = 3, sum hess = 3 → leaf value = -3/(3+1) = -0.75.
    CHECK(predict_one(tree, std::vector<float>{0.5F}) == -0.75F);
}

TEST_CASE("LeafwiseGrower: leaf budget stops growth at exactly max_leaves",
          "[grower][leafwise][budget]")
{
    auto               built = build(quadrant_batch());
    auto               grad  = quadrant_grad();
    std::vector<float> hess(8, 1.0F);
    auto               rows = iota_rows(8);

    TreeConfig       cfg{.min_child_hess   = 0.0F,
                         .lambda_l2        = 1.0F,
                         .max_depth        = 6,
                         .min_data_in_leaf = 0,
                         .max_leaves       = 3};
    LeafwiseGrower<> grower{cfg};
    auto [tree, train_leaf_values] = grower.grow(built.ds, grad, hess, rows);

    CHECK(tree.params().n_leaves == 3);
    CHECK(tree.params().depth == 2);

    // Root splits on f1; both children have equal-gain f0 splits, so the
    // tie-break (lower node id) expands the left (f1-lo) child. Its leaves:
    //   rows {0,1}: grad -1, hess 2 → +1/3;  rows {4,5}: grad -10 → +10/3.
    // The f1-hi child stays a leaf: grad +11, hess 4 → -11/5.
    float const p_lolo = predict_one(tree, std::vector<float>{0.0F, 0.0F});
    float const p_hilo = predict_one(tree, std::vector<float>{2.0F, 0.0F});
    float const p_lohi = predict_one(tree, std::vector<float>{0.0F, 2.0F});
    float const p_hihi = predict_one(tree, std::vector<float>{2.0F, 2.0F});
    CHECK(p_lolo == Catch::Approx(1.0F / 3.0F).epsilon(1e-5));
    CHECK(p_hilo == Catch::Approx(10.0F / 3.0F).epsilon(1e-5));
    CHECK(p_lohi == Catch::Approx(-11.0F / 5.0F).epsilon(1e-5));
    CHECK(p_hihi == Catch::Approx(-11.0F / 5.0F).epsilon(1e-5));
}

TEST_CASE("LeafwiseGrower: max_leaves=0 is unbounded (depth-capped only)",
          "[grower][leafwise][budget]")
{
    auto               built = build(quadrant_batch());
    auto               grad  = quadrant_grad();
    std::vector<float> hess(8, 1.0F);
    auto               rows = iota_rows(8);

    TreeConfig       cfg{.min_child_hess   = 0.0F,
                         .lambda_l2        = 1.0F,
                         .max_depth        = 2,
                         .min_data_in_leaf = 0,
                         .max_leaves       = 0};
    LeafwiseGrower<> grower{cfg};
    auto [tree, train_leaf_values] = grower.grow(built.ds, grad, hess, rows);

    // Fully grown to the depth cap: same 4 leaves as depthwise at depth 2.
    CHECK(tree.params().n_leaves == 4);
    CHECK(tree.params().depth == 2);

    DepthwiseGrower<> dw{cfg};
    auto [dw_tree, dw_values] = dw.grow(built.ds, grad, hess, rows);
    for (auto const &pt : {std::vector<float>{0.0F, 0.0F}, {0.0F, 2.0F},
                           {2.0F, 0.0F}, {2.0F, 2.0F}})
    {
        CHECK(predict_one(tree, pt) == predict_one(dw_tree, pt));
    }
}

TEST_CASE("LeafwiseGrower: max_depth caps growth below the leaf budget",
          "[grower][leafwise][depth]")
{
    auto               built = build(quadrant_batch());
    auto               grad  = quadrant_grad();
    std::vector<float> hess(8, 1.0F);
    auto               rows = iota_rows(8);

    TreeConfig       cfg{.min_child_hess   = 0.0F,
                         .lambda_l2        = 1.0F,
                         .max_depth        = 1,
                         .min_data_in_leaf = 0,
                         .max_leaves       = 100};
    LeafwiseGrower<> grower{cfg};
    auto [tree, train_leaf_values] = grower.grow(built.ds, grad, hess, rows);

    CHECK(tree.params().n_leaves == 2);
    CHECK(tree.params().depth == 1);
}

TEST_CASE("LeafwiseGrower: grows unbalanced tree when one side has no gain",
          "[grower][leafwise][asymmetric]")
{
    // Same layout as the depthwise asymmetric test: left half uniform grad
    // (leaf), right half separable (two leaves at depth 2).
    detail::ColumnBatch batch{
        .features      = {{0.0F, 0.1F, 0.2F, 0.3F, 1.0F, 1.1F, 1.2F, 1.3F}},
        .labels        = std::vector<float>(8, 0.0F),
        .weights       = {},
        .feature_names = {"a"},
    };
    auto               built = build(std::move(batch));
    std::vector<float> grad{-5.0F, -5.0F, -5.0F, -5.0F, -1.0F, -1.0F, +1.0F, +1.0F};
    std::vector<float> hess(8, 1.0F);
    auto               rows = iota_rows(8);

    TreeConfig       cfg{.min_child_hess   = 0.0F,
                         .lambda_l2        = 1.0F,
                         .max_depth        = 3,
                         .min_data_in_leaf = 0,
                         .max_leaves       = 0};
    LeafwiseGrower<> grower{cfg};
    auto [tree, train_leaf_values] = grower.grow(built.ds, grad, hess, rows);

    CHECK(tree.params().n_leaves == 3);
    CHECK(tree.params().depth == 2);

    float const left_leaf = predict_one(tree, std::vector<float>{0.0F});
    float const right_lo  = predict_one(tree, std::vector<float>{1.0F});
    float const right_hi  = predict_one(tree, std::vector<float>{1.3F});
    CHECK(left_leaf == Catch::Approx(4.0F).epsilon(1e-5));
    CHECK(right_lo == Catch::Approx(2.0F / 3.0F).epsilon(1e-5));
    CHECK(right_hi == Catch::Approx(-2.0F / 3.0F).epsilon(1e-5));
}

TEST_CASE("LeafwiseGrower: identical inputs grow identical trees",
          "[grower][leafwise][determinism]")
{
    auto               built = build(quadrant_batch());
    auto               grad  = quadrant_grad();
    std::vector<float> hess(8, 1.0F);
    auto               rows = iota_rows(8);

    TreeConfig       cfg{.min_child_hess   = 0.0F,
                         .lambda_l2        = 1.0F,
                         .max_depth        = 4,
                         .min_data_in_leaf = 0,
                         .max_leaves       = 3};
    LeafwiseGrower<> grower{cfg};
    auto [tree_a, values_a] = grower.grow(built.ds, grad, hess, rows);
    auto [tree_b, values_b] = grower.grow(built.ds, grad, hess, rows);

    REQUIRE(tree_a.nodes().size() == tree_b.nodes().size());
    for (size_t i = 0; i < tree_a.nodes().size(); ++i)
    {
        auto const &a = tree_a.nodes()[i];
        auto const &b = tree_b.nodes()[i];
        CHECK(a.feature_id == b.feature_id);
        CHECK(a.threshold_or_value == b.threshold_or_value);
        CHECK(a.left == b.left);
        CHECK(a.right == b.right);
        CHECK(a.default_left == b.default_left);
    }
    CHECK(values_a == values_b);
}

TEST_CASE("LeafwiseGrower: empty row_indices yields zero-valued single leaf",
          "[grower][leafwise][edge]")
{
    auto in  = uniform_3row();
    in.rows  = {};
    TreeConfig       cfg{.min_child_hess   = 0.0F,
                         .lambda_l2        = 1.0F,
                         .max_depth        = 3,
                         .min_data_in_leaf = 0,
                         .max_leaves       = 31};
    LeafwiseGrower<> grower{cfg};
    auto [tree, train_leaf_values] =
        grower.grow(in.built.ds, in.grad, in.hess, in.rows);

    CHECK(tree.params().n_leaves == 1);
    CHECK(tree.params().depth == 0);
    CHECK(predict_one(tree, std::vector<float>{0.5F}) == 0.0F);
}
