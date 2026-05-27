#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "bonsai/config/tree_config.hpp"
#include "bonsai/detail/column_batch.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/types.hpp"
#include "test_grower_helpers.hpp"

using namespace bonsai;       // NOLINT
using namespace bonsai::test; // NOLINT

TEST_CASE("DepthwiseGrower: depth=1 separable yields one split, two leaves",
          "[grower][smoke]")
{
    auto              in = separable_4row();
    TreeConfig        cfg{.min_child_hess   = 0.0F,
                          .lambda_l2        = 1.0F,
                          .max_depth        = 1,
                          .min_data_in_leaf = 0};
    DepthwiseGrower<> grower{cfg};
    auto [tree, train_leaf_values] =
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

TEST_CASE("DepthwiseGrower: depth=2 separable yields four leaves with correct routing",
          "[grower][depth]")
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
    auto built = build(std::move(batch));
    // Row layout (f0,f1 quadrant):
    //   rows 0,1: (lo,lo); rows 2,3: (lo,hi); rows 4,5: (hi,lo); rows 6,7: (hi,hi).
    // Grads sized so f1 wins level 1 (sums ±11 with f0 contributions
    // cancelling), and within each f1 child, f0 splits via magnitude
    // asymmetry between the lo and hi halves.
    std::vector<float> grad{-0.5F, -0.5F, +0.5F, +0.5F, -5.0F, -5.0F, +5.0F, +5.0F};
    std::vector<float> hess(8, 1.0F);
    auto               rows = iota_rows(8);

    TreeConfig        cfg{.min_child_hess   = 0.0F,
                          .lambda_l2        = 1.0F,
                          .max_depth        = 2,
                          .min_data_in_leaf = 0};
    DepthwiseGrower<> grower{cfg};
    auto [tree, train_leaf_values] = grower.grow(built.ds, grad, hess, rows);

    CHECK(tree.params().n_leaves == 4);
    CHECK(tree.params().depth == 2);

    // One representative point per quadrant; sign should match negation of
    // the quadrant's mean grad.
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

TEST_CASE("DepthwiseGrower: max_depth=0 returns single-leaf tree", "[grower][edge]")
{
    auto              in = two_value_pair();
    TreeConfig        cfg{.lambda_l2 = 1.0F, .max_depth = 0, .min_data_in_leaf = 0};
    DepthwiseGrower<> grower{cfg};
    auto [tree, train_leaf_values] =
        grower.grow(in.built.ds, in.grad, in.hess, in.rows);

    CHECK(tree.params().n_leaves == 1);
    CHECK(tree.params().depth == 0);
    // Sum grad = 0, sum hess = 2 → leaf value = -0/(2+1) = 0.
    CHECK(predict_one(tree, std::vector<float>{0.5F}) == 0.0F);
}

TEST_CASE("DepthwiseGrower: no positive-gain split yields single leaf",
          "[grower][no_split]")
{
    auto              in = uniform_3row();
    TreeConfig        cfg{.min_child_hess   = 0.0F,
                          .lambda_l2        = 1.0F,
                          .max_depth        = 3,
                          .min_data_in_leaf = 0};
    DepthwiseGrower<> grower{cfg};
    auto [tree, train_leaf_values] =
        grower.grow(in.built.ds, in.grad, in.hess, in.rows);

    CHECK(tree.params().n_leaves == 1);
    // Root never split: depth should be 0, not the loop's iteration count.
    CHECK(tree.params().depth == 0);
    // Sum grad = 3, sum hess = 3 → leaf value = -3/(3+1) = -0.75.
    CHECK(predict_one(tree, std::vector<float>{0.5F}) == -0.75F);
}

TEST_CASE("DepthwiseGrower: NaN predict routes via default_left", "[grower][missing]")
{
    auto              in = separable_4row();
    TreeConfig        cfg{.min_child_hess   = 0.0F,
                          .lambda_l2        = 1.0F,
                          .max_depth        = 1,
                          .min_data_in_leaf = 0};
    DepthwiseGrower<> grower{cfg};
    auto [tree, train_leaf_values] =
        grower.grow(in.built.ds, in.grad, in.hess, in.rows);

    // With no missing values at fit time, the splitter's default_left tie-break
    // favors `true`, so NaN inputs at predict time must route to the left leaf
    // (grad sum -2 → leaf value +2/3).
    float const nan_pred =
        predict_one(tree, std::vector<float>{std::numeric_limits<float>::quiet_NaN()});
    CHECK(std::isfinite(nan_pred));
    CHECK(nan_pred == Catch::Approx(2.0F / 3.0F).epsilon(1e-5));
}

TEST_CASE("DepthwiseGrower: min_child_hess starves all splits → single leaf",
          "[grower][min_child_hess]")
{
    // Same separable layout as the smoke test, but min_child_hess > max
    // achievable hess on either side (2.0). Every candidate is rejected;
    // root falls through to a leaf.
    auto              in = separable_4row();
    TreeConfig        cfg{.min_child_hess   = 3.0F,
                          .lambda_l2        = 1.0F,
                          .max_depth        = 2,
                          .min_data_in_leaf = 0};
    DepthwiseGrower<> grower{cfg};
    auto [tree, train_leaf_values] =
        grower.grow(in.built.ds, in.grad, in.hess, in.rows);

    CHECK(tree.params().n_leaves == 1);
    CHECK(tree.params().depth == 0);
    // Sum grad = 0, sum hess = 4 → leaf value = -0/(4+1) = 0.
    CHECK(predict_one(tree, std::vector<float>{0.5F}) == 0.0F);
}

TEST_CASE("DepthwiseGrower: asymmetric tree — one child splits, other stays a leaf",
          "[grower][asymmetric]")
{
    // Single feature, 8 rows. Root splits between rows 3 and 4.
    //   Left subtree (rows 0-3): grad uniform = -5 → every sub-split has
    //     non-positive gain → becomes a leaf.
    //   Right subtree (rows 4-7): grads -1,-1,+1,+1 → splits between rows 5
    //     and 6 → two leaves.
    // Total: 3 leaves, depth 2.
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

    TreeConfig        cfg{.min_child_hess   = 0.0F,
                          .lambda_l2        = 1.0F,
                          .max_depth        = 2,
                          .min_data_in_leaf = 0};
    DepthwiseGrower<> grower{cfg};
    auto [tree, train_leaf_values] = grower.grow(built.ds, grad, hess, rows);

    CHECK(tree.params().n_leaves == 3);
    CHECK(tree.params().depth == 2);

    // Left leaf: grad sum -20, hess 4, lambda 1 → -(-20)/(4+1) = +4.
    float const left_leaf = predict_one(tree, std::vector<float>{0.0F});
    CHECK(left_leaf == Catch::Approx(4.0F).epsilon(1e-5));
    // Right subtree leaves: rows {4,5} grad -2, hess 2 → +2/3;
    //                       rows {6,7} grad +2, hess 2 → -2/3.
    float const right_lo = predict_one(tree, std::vector<float>{1.0F});
    float const right_hi = predict_one(tree, std::vector<float>{1.3F});
    CHECK(right_lo == Catch::Approx(2.0F / 3.0F).epsilon(1e-5));
    CHECK(right_hi == Catch::Approx(-2.0F / 3.0F).epsilon(1e-5));
}

TEST_CASE("DepthwiseGrower: empty row_indices yields zero-valued single leaf",
          "[grower][edge]")
{
    auto in = uniform_3row();
    in.rows = {}; // empty
    TreeConfig        cfg{.min_child_hess   = 0.0F,
                          .lambda_l2        = 1.0F,
                          .max_depth        = 3,
                          .min_data_in_leaf = 0};
    DepthwiseGrower<> grower{cfg};
    auto [tree, train_leaf_values] =
        grower.grow(in.built.ds, in.grad, in.hess, in.rows);

    CHECK(tree.params().n_leaves == 1);
    CHECK(tree.params().depth == 0);
    // Sum grad = 0, sum hess = 0 → leaf value = -0/(0+1) = 0.
    CHECK(predict_one(tree, std::vector<float>{0.5F}) == 0.0F);
}
