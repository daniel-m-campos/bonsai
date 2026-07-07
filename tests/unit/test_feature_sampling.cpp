#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <set>
#include <utility>
#include <vector>

#include "bonsai/config/tree_config.hpp"
#include "bonsai/detail/column_batch.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/tree.hpp"
#include "bonsai/types.hpp"
#include "test_grower_helpers.hpp"

using namespace bonsai;       // NOLINT
using namespace bonsai::test; // NOLINT

namespace
{

// 4 identical informative features; any single one separates the data.
detail::ColumnBatch four_identical_features()
{
    std::vector<float> const col{0.0F, 0.1F, 0.9F, 1.0F};
    return detail::ColumnBatch{
        .features      = {col, col, col, col},
        .labels        = std::vector<float>(4, 0.0F),
        .weights       = {},
        .feature_names = {"a", "b", "c", "d"},
    };
}

std::set<feature_id_t> split_features_of(DenseTree const &tree)
{
    std::set<feature_id_t> used;
    for (auto const &n : tree.nodes())
    {
        if (!DenseTree::is_leaf(n))
        {
            used.insert(n.feature_id);
        }
    }
    return used;
}

} // namespace

TEST_CASE("feature_fraction=1 uses the full feature set (baseline unchanged)",
          "[grower][colsample]")
{
    auto              in = separable_4row();
    TreeConfig        cfg{.min_child_hess   = 0.0F,
                          .lambda_l2        = 1.0F,
                          .feature_fraction = 1.0F,
                          .max_depth        = 1,
                          .min_data_in_leaf = 0};
    DepthwiseGrower<> grower{cfg};
    auto [tree, values, tree_lids] =
        grower.grow(in.built.ds, in.grad, in.hess, in.rows);
    CHECK(tree.params().n_leaves == 2);
}

TEST_CASE("feature_fraction=0.25 restricts each tree to one feature",
          "[grower][colsample]")
{
    auto               built = build(four_identical_features());
    std::vector<float> grad{-1.0F, -1.0F, +1.0F, +1.0F};
    std::vector<float> hess(4, 1.0F);
    auto               rows = iota_rows(4);

    TreeConfig        cfg{.min_child_hess   = 0.0F,
                          .lambda_l2        = 1.0F,
                          .feature_fraction = 0.25F,
                          .max_depth        = 3,
                          .min_data_in_leaf = 0};
    DepthwiseGrower<> grower{cfg};

    // Across several trees, every split must use exactly the one feature the
    // per-tree draw selected, and different trees should not all agree
    // (the rng advances between grow calls).
    std::set<feature_id_t> seen_across_trees;
    for (int t = 0; t < 8; ++t)
    {
        auto [tree, values, tree_lids] = grower.grow(built.ds, grad, hess, rows);
        auto const used                = split_features_of(tree);
        CHECK(used.size() == 1); // one selected feature per tree
        seen_across_trees.insert(used.begin(), used.end());
    }
    CHECK(seen_across_trees.size() > 1); // draws vary across trees
}

TEST_CASE("feature_fraction leaf values still use full node totals",
          "[grower][colsample]")
{
    // With one of four features selected, leaf values must come from the
    // node's grad/hess sums (identical for any selected feature), not zeros.
    auto               built = build(four_identical_features());
    std::vector<float> grad{-1.0F, -1.0F, +1.0F, +1.0F};
    std::vector<float> hess(4, 1.0F);
    auto               rows = iota_rows(4);

    TreeConfig        cfg{.min_child_hess   = 0.0F,
                          .lambda_l2        = 1.0F,
                          .feature_fraction = 0.25F,
                          .max_depth        = 1,
                          .min_data_in_leaf = 0};
    DepthwiseGrower<> grower{cfg};
    auto [tree, values, tree_lids] = grower.grow(built.ds, grad, hess, rows);

    REQUIRE(tree.params().n_leaves == 2);
    // Same expected values as the unsampled separable smoke test.
    float const left  = predict_one(tree, std::vector<float>{0.0F, 0.0F, 0.0F, 0.0F});
    float const right = predict_one(tree, std::vector<float>{1.0F, 1.0F, 1.0F, 1.0F});
    CHECK(left == Catch::Approx(2.0F / 3.0F).epsilon(1e-5));
    CHECK(right == Catch::Approx(-2.0F / 3.0F).epsilon(1e-5));
}

TEST_CASE("feature_fraction draws are deterministic per seed",
          "[grower][colsample][determinism]")
{
    auto               built = build(four_identical_features());
    std::vector<float> grad{-1.0F, -1.0F, +1.0F, +1.0F};
    std::vector<float> hess(4, 1.0F);
    auto               rows = iota_rows(4);

    TreeConfig cfg{.min_child_hess   = 0.0F,
                   .lambda_l2        = 1.0F,
                   .feature_fraction = 0.5F,
                   .max_depth        = 2,
                   .min_data_in_leaf = 0,
                   .feature_seed     = 7};

    auto grow_sequence = [&]
    {
        DepthwiseGrower<>                   g{cfg};
        std::vector<std::set<feature_id_t>> seq;
        for (int t = 0; t < 5; ++t)
        {
            auto [tree, values, tree_lids] = g.grow(built.ds, grad, hess, rows);
            seq.push_back(split_features_of(tree));
        }
        return seq;
    };
    CHECK(grow_sequence() == grow_sequence());
}
