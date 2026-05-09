#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <utility>

#include "bonsai/tree.hpp"
#include "bonsai/types.hpp"
#include "test_tree_constants.hpp"

using namespace bonsai;       // NOLINT
using namespace bonsai::test; // NOLINT

namespace
{

DenseTree single_leaf(float value)
{
    DenseTree::DenseTree::Nodes nodes;
    nodes.emplace_back(DenseTree::DenseTree::LeafNode{value});
    return DenseTree{std::move(nodes), DenseTree::Params{.depth = 0, .n_leaves = 1}};
}

// One internal node at index 0 splitting on `fid` at `default_threshold`,
// with two leaves at indices 1 (left = default_left_leaf) and 2 (right =
// default_right_leaf).
DenseTree one_split(feature_id_t fid, bool default_left)
{
    DenseTree::DenseTree::Nodes nodes;
    nodes.emplace_back(DenseTree::DenseTree::InternalNode{
        .feature_id   = fid,
        .threshold    = default_threshold,
        .left         = node_id_t{1},
        .right        = node_id_t{2},
        .default_left = default_left,
    });
    nodes.emplace_back(DenseTree::DenseTree::LeafNode{default_left_leaf});
    nodes.emplace_back(DenseTree::DenseTree::LeafNode{default_right_leaf});
    return DenseTree{std::move(nodes), DenseTree::Params{.depth = 1, .n_leaves = 2}};
}

// Depth-2 tree on 2 features:
//
//                 [f0 < 1.0]
//                 /         \
//          [f1 < 0.5]   [f1 < 2.0]
//          /      \      /      \
//       L=-1    L=-0.5  L=+0.5  L=+1
DenseTree depth2_two_feature_tree()
{
    DenseTree::Nodes nodes;
    nodes.emplace_back(DenseTree::InternalNode{.feature_id   = fid_0,
                                               .threshold    = root_threshold,
                                               .left         = left_internal_node,
                                               .right        = right_internal_node,
                                               .default_left = false});
    nodes.emplace_back(DenseTree::InternalNode{.feature_id   = fid_1,
                                               .threshold    = left_subtree_threshold,
                                               .left         = leaf_ll_idx,
                                               .right        = leaf_lr_idx,
                                               .default_left = false});
    nodes.emplace_back(DenseTree::InternalNode{.feature_id   = fid_1,
                                               .threshold    = right_subtree_threshold,
                                               .left         = leaf_rl_idx,
                                               .right        = leaf_rr_idx,
                                               .default_left = false});
    nodes.emplace_back(DenseTree::LeafNode{leaf_ll_value});
    nodes.emplace_back(DenseTree::LeafNode{leaf_lr_value});
    nodes.emplace_back(DenseTree::LeafNode{leaf_rl_value});
    nodes.emplace_back(DenseTree::LeafNode{leaf_rr_value});
    return DenseTree{std::move(nodes),
                     DenseTree::Params{.depth    = params_multi_depth,
                                       .n_leaves = params_multi_leaves}};
}

} // namespace

TEST_CASE("DenseTree: predict returns the leaf value for a single-leaf tree",
          "[dense_tree][predict]")
{
    auto tree = single_leaf(single_leaf_value);
    std::array<float, 1> row{single_leaf_input};
    CHECK(tree.predict(row) == single_leaf_value);
}

TEST_CASE("DenseTree: predict routes by strict less-than threshold",
          "[dense_tree][predict]")
{
    auto tree = one_split(fid_0, /*default_left=*/false);

    std::array<float, 1> below{below_threshold};
    std::array<float, 1> above{above_threshold};

    CHECK(tree.predict(below) == default_left_leaf);
    CHECK(tree.predict(above) == default_right_leaf);
}

TEST_CASE("DenseTree: predict routes value equal to threshold to the right",
          "[dense_tree][predict][edge]")
{
    // Comparison is `v < threshold`, so v == threshold goes right.
    auto tree = one_split(fid_0, /*default_left=*/false);

    std::array<float, 1> at{at_threshold};

    CHECK(tree.predict(at) == default_right_leaf);
}

TEST_CASE("DenseTree: predict routes NaN left when default_left is true",
          "[dense_tree][predict][nan]")
{
    auto tree = one_split(fid_0, /*default_left=*/true);

    std::array<float, 1> nan_row{f_nan};

    CHECK(tree.predict(nan_row) == default_left_leaf);
}

TEST_CASE("DenseTree: predict routes NaN right when default_left is false",
          "[dense_tree][predict][nan]")
{
    auto tree = one_split(fid_0, /*default_left=*/false);

    std::array<float, 1> nan_row{f_nan};

    CHECK(tree.predict(nan_row) == default_right_leaf);
}

TEST_CASE("DenseTree: predict walks a multi-level tree to the correct leaf",
          "[dense_tree][predict]")
{
    auto tree = depth2_two_feature_tree();

    std::array<float, 2> ll{below_threshold, f1_below_half};
    std::array<float, 2> lr{below_threshold, f1_one};
    std::array<float, 2> rl{above_threshold, f1_below_two};
    std::array<float, 2> rr{above_threshold, f1_above_two};

    CHECK(tree.predict(ll) == leaf_ll_value);
    CHECK(tree.predict(lr) == leaf_lr_value);
    CHECK(tree.predict(rl) == leaf_rl_value);
    CHECK(tree.predict(rr) == leaf_rr_value);
}

TEST_CASE("DenseTree: params() returns construction parameters", "[dense_tree][params]")
{
    auto single = single_leaf(single_leaf_value);
    CHECK(single.params().depth == params_single_depth);
    CHECK(single.params().n_leaves == params_single_leaves);

    auto multi = depth2_two_feature_tree();
    CHECK(multi.params().depth == params_multi_depth);
    CHECK(multi.params().n_leaves == params_multi_leaves);
}

TEST_CASE("DenseTree: batch predict strides correctly across multiple features",
          "[dense_tree][predict]")
{
    auto tree = depth2_two_feature_tree();

    std::array<float, 8> rows{
        below_threshold, f1_below_half, below_threshold, f1_one,
        above_threshold, f1_below_two,  above_threshold, f1_above_two,
    };
    std::array<float, 4> out{};

    tree.predict(rows, /*n_features=*/2, out);

    CHECK(out[0] == leaf_ll_value);
    CHECK(out[1] == leaf_lr_value);
    CHECK(out[2] == leaf_rl_value);
    CHECK(out[3] == leaf_rr_value);
}

TEST_CASE("DenseTree: batch predict produces same results as single-row predict",
          "[dense_tree][predict]")
{
    auto tree = one_split(fid_0, /*default_left=*/true);

    // 4 rows × 1 feature, row-major. Includes a NaN to exercise default routing.
    std::array<float, 4> rows{
        batch_below,       // row 0: <1.0  → -0.5
        batch_above,       // row 1: >=1.0 → +0.5
        f_nan,             // row 2: NaN, default_left → -0.5
        default_threshold, // row 3: ==1.0 → +0.5 (strict <)
    };
    std::array<float, 4> out{};

    tree.predict(rows, /*n_features=*/1, out);

    CHECK(out[0] == default_left_leaf);
    CHECK(out[1] == default_right_leaf);
    CHECK(out[2] == default_left_leaf);
    CHECK(out[3] == default_right_leaf);
}
