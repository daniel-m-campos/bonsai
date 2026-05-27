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

template <typename TreeT, size_t N>
float predict_one(TreeT const &tree, std::array<float, N> const &row)
{
    std::array<float, 1> out{};
    tree.predict(features_view{row.data(), 1, N}, floats_out{out});
    return out[0];
}

DenseTree single_leaf(float value)
{
    DenseTree::Nodes nodes;
    nodes.emplace_back(DenseTree::leaf(value));
    return DenseTree{std::move(nodes), DenseTree::Params{.depth = 0, .n_leaves = 1}};
}

// One internal node at index 0 splitting on `fid` at `default_threshold`,
// with two leaves at indices 1 (left = default_left_leaf) and 2 (right =
// default_right_leaf).
DenseTree one_split(feature_id_t fid, bool default_left)
{
    DenseTree::Nodes nodes;
    nodes.emplace_back(DenseTree::internal(fid, default_threshold, node_id_t{1},
                                           node_id_t{2}, default_left));
    nodes.emplace_back(DenseTree::leaf(default_left_leaf));
    nodes.emplace_back(DenseTree::leaf(default_right_leaf));
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
    nodes.emplace_back(DenseTree::internal(fid_0, root_threshold, left_internal_node,
                                           right_internal_node, false));
    nodes.emplace_back(DenseTree::internal(fid_1, left_subtree_threshold, leaf_ll_idx,
                                           leaf_lr_idx, false));
    nodes.emplace_back(DenseTree::internal(fid_1, right_subtree_threshold, leaf_rl_idx,
                                           leaf_rr_idx, false));
    nodes.emplace_back(DenseTree::leaf(leaf_ll_value));
    nodes.emplace_back(DenseTree::leaf(leaf_lr_value));
    nodes.emplace_back(DenseTree::leaf(leaf_rl_value));
    nodes.emplace_back(DenseTree::leaf(leaf_rr_value));
    return DenseTree{std::move(nodes),
                     DenseTree::Params{.depth    = params_multi_depth,
                                       .n_leaves = params_multi_leaves}};
}

} // namespace

TEST_CASE("DenseTree: predict returns the leaf value for a single-leaf tree",
          "[dense_tree][predict]")
{
    auto                 tree = single_leaf(single_leaf_value);
    std::array<float, 1> row{single_leaf_input};
    CHECK(predict_one(tree, row) == single_leaf_value);
}

TEST_CASE("DenseTree: predict routes by less-than-or-equal threshold",
          "[dense_tree][predict]")
{
    auto tree = one_split(fid_0, /*default_left=*/false);

    std::array<float, 1> below{below_threshold};
    std::array<float, 1> above{above_threshold};

    CHECK(predict_one(tree, below) == default_left_leaf);
    CHECK(predict_one(tree, above) == default_right_leaf);
}

TEST_CASE("DenseTree: predict routes value equal to threshold to the left",
          "[dense_tree][predict][edge]")
{
    // Comparison is `v <= threshold`, matching the binner's right-inclusive
    // bin partition (v ∈ (cuts[b-1], cuts[b]]), so v == threshold goes left.
    auto tree = one_split(fid_0, /*default_left=*/false);

    std::array<float, 1> at{at_threshold};

    CHECK(predict_one(tree, at) == default_left_leaf);
}

TEST_CASE("DenseTree: predict routes NaN left when default_left is true",
          "[dense_tree][predict][nan]")
{
    auto tree = one_split(fid_0, /*default_left=*/true);

    std::array<float, 1> nan_row{f_nan};

    CHECK(predict_one(tree, nan_row) == default_left_leaf);
}

TEST_CASE("DenseTree: predict routes NaN right when default_left is false",
          "[dense_tree][predict][nan]")
{
    auto tree = one_split(fid_0, /*default_left=*/false);

    std::array<float, 1> nan_row{f_nan};

    CHECK(predict_one(tree, nan_row) == default_right_leaf);
}

TEST_CASE("DenseTree: predict walks a multi-level tree to the correct leaf",
          "[dense_tree][predict]")
{
    auto tree = depth2_two_feature_tree();

    std::array<float, 2> ll{below_threshold, f1_below_half};
    std::array<float, 2> lr{below_threshold, f1_one};
    std::array<float, 2> rl{above_threshold, f1_below_two};
    std::array<float, 2> rr{above_threshold, f1_above_two};

    CHECK(predict_one(tree, ll) == leaf_ll_value);
    CHECK(predict_one(tree, lr) == leaf_lr_value);
    CHECK(predict_one(tree, rl) == leaf_rl_value);
    CHECK(predict_one(tree, rr) == leaf_rr_value);
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

    tree.predict(features_view{rows.data(), 4, 2}, out);

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
        batch_below,       // row 0: <=1.0 → -0.5
        batch_above,       // row 1: >1.0  → +0.5
        f_nan,             // row 2: NaN, default_left → -0.5
        default_threshold, // row 3: ==1.0 → -0.5 (v <= threshold goes left)
    };
    std::array<float, 4> out{};

    tree.predict(features_view{rows.data(), 4, 1}, out);

    CHECK(out[0] == default_left_leaf);
    CHECK(out[1] == default_right_leaf);
    CHECK(out[2] == default_left_leaf);
    CHECK(out[3] == default_left_leaf);
}
