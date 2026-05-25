#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>

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

// Depth-0 tree with a single leaf. No splits.
ObliviousTree single_leaf(float value)
{
    return ObliviousTree{ObliviousTree::LevelSplits{}, ObliviousTree::LeafTable{value}};
}

// Depth-1 tree splitting on `fid` at `default_threshold`.
// leaf_values_[0] = default_left_leaf, leaf_values_[1] = default_right_leaf.
ObliviousTree one_split(feature_id_t fid, bool default_left)
{
    return ObliviousTree{
        ObliviousTree::LevelSplits{
            {.feature_id   = fid,
             .threshold    = default_threshold,
             .default_left = default_left},
        },
        ObliviousTree::LeafTable{default_left_leaf, default_right_leaf},
    };
}

// Depth-2 tree on 2 features. Identical *semantics* to the DenseTree
// multi-level test — but oblivious shares splits across all nodes at a
// level, so we can't replicate the DenseTree's per-node thresholds. Use
// the same root + a single second-level threshold for both subtrees.
//
//                 [f0 < root_threshold]
//                 /                    \
//          [f1 < right_subtree_threshold (shared)]
//          /      \                    /      \
//      leaf_0   leaf_1              leaf_2   leaf_3
//
// path_bits encode (f0_dir, f1_dir): 00=LL, 01=LR, 10=RL, 11=RR.
ObliviousTree depth2_two_feature_tree()
{
    return ObliviousTree{
        ObliviousTree::LevelSplits{
            {.feature_id = fid_0, .threshold = root_threshold, .default_left = false},
            {.feature_id   = fid_1,
             .threshold    = right_subtree_threshold,
             .default_left = false},
        },
        ObliviousTree::LeafTable{leaf_ll_value, leaf_lr_value, leaf_rl_value,
                                 leaf_rr_value},
    };
}

} // namespace

TEST_CASE("ObliviousTree: predict returns the leaf value for a depth-0 tree",
          "[oblivious_tree][predict]")
{
    auto tree = single_leaf(single_leaf_value);
    std::array<float, 1> row{single_leaf_input};
    CHECK(predict_one(tree, row) == single_leaf_value);
}

TEST_CASE("ObliviousTree: predict routes by less-than-or-equal threshold",
          "[oblivious_tree][predict]")
{
    auto tree = one_split(fid_0, /*default_left=*/false);

    std::array<float, 1> below{below_threshold};
    std::array<float, 1> above{above_threshold};

    CHECK(predict_one(tree, below) == default_left_leaf);
    CHECK(predict_one(tree, above) == default_right_leaf);
}

TEST_CASE("ObliviousTree: predict routes value equal to threshold to the left",
          "[oblivious_tree][predict][edge]")
{
    // Comparison is `v <= threshold`, matching the binner's right-inclusive
    // bin partition, so v == threshold goes left.
    auto tree = one_split(fid_0, /*default_left=*/false);

    std::array<float, 1> at{at_threshold};

    CHECK(predict_one(tree, at) == default_left_leaf);
}

TEST_CASE("ObliviousTree: predict routes NaN left when default_left is true",
          "[oblivious_tree][predict][nan]")
{
    auto tree = one_split(fid_0, /*default_left=*/true);

    std::array<float, 1> nan_row{f_nan};

    CHECK(predict_one(tree, nan_row) == default_left_leaf);
}

TEST_CASE("ObliviousTree: predict routes NaN right when default_left is false",
          "[oblivious_tree][predict][nan]")
{
    auto tree = one_split(fid_0, /*default_left=*/false);

    std::array<float, 1> nan_row{f_nan};

    CHECK(predict_one(tree, nan_row) == default_right_leaf);
}

TEST_CASE("ObliviousTree: predict honors per-level default_left for NaN routing",
          "[oblivious_tree][predict][nan]")
{
    ObliviousTree tree{
        ObliviousTree::LevelSplits{
            {.feature_id = fid_0, .threshold = root_threshold, .default_left = true},
            {.feature_id   = fid_1,
             .threshold    = right_subtree_threshold,
             .default_left = false},
        },
        ObliviousTree::LeafTable{leaf_ll_value, leaf_lr_value, leaf_rl_value,
                                 leaf_rr_value},
    };

    std::array<float, 2> nan_f0_low_f1{f_nan, f1_below_two};
    CHECK(predict_one(tree, nan_f0_low_f1) == leaf_ll_value);

    std::array<float, 2> low_f0_nan_f1{below_threshold, f_nan};
    CHECK(predict_one(tree, low_f0_nan_f1) == leaf_lr_value);
}

TEST_CASE("ObliviousTree: predict walks a multi-level tree to the correct leaf",
          "[oblivious_tree][predict]")
{
    auto tree = depth2_two_feature_tree();

    // (f0, f1) → expected leaf
    // f0 routes by root_threshold (1.0); f1 routes by right_subtree_threshold (2.0).
    std::array<float, 2> ll{below_threshold, f1_below_two}; // 00 → leaf_ll_value
    std::array<float, 2> lr{below_threshold, f1_above_two}; // 01 → leaf_lr_value
    std::array<float, 2> rl{above_threshold, f1_below_two}; // 10 → leaf_rl_value
    std::array<float, 2> rr{above_threshold, f1_above_two}; // 11 → leaf_rr_value

    CHECK(predict_one(tree, ll) == leaf_ll_value);
    CHECK(predict_one(tree, lr) == leaf_lr_value);
    CHECK(predict_one(tree, rl) == leaf_rl_value);
    CHECK(predict_one(tree, rr) == leaf_rr_value);
}

TEST_CASE("ObliviousTree: params() returns construction parameters",
          "[oblivious_tree][params]")
{
    auto single = single_leaf(single_leaf_value);
    CHECK(single.params().depth == params_single_depth);
    CHECK(single.params().n_leaves == params_single_leaves);

    auto multi = depth2_two_feature_tree();
    CHECK(multi.params().depth == params_multi_depth);
    CHECK(multi.params().n_leaves == params_multi_leaves);
}

TEST_CASE("ObliviousTree: batch predict strides correctly across multiple features",
          "[oblivious_tree][predict]")
{
    auto tree = depth2_two_feature_tree();

    std::array<float, 8> rows{
        below_threshold, f1_below_two, // row 0 → LL
        below_threshold, f1_above_two, // row 1 → LR
        above_threshold, f1_below_two, // row 2 → RL
        above_threshold, f1_above_two, // row 3 → RR
    };
    std::array<float, 4> out{};

    tree.predict(features_view{rows.data(), 4, 2}, out);

    CHECK(out[0] == leaf_ll_value);
    CHECK(out[1] == leaf_lr_value);
    CHECK(out[2] == leaf_rl_value);
    CHECK(out[3] == leaf_rr_value);
}

TEST_CASE("ObliviousTree: batch predict produces same results as single-row predict",
          "[oblivious_tree][predict]")
{
    auto tree = one_split(fid_0, /*default_left=*/true);

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
