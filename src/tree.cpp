#include "bonsai/tree.hpp"
#include "bonsai/types.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <utility>

namespace bonsai
{

DenseTree::DenseTree(Nodes nodes, Params params)
    : nodes_(std::move(nodes)), params_(params)
{
}

float DenseTree::walk_row(features_view X, row_id_t i) const
{
    node_id_t   index = 0;
    Node const *node  = &nodes_[index];
    while (node->feature_id != k_leaf_flag)
    {
        float v      = X[i, node->feature_id];
        bool  is_nan = std::isnan(v);
        // Binner is right-inclusive: bin b contains v ∈ (cuts[b-1], cuts[b]].
        // Grower routes bin <= split.bin_id left, so v == threshold goes left.
        bool less = !is_nan && (v <= node->threshold_or_value);
        // NOLINTNEXTLINE(readability-implicit-bool-conversion)
        bool go_left = less | (is_nan & node->default_left);
        index        = go_left ? node->left : node->right;
        node         = &nodes_[index];
    }
    return node->threshold_or_value;
}

void DenseTree::predict(features_view X, floats_out out) const
{
    assert(X.extent(0) == out.size());
    for (row_id_t i = 0; i < out.size(); ++i)
    {
        out[i] += walk_row(X, i);
    }
}

ObliviousTree::ObliviousTree(LevelSplits splits, LeafTable values)
    : splits_(std::move(splits)), leaf_table_(std::move(values)),
      params_{.depth = splits_.size(), .n_leaves = leaf_table_.size()}
{
    assert(leaf_table_.size() == (1ULL << splits_.size()));
}

float ObliviousTree::walk_row(features_view X, row_id_t i) const
{
    node_id_t index = 0;
    for (auto const &s : splits_)
    {
        float v      = X[i, s.feature_id];
        bool  is_nan = std::isnan(v);
        bool  less   = !is_nan && (v <= s.threshold);
        // NOLINTNEXTLINE(readability-implicit-bool-conversion)
        bool go_left = less | (is_nan & s.default_left);
        index        = (index << 1) | (go_left ? 0U : 1U);
    }
    return leaf_table_[index];
}

void ObliviousTree::predict(features_view X, floats_out out) const
{
    assert(X.extent(0) == out.size());
    for (row_id_t i = 0; i < out.size(); ++i)
    {
        out[i] += walk_row(X, i);
    }
}

} // namespace bonsai
