#include "bonsai/tree.hpp"
#include "bonsai/types.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <utility>
#include <variant>

namespace bonsai
{

namespace
{

template <Tree T>
void predict(T const &tree, floats_view rows, size_t n_features, floats_out out)
{
    assert(rows.size() == out.size() * n_features);
    for (row_id_t i = 0; i < out.size(); ++i)
    {
        out[i] = tree.predict(rows.subspan(i * n_features, n_features));
    }
}
} // namespace

DenseTree::DenseTree(Nodes nodes, Params params)
    : nodes_(std::move(nodes)), params_(params)
{
}

float DenseTree::predict(floats_view row) const
{
    node_id_t index = 0;
    while (auto const *node = std::get_if<InternalNode>(&nodes_[index]))
    {
        float v     = row[node->feature_id];
        bool is_nan = std::isnan(v);
        bool less   = !is_nan && (v < node->threshold);
        // NOLINTNEXTLINE(readability-implicit-bool-conversion)
        bool go_left = less | (is_nan & node->default_left);
        index        = go_left ? node->left : node->right;
    }
    return std::get<LeafNode>(nodes_[index]).value;
}

void DenseTree::predict(floats_view rows, size_t n_features, floats_out out) const
{
    ::bonsai::predict(*this, rows, n_features, out);
}

ObliviousTree::ObliviousTree(LevelSplits splits, LeafValues values)
    : splits_(std::move(splits)), leaf_values_(std::move(values)),
      params_{.depth = splits_.size(), .n_leaves = leaf_values_.size()}
{
    assert(leaf_values_.size() == (1ULL << splits_.size()));
}

float ObliviousTree::predict(floats_view row) const
{
    node_id_t index = 0;
    for (auto const &s : splits_)
    {
        float v     = row[s.feature_id];
        bool is_nan = std::isnan(v);
        bool less   = !is_nan && (v < s.threshold);
        // NOLINTNEXTLINE(readability-implicit-bool-conversion)
        bool go_left = less | (is_nan & s.default_left);
        index        = (index << 1) | (go_left ? 0U : 1U);
    }
    return leaf_values_[index];
}

void ObliviousTree::predict(floats_view rows, size_t n_features, floats_out out) const
{
    ::bonsai::predict(*this, rows, n_features, out);
}

} // namespace bonsai
