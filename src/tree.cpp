#include "bonsai/tree.hpp"
#include "bonsai/parallel.hpp"
#include "bonsai/types.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace bonsai
{

DenseTree::DenseTree(Nodes nodes, Params params, std::vector<float> split_gains,
                     std::vector<float> covers)
    : nodes_(std::move(nodes)), params_(params), split_gains_(std::move(split_gains)),
      covers_(std::move(covers))
{
}

node_id_t DenseTree::leaf_for(features_view X, row_id_t i) const
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
    return index;
}

float DenseTree::walk_row(features_view X, row_id_t i) const
{
    return nodes_[leaf_for(X, i)].threshold_or_value;
}

void DenseTree::predict(features_view X, floats_out out) const
{
    assert(X.extent(0) == out.size());
    parallel::for_each_index(out.size(), [&](size_t i)
                             { out[i] += walk_row(X, static_cast<row_id_t>(i)); });
}

ObliviousTree::ObliviousTree(LevelSplits splits, LeafTable values,
                             std::vector<float> level_gains,
                             std::vector<float> leaf_covers)
    : splits_(std::move(splits)), leaf_table_(std::move(values)),
      leaf_covers_(std::move(leaf_covers)),
      params_{.depth = splits_.size(), .n_leaves = leaf_table_.size()},
      level_gains_(std::move(level_gains))
{
    assert(leaf_table_.size() == (1ULL << splits_.size()));
    assert(leaf_covers_.empty() || leaf_covers_.size() == leaf_table_.size());
}

DenseTree dense_equivalent(ObliviousTree const &tree)
{
    auto const &splits = tree.splits();
    auto const &leaves = tree.leaf_table();
    auto const &lc     = tree.leaf_covers();
    if (lc.size() != leaves.size())
    {
        throw std::invalid_argument(
            "oblivious tree carries no covers (model predates leaf-cover "
            "recording); retrain to enable pred_contribs");
    }
    auto const        &gains = tree.level_gains();
    size_t const       depth = splits.size();
    DenseTree::Nodes   nodes;
    std::vector<float> covers;
    std::vector<float> node_gains;
    // Total training cover of the subtree rooted at (lvl, path): the sum of
    // its leaf slots' covers.
    auto subtree_cover = [&](size_t lvl, size_t path)
    {
        size_t const shift = depth - lvl;
        float        total = 0.0F;
        for (size_t j = path << shift; j < (path + 1) << shift; ++j)
        {
            total += lc[j];
        }
        return total;
    };
    // Depth-first expansion; leaf slot = the walk's path bits (level 0 is
    // the most significant, left = 0), matching ObliviousTree::leaf_for.
    // Zero-cover subtrees collapse into their sibling: no training row ever
    // reached them (the broadcast split created the slot, not the data), and
    // TreeSHAP's cover fractions would be 0/0 inside them. Training-row
    // predictions are unaffected, which is what the efficiency property
    // checks against.
    auto build = [&](auto &&self, size_t lvl,
                     size_t path) -> std::pair<node_id_t, float>
    {
        if (lvl == depth)
        {
            nodes.push_back(DenseTree::leaf(leaves[path]));
            covers.push_back(lc[path]);
            node_gains.push_back(0.0F);
            return {static_cast<node_id_t>(nodes.size() - 1), lc[path]};
        }
        bool const left_dead  = subtree_cover(lvl + 1, path << 1) == 0.0F;
        bool const right_dead = subtree_cover(lvl + 1, (path << 1) | 1U) == 0.0F;
        if (left_dead)
        {
            return self(self, lvl + 1, (path << 1) | 1U);
        }
        if (right_dead)
        {
            return self(self, lvl + 1, path << 1);
        }
        auto const id = static_cast<node_id_t>(nodes.size());
        nodes.push_back(DenseTree::leaf(0.0F)); // placeholder, patched below
        covers.push_back(0.0F);
        node_gains.push_back(lvl < gains.size() ? gains[lvl] : 0.0F);
        auto const [l, cl] = self(self, lvl + 1, path << 1);
        auto const [r, cr] = self(self, lvl + 1, (path << 1) | 1U);
        auto const &sp     = splits[lvl];
        nodes[id] =
            DenseTree::internal(sp.feature_id, sp.threshold, l, r, sp.default_left);
        covers[id] = cl + cr;
        return {id, covers[id]};
    };
    build(build, 0, 0);
    return DenseTree{std::move(nodes),
                     {.depth = depth, .n_leaves = leaves.size()},
                     std::move(node_gains),
                     std::move(covers)};
}

node_id_t ObliviousTree::leaf_for(features_view X, row_id_t i) const
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
    return index;
}

float ObliviousTree::walk_row(features_view X, row_id_t i) const
{
    return leaf_table_[leaf_for(X, i)];
}

void ObliviousTree::predict(features_view X, floats_out out) const
{
    assert(X.extent(0) == out.size());
    parallel::for_each_index(out.size(), [&](size_t i)
                             { out[i] += walk_row(X, static_cast<row_id_t>(i)); });
}

} // namespace bonsai
