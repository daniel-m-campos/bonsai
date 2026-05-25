#include "bonsai/grower.hpp"
#include "bonsai/config/tree_config.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/split.hpp"
#include "bonsai/tree.hpp"
#include "bonsai/types.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace bonsai
{

namespace
{

inline float leaf_value(double grad, double hess, double lambda)
{
    return static_cast<float>(-grad / (hess + lambda));
}

inline void finalize_as_leaf(DenseTree::Nodes &nodes, SplitInput const &node,
                             double lambda, size_t &n_leaves, train_leaf_values &values)
{
    float const v  = leaf_value(node.total_grad(), node.total_hess(), lambda);
    nodes[node.id] = DenseTree::LeafNode{.value = v};
    for (row_id_t r : node.rows)
    {
        values[r] = v;
    }
    ++n_leaves;
}

inline void populate_from_rows(Dataset const &ds, floats_view grad, floats_view hess,
                               SplitInput &node)
{
    node.hists.reserve(ds.n_features());
    for (feature_id_t fid = 0; fid < ds.n_features(); ++fid)
    {
        Histogram h{ds.n_bins(fid)};
        auto const &bins = ds.feature_bins(fid);
        for (row_id_t r : node.rows)
        {
            h.add(bins[r], grad[r], hess[r]);
        }
        node.hists.push_back(std::move(h));
    }
}

SplitInput make_root(Dataset const &ds, floats_view grad, floats_view hess,
                     row_index_view row_indices)
{
    SplitInput root;
    root.id = 0;
    root.rows.assign(row_indices.begin(), row_indices.end());
    populate_from_rows(ds, grad, hess, root);
    return root;
}

inline std::pair<SplitInput, SplitInput>
split_node(Dataset const &ds, floats_view grad, floats_view hess, SplitInput parent,
           SplitOutput const &s, node_id_t left_id, node_id_t right_id)
{
    // 1. Partition parent.rows in place: lefts first, then rights.
    auto const &bins    = ds.feature_bins(s.feature_id);
    auto const last_bin = static_cast<bin_id_t>(ds.n_bins(s.feature_id) - 1);
    auto goes_left      = [&](row_id_t r)
    {
        bin_id_t const b = bins[r];
        if (b == last_bin)
        {
            return s.default_left;
        }
        return b <= s.bin_id;
    };
    auto mid = std::partition(parent.rows.begin(), parent.rows.end(), goes_left);

    SplitInput left;
    SplitInput right;
    left.id  = left_id;
    right.id = right_id;
    left.rows.assign(parent.rows.begin(), mid);
    right.rows.assign(mid, parent.rows.end());

    bool const left_smaller = left.rows.size() <= right.rows.size();
    SplitInput &small       = left_smaller ? left : right;
    SplitInput &large       = left_smaller ? right : left;
    populate_from_rows(ds, grad, hess, small);
    large.hists.reserve(ds.n_features());
    for (feature_id_t f = 0; f < ds.n_features(); ++f)
    {
        parent.hists[f] -= small.hists[f];
        large.hists.push_back(std::move(parent.hists[f]));
    }

    return {std::move(left), std::move(right)};
}

inline void update_nodes(Dataset const &ds, floats_view grad, floats_view hess,
                         TreeConfig const &config, std::vector<SplitInput> &current,
                         std::vector<SplitInput> &next,
                         std::vector<SplitOutput> const &splits,
                         DenseTree::Nodes &nodes, size_t &n_leaves,
                         train_leaf_values &values)
{
    for (node_id_t i = 0; i < current.size(); ++i)
    {
        auto &node        = current[i];
        auto const &split = splits[i];
        if (!split.valid) // assume valid incorporates all cfg parameter logic
        {
            finalize_as_leaf(nodes, node, config.lambda_l2, n_leaves, values);
            continue;
        }
        node_id_t const left_id = nodes.size();
        nodes.emplace_back(DenseTree::LeafNode{});
        node_id_t const right_id = nodes.size();
        nodes.emplace_back(DenseTree::LeafNode{});

        float const threshold = ds.mappers()[split.feature_id].cuts()[split.bin_id];
        nodes[node.id]        = DenseTree::InternalNode{
                   .feature_id   = split.feature_id,
                   .threshold    = threshold,
                   .left         = left_id,
                   .right        = right_id,
                   .default_left = split.default_left,
        };

        auto [left, right] =
            split_node(ds, grad, hess, std::move(node), split, left_id, right_id);
        next.push_back(std::move(left));
        next.push_back(std::move(right));
    }
    std::swap(current, next);
    next.clear();
}

} // namespace

template <NodeSplitFinder SplitterT>
DepthwiseGrower<SplitterT>::DepthwiseGrower(TreeConfig const &cfg) : config_(cfg)
{
}

template <NodeSplitFinder SplitterT>
auto DepthwiseGrower<SplitterT>::grow(Dataset const &ds, floats_view grad,
                                      floats_view hess, row_index_view row_indices)
    -> GrowResult<Tree>
{
    Tree::Nodes nodes;
    train_leaf_values values(ds.n_rows(), 0.0F);
    std::vector<SplitInput> current;
    std::vector<SplitInput> next;
    std::vector<SplitOutput> splits;
    current.push_back(make_root(ds, grad, hess, row_indices));
    nodes.emplace_back(DenseTree::LeafNode{});
    uint8_t depth   = 0;
    size_t n_leaves = 0;
    while (depth < config_.max_depth)
    {
        splits.clear();
        splits.reserve(current.size());
        for (auto const &node : current)
        {
            splits.push_back(SplitterT::find(node, config_));
        }
        update_nodes(ds, grad, hess, config_, current, next, splits, nodes, n_leaves,
                     values);
        if (current.empty())
        {
            break;
        }
        ++depth;
    }

    for (auto const &node : current)
    {
        finalize_as_leaf(nodes, node, config_.lambda_l2, n_leaves, values);
    }

    return {.tree   = Tree(std::move(nodes), {.depth = depth, .n_leaves = n_leaves}),
            .values = std::move(values)};
}

template class DepthwiseGrower<HistogramNodeSplitFinder>;

template <LevelSplitFinder SplitterT>
ObliviousGrower<SplitterT>::ObliviousGrower(TreeConfig const &cfg) : config_(cfg)
{
}

template <LevelSplitFinder SplitterT>
auto ObliviousGrower<SplitterT>::grow(Dataset const &ds, floats_view grad,
                                      floats_view hess, row_index_view row_indices)
    -> GrowResult<Tree>
{
    Tree::LevelSplits level_splits;
    Tree::LeafTable leaf_table;
    train_leaf_values values(ds.n_rows(), 0.0F);

    std::vector<SplitInput> frontier;
    std::vector<SplitInput> next;
    frontier.push_back(make_root(ds, grad, hess, row_indices));

    size_t depth = 0;
    while (depth < config_.max_depth)
    {
        SplitOutput const split = SplitterT::find(frontier, config_);
        if (!split.valid)
        {
            break;
        }
        float const threshold = ds.mappers()[split.feature_id].cuts()[split.bin_id];
        level_splits.push_back({.feature_id   = split.feature_id,
                                .threshold    = threshold,
                                .default_left = split.default_left});
        next.reserve(frontier.size() * 2);
        for (auto &node : frontier)
        {
            auto [left, right] =
                split_node(ds, grad, hess, std::move(node), split, 0, 0);
            next.push_back(std::move(left));
            next.push_back(std::move(right));
        }
        std::swap(frontier, next);
        next.clear();
        ++depth;
    }

    leaf_table.reserve(frontier.size());
    for (auto const &leaf : frontier)
    {
        float const v =
            leaf_value(leaf.total_grad(), leaf.total_hess(), config_.lambda_l2);
        leaf_table.push_back(v);
        for (row_id_t r : leaf.rows)
        {
            values[r] = v;
        }
    }

    return {.tree   = Tree(std::move(level_splits), std::move(leaf_table)),
            .values = std::move(values)};
}

template class ObliviousGrower<HistogramLevelSplitFinder>;

} // namespace bonsai
