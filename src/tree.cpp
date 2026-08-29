#include "bonsai/tree.hpp"
#include "bonsai/parallel.hpp"
#include "bonsai/types.hpp"

#include <algorithm>
#include <array>
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
        bool const right = routes_right(X[i, node->feature_id],
                                        node->threshold_or_value, node->default_left);
        index            = right ? node->right : node->left;
        node             = &nodes_[index];
    }
    return index;
}

float DenseTree::value_for(features_view X, row_id_t i) const
{
    return nodes_[leaf_for(X, i)].threshold_or_value;
}

void DenseTree::predict(features_view X, floats_out out) const
{
    assert(X.extent(0) == out.size());
    parallel::for_each_index(out.size(), [&](size_t i)
                             { out[i] += value_for(X, static_cast<row_id_t>(i)); });
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
    auto               build = [&](auto &&self, size_t lvl,
                     size_t path) -> std::pair<node_id_t, float>
    {
        if (lvl == depth)
        {
            nodes.push_back(DenseTree::leaf(leaves[path]));
            covers.push_back(lc[path]);
            node_gains.push_back(0.0F);
            return {static_cast<node_id_t>(nodes.size() - 1), lc[path]};
        }
        auto const id = static_cast<node_id_t>(nodes.size());
        nodes.push_back(DenseTree::leaf(0.0F));
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
        bool const right =
            routes_right(X[i, s.feature_id], s.threshold, s.default_left);
        index = (index << 1) | static_cast<node_id_t>(right);
    }
    return index;
}

float ObliviousTree::value_for(features_view X, row_id_t i) const
{
    return leaf_table_[leaf_for(X, i)];
}

void ObliviousTree::predict(features_view X, floats_out out) const
{
    assert(X.extent(0) == out.size());
    parallel::for_each_index(out.size(), [&](size_t i)
                             { out[i] += value_for(X, static_cast<row_id_t>(i)); });
}

namespace
{
// perf: DenseWalk's crossover at 8 threads (64 cols, depth 8, 100
// trees): parallel wins from 128 rows on both M2 (507 vs 1015 ns/row)
// and EPYC 9654 (902 vs 1326); serial wins at 64 on both.
constexpr size_t k_walk_parallel_floor = 128;
// perf: the blocked oblivious walk has its own crossover, because a
// serial block is ~5x a serial row and a small predict is only a few
// blocks across a team: at 128 rows parallel LOSES on both hosts (258
// vs 190 M2, 476 vs 200 EPYC 9654) and 512 is the first clear M2 win
// (115 vs 183), with the EPYC near break-even between its 256 loss and
// its 1024 win.
constexpr size_t k_blocked_parallel_floor = 512;
constexpr size_t k_dense_walk_width       = 8;
constexpr size_t k_row_block              = 64;

} // namespace

DenseWalk::DenseWalk(std::span<DenseTree const> trees)
{
    std::vector<size_t> depths;
    depths.reserve(trees.size());
    for (auto const &tree : trees)
    {
        auto const base = static_cast<uint32_t>(nodes_.size());
        roots_.push_back(base);
        depths.push_back(tree.params().depth);
        for (auto const &n : tree.nodes())
        {
            bool const leaf = n.feature_id == DenseTree::k_leaf_flag;
            auto const self = static_cast<uint32_t>(nodes_.size());
            nodes_.push_back({.threshold = n.threshold_or_value,
                              .feature   = leaf ? 0 : n.feature_id,
                              .left      = leaf ? self : base + n.left,
                              .right     = leaf ? self : base + n.right});
            default_left_.push_back(n.default_left ? 1 : 0);
        }
    }
    for (size_t g = 0; g * k_dense_walk_width < trees.size(); ++g)
    {
        size_t const t0   = g * k_dense_walk_width;
        size_t const t1   = std::min(t0 + k_dense_walk_width, trees.size());
        size_t       deep = 0;
        for (size_t t = t0; t < t1; ++t)
        {
            deep = std::max(deep, depths[t]);
        }
        group_depth_.push_back(static_cast<uint32_t>(deep));
    }
}

void DenseWalk::accumulate(features_view X, size_t n_trees, floats_out out) const
{
    assert(X.extent(0) == out.size());
    assert(n_trees <= roots_.size());
    size_t const n_cols = X.extent(1);
    int const workers = out.size() < k_walk_parallel_floor ? 1 : parallel::n_threads();
    parallel::for_each_index_on(
        workers, out.size(),
        [&](size_t i)
        {
            float const *row = X.data_handle() + (i * n_cols);
            float        sum = 0.0F;
            size_t       t   = 0;
            for (; t + k_dense_walk_width <= n_trees; t += k_dense_walk_width)
            {
                std::array<uint32_t, k_dense_walk_width> idx;
                for (size_t w = 0; w < k_dense_walk_width; ++w)
                {
                    idx[w] = roots_[t + w];
                }
                uint32_t const depth = group_depth_[t / k_dense_walk_width];
                for (uint32_t d = 0; d < depth; ++d)
                {
                    for (size_t w = 0; w < k_dense_walk_width; ++w)
                    {
                        PackedNode const &n = nodes_[idx[w]];
                        bool const right    = routes_right(row[n.feature], n.threshold,
                                                           default_left_[idx[w]] != 0);
                        idx[w]              = right ? n.right : n.left;
                    }
                }
                for (size_t w = 0; w < k_dense_walk_width; ++w)
                {
                    sum += nodes_[idx[w]].threshold;
                }
            }
            for (; t < n_trees; ++t)
            {
                uint32_t       idx   = roots_[t];
                uint32_t const depth = group_depth_[t / k_dense_walk_width];
                for (uint32_t d = 0; d < depth; ++d)
                {
                    PackedNode const &n     = nodes_[idx];
                    bool const        right = routes_right(row[n.feature], n.threshold,
                                                           default_left_[idx] != 0);
                    idx                     = right ? n.right : n.left;
                }
                sum += nodes_[idx].threshold;
            }
            out[i] += sum;
        });
}

ObliviousWalk::ObliviousWalk(std::span<ObliviousTree const> trees)
{
    split_off_.reserve(trees.size() + 1);
    leaf_off_.reserve(trees.size() + 1);
    split_off_.push_back(0);
    leaf_off_.push_back(0);
    for (auto const &tree : trees)
    {
        for (auto const &s : tree.splits())
        {
            feat_.push_back(s.feature_id);
            thr_.push_back(s.threshold);
            default_left_.push_back(s.default_left ? 1 : 0);
        }
        for (float const v : tree.leaf_table())
        {
            leaf_.push_back(v);
        }
        split_off_.push_back(static_cast<uint32_t>(feat_.size()));
        leaf_off_.push_back(static_cast<uint32_t>(leaf_.size()));
    }
}

void ObliviousWalk::walk_rows(features_view X, size_t n_trees, size_t row0,
                              size_t count, floats_out out) const
{
    size_t const n_cols = X.extent(1);
    for (size_t i = row0; i < row0 + count; ++i)
    {
        float const *row = X.data_handle() + (i * n_cols);
        float        sum = 0.0F;
        for (size_t t = 0; t < n_trees; ++t)
        {
            uint32_t idx = 0;
            for (uint32_t l = split_off_[t]; l < split_off_[t + 1]; ++l)
            {
                bool const right =
                    routes_right(row[feat_[l]], thr_[l], default_left_[l] != 0);
                idx = (idx << 1) | static_cast<uint32_t>(right);
            }
            sum += leaf_[leaf_off_[t] + idx];
        }
        out[i] += sum;
    }
}

void ObliviousWalk::walk_block(features_view X, size_t n_trees, size_t row0,
                               float *scratch, floats_out out) const
{
    size_t const n_cols = X.extent(1);
    for (size_t w = 0; w < k_row_block; ++w)
    {
        float const *row = X.data_handle() + ((row0 + w) * n_cols);
        for (size_t c = 0; c < n_cols; ++c)
        {
            scratch[(c * k_row_block) + w] = row[c];
        }
    }
    std::array<float, k_row_block>    sum{};
    std::array<uint32_t, k_row_block> idx;
    for (size_t t = 0; t < n_trees; ++t)
    {
        idx.fill(0);
        for (uint32_t l = split_off_[t]; l < split_off_[t + 1]; ++l)
        {
            float const *col  = scratch + (size_t{feat_[l]} * k_row_block);
            float const  thr  = thr_[l];
            bool const   deft = default_left_[l] != 0;
            for (size_t w = 0; w < k_row_block; ++w)
            {
                idx[w] = (idx[w] << 1) |
                         static_cast<uint32_t>(routes_right(col[w], thr, deft));
            }
        }
        float const *leaves = leaf_.data() + leaf_off_[t];
        for (size_t w = 0; w < k_row_block; ++w)
        {
            sum[w] += leaves[idx[w]];
        }
    }
    for (size_t w = 0; w < k_row_block; ++w)
    {
        out[row0 + w] += sum[w];
    }
}

void ObliviousWalk::accumulate(features_view X, size_t n_trees, floats_out out) const
{
    assert(X.extent(0) == out.size());
    assert(n_trees < split_off_.size());
    size_t const n_cols   = X.extent(1);
    size_t const n_rows   = out.size();
    size_t const n_blocks = n_rows / k_row_block;
    size_t const tail     = n_rows - (n_blocks * k_row_block);
    int const workers = n_rows < k_blocked_parallel_floor ? 1 : parallel::n_threads();
    parallel::for_each_index_on(workers, n_blocks,
                                [&](size_t b)
                                {
                                    std::vector<float> scratch(n_cols * k_row_block);
                                    walk_block(X, n_trees, b * k_row_block,
                                               scratch.data(), out);
                                });
    if (tail > 0)
    {
        walk_rows(X, n_trees, n_blocks * k_row_block, tail, out);
    }
}

} // namespace bonsai
