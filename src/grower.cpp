#include "bonsai/grower.hpp"
#include "bonsai/config/tree_config.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/parallel.hpp"
#include "bonsai/split.hpp"
#include "bonsai/tree.hpp"
#include "bonsai/types.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <random>
#include <span>
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

using feature_view = std::span<feature_id_t const>;

// Per-tree feature subsample: a sorted draw of ceil(fraction * n) distinct
// feature ids. fraction >= 1 selects everything.
std::vector<feature_id_t> sample_features(size_t n_features, float fraction,
                                          std::mt19937 &rng)
{
    std::vector<feature_id_t> all(n_features);
    std::iota(all.begin(), all.end(), feature_id_t{0});
    if (fraction >= 1.0F || n_features <= 1)
    {
        return all;
    }
    auto const k = std::max<size_t>(
        1, static_cast<size_t>(std::ceil(fraction * static_cast<float>(n_features))));
    std::vector<feature_id_t> selected;
    selected.reserve(k);
    std::sample(all.begin(), all.end(), std::back_inserter(selected), k, rng);
    return selected; // std::sample preserves order -> sorted
}

inline void finalize_as_leaf(DenseTree::Nodes &nodes, SplitInput const &node,
                             double lambda, size_t &n_leaves, train_leaf_values &values)
{
    float const v  = leaf_value(node.total_grad(), node.total_hess(), lambda);
    nodes[node.id] = DenseTree::leaf(v);
    for (row_id_t r : node.rows)
    {
        values[r] = v;
    }
    ++n_leaves;
}

// Builds histograms for `selected` features only; unselected slots stay
// zero-binned placeholders the split finders skip.
inline void populate_from_rows(Dataset const &ds, floats_view grad, floats_view hess,
                               SplitInput &node, feature_view selected)
{
    // Gather grad/hess into node-row order once, so every feature's scan
    // below reads them sequentially instead of re-walking the full arrays
    // with scattered indices (n_features x full-array traffic otherwise).
    static thread_local std::vector<float> ordered_grad;
    static thread_local std::vector<float> ordered_hess;
    size_t const n = node.rows.size();
    ordered_grad.resize(n);
    ordered_hess.resize(n);
    // Capture raw pointers: naming the thread_local inside the parallel
    // region would resolve to each worker's own (empty) vector.
    float *const og = ordered_grad.data();
    float *const oh = ordered_hess.data();
    parallel::for_each_index(n,
                             [&, og, oh](size_t k)
                             {
                                 row_id_t const r = node.rows[k];
                                 og[k]            = grad[r];
                                 oh[k]            = hess[r];
                             });

    node.hists.reserve(ds.n_features());
    size_t j = 0;
    for (feature_id_t fid = 0; fid < ds.n_features(); ++fid)
    {
        bool const sel = j < selected.size() && selected[j] == fid;
        node.hists.emplace_back(sel ? ds.n_bins(fid) : 0);
        j += sel ? 1 : 0;
    }
    // Feature-parallel: each feature's histogram is owned by one thread and
    // filled in row order, so results are bit-identical at any thread count.
    parallel::for_each_index(selected.size(),
                             [&, og, oh](size_t s)
                             {
                                 feature_id_t const fid  = selected[s];
                                 Histogram         &h    = node.hists[fid];
                                 auto const        &bins = ds.feature_bins(fid);
                                 for (size_t k = 0; k < n; ++k)
                                 {
                                     h.add(bins[node.rows[k]], og[k], oh[k]);
                                 }
                             });
}

SplitInput make_root(Dataset const &ds, floats_view grad, floats_view hess,
                     row_index_view row_indices, feature_view selected)
{
    SplitInput root;
    root.id = 0;
    root.rows.assign(row_indices.begin(), row_indices.end());
    populate_from_rows(ds, grad, hess, root, selected);
    return root;
}

inline std::pair<SplitInput, SplitInput>
split_node(Dataset const &ds, floats_view grad, floats_view hess, SplitInput parent,
           SplitOutput const &s, node_id_t left_id, node_id_t right_id,
           feature_view selected)
{
    // 1. Scatter parent.rows into the children in one stable pass. Stability
    //    keeps every node's rows ascending (the root's are iota), so later
    //    per-feature bin lookups walk memory near-sequentially.
    auto const &bins      = ds.feature_bins(s.feature_id);
    auto const  last_bin  = static_cast<bin_id_t>(ds.n_bins(s.feature_id) - 1);
    auto        goes_left = [&](row_id_t r)
    {
        bin_id_t const b = bins[r];
        if (b == last_bin)
        {
            return s.default_left;
        }
        return b <= s.bin_id;
    };

    SplitInput left;
    SplitInput right;
    left.id  = left_id;
    right.id = right_id;
    size_t n_left = 0;
    for (row_id_t const r : parent.rows)
    {
        n_left += goes_left(r) ? 1 : 0;
    }
    left.rows.resize(n_left);
    right.rows.resize(parent.rows.size() - n_left);
    size_t li = 0;
    size_t ri = 0;
    for (row_id_t const r : parent.rows)
    {
        if (goes_left(r))
        {
            left.rows[li++] = r;
        }
        else
        {
            right.rows[ri++] = r;
        }
    }

    bool const  left_smaller = left.rows.size() <= right.rows.size();
    SplitInput &small        = left_smaller ? left : right;
    SplitInput &large        = left_smaller ? right : left;
    populate_from_rows(ds, grad, hess, small, selected);
    large.hists.reserve(ds.n_features());
    for (feature_id_t f = 0; f < ds.n_features(); ++f)
    {
        large.hists.push_back(std::move(parent.hists[f]));
    }
    // Unselected slots are zero-binned on both sides: no-op subtraction.
    parallel::for_each_index(ds.n_features(),
                             [&](size_t f) { large.hists[f] -= small.hists[f]; });

    return {std::move(left), std::move(right)};
}

inline void update_nodes(Dataset const &ds, floats_view grad, floats_view hess,
                         TreeConfig const &config, std::vector<SplitInput> &current,
                         std::vector<SplitInput>        &next,
                         std::vector<SplitOutput> const &splits,
                         DenseTree::Nodes &nodes, size_t &n_leaves,
                         train_leaf_values &values, feature_view selected,
                         std::vector<bin_id_t> &split_bins)
{
    for (node_id_t i = 0; i < current.size(); ++i)
    {
        auto       &node  = current[i];
        auto const &split = splits[i];
        if (!split.valid) // assume valid incorporates all cfg parameter logic
        {
            finalize_as_leaf(nodes, node, config.lambda_l2, n_leaves, values);
            continue;
        }
        node_id_t const left_id = nodes.size();
        nodes.emplace_back(DenseTree::leaf(0.0F));
        node_id_t const right_id = nodes.size();
        nodes.emplace_back(DenseTree::leaf(0.0F));
        split_bins.resize(nodes.size(), 0);
        split_bins[node.id] = split.bin_id;

        float const threshold = ds.mappers()[split.feature_id].cuts()[split.bin_id];
        nodes[node.id] = DenseTree::internal(split.feature_id, threshold, left_id,
                                             right_id, split.default_left);

        auto [left, right] = split_node(ds, grad, hess, std::move(node), split,
                                        left_id, right_id, selected);
        next.push_back(std::move(left));
        next.push_back(std::move(right));
    }
    std::swap(current, next);
    next.clear();
}

// A leaf awaiting expansion: its histograms/rows, its best split (heap key),
// and its depth (to enforce the max_depth cap on children).
struct Candidate
{
    SplitInput  node;
    SplitOutput split;
    uint8_t     depth = 0;
};

// Rows not in `sampled` (sorted) never reach a leaf during growth, so
// finalize_as_leaf can't stamp their train values — yet the booster's score
// accumulator covers every row. Route them through the finished tree in bin
// space (split_bins[i] is node i's split bin; bin <= split_bin routes left,
// exactly matching the float-threshold predict path) and fill their values.
// Skipping this desynchronizes training scores from the real model for any
// sampler that drops rows: gradients go stale and GOSS-style samplers, which
// re-pick rows by |grad|, diverge outright.
inline void route_unsampled(Dataset const &ds, DenseTree::Nodes const &nodes,
                            std::vector<bin_id_t> const &split_bins,
                            row_index_view sampled, train_leaf_values &values)
{
    size_t const n = ds.n_rows();
    if (sampled.size() == n)
    {
        return;
    }
    std::vector<row_id_t> oob;
    oob.reserve(n - sampled.size());
    size_t j = 0;
    for (row_id_t r = 0; r < n; ++r)
    {
        if (j < sampled.size() && sampled[j] == r)
        {
            ++j;
            continue;
        }
        oob.push_back(r);
    }
    parallel::for_each_index(
        oob.size(),
        [&](size_t k)
        {
            row_id_t const r   = oob[k];
            node_id_t      idx = 0;
            while (!DenseTree::is_leaf(nodes[idx]))
            {
                auto const    &nd   = nodes[idx];
                auto const    &bins = ds.feature_bins(nd.feature_id);
                auto const     last =
                    static_cast<bin_id_t>(ds.n_bins(nd.feature_id) - 1);
                bin_id_t const b = bins[r];
                bool const     left =
                    (b == last) ? nd.default_left : b <= split_bins[idx];
                idx = left ? nd.left : nd.right;
            }
            values[r] = nodes[idx].threshold_or_value;
        });
}

} // namespace

template <NodeSplitFinder SplitterT>
DepthwiseGrower<SplitterT>::DepthwiseGrower(TreeConfig const &cfg)
    : config_(cfg), feature_rng_(cfg.feature_seed)
{
}

template <NodeSplitFinder SplitterT>
auto DepthwiseGrower<SplitterT>::grow(Dataset const &ds, floats_view grad,
                                      floats_view hess, row_index_view row_indices)
    -> GrowResult<Tree>
{
    Tree::Nodes              nodes;
    train_leaf_values        values(ds.n_rows(), 0.0F);
    std::vector<SplitInput>  current;
    std::vector<SplitInput>  next;
    std::vector<SplitOutput> splits;
    std::vector<bin_id_t>    split_bins(1, 0);
    auto const               selected =
        sample_features(ds.n_features(), config_.feature_fraction, feature_rng_);
    current.push_back(make_root(ds, grad, hess, row_indices, selected));
    nodes.emplace_back(DenseTree::leaf(0.0F));
    uint8_t depth    = 0;
    size_t  n_leaves = 0;
    while (depth < config_.max_depth)
    {
        splits.clear();
        splits.reserve(current.size());
        for (auto const &input : current)
        {
            splits.push_back(SplitterT::find(input, config_));
        }
        update_nodes(ds, grad, hess, config_, current, next, splits, nodes, n_leaves,
                     values, selected, split_bins);
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
    route_unsampled(ds, nodes, split_bins, row_indices, values);

    return {.tree   = Tree(std::move(nodes), {.depth = depth, .n_leaves = n_leaves}),
            .values = std::move(values)};
}

template class DepthwiseGrower<HistogramNodeSplitFinder>;

template <LevelSplitFinder SplitterT>
ObliviousGrower<SplitterT>::ObliviousGrower(TreeConfig const &cfg)
    : config_(cfg), feature_rng_(cfg.feature_seed)
{
}

template <LevelSplitFinder SplitterT>
auto ObliviousGrower<SplitterT>::grow(Dataset const &ds, floats_view grad,
                                      floats_view hess, row_index_view row_indices)
    -> GrowResult<Tree>
{
    Tree::LevelSplits level_splits;
    Tree::LeafTable   leaf_table;
    train_leaf_values values(ds.n_rows(), 0.0F);

    std::vector<SplitInput> frontier;
    std::vector<SplitInput> next;
    std::vector<bin_id_t>   level_bins;
    auto const              selected =
        sample_features(ds.n_features(), config_.feature_fraction, feature_rng_);
    frontier.push_back(make_root(ds, grad, hess, row_indices, selected));

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
        level_bins.push_back(split.bin_id);
        next.reserve(frontier.size() * 2);
        for (auto &node : frontier)
        {
            auto [left, right] =
                split_node(ds, grad, hess, std::move(node), split, 0, 0, selected);
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

    // Same stale-score hazard as route_unsampled: rows the sampler dropped
    // still need this tree's contribution in their train values.
    if (row_indices.size() != ds.n_rows())
    {
        std::vector<row_id_t> oob;
        oob.reserve(ds.n_rows() - row_indices.size());
        size_t j = 0;
        for (row_id_t r = 0; r < ds.n_rows(); ++r)
        {
            if (j < row_indices.size() && row_indices[j] == r)
            {
                ++j;
                continue;
            }
            oob.push_back(r);
        }
        parallel::for_each_index(
            oob.size(),
            [&](size_t k)
            {
                row_id_t const r     = oob[k];
                size_t         index = 0;
                for (size_t lvl = 0; lvl < level_splits.size(); ++lvl)
                {
                    auto const    &s    = level_splits[lvl];
                    auto const    &bins = ds.feature_bins(s.feature_id);
                    auto const     last =
                        static_cast<bin_id_t>(ds.n_bins(s.feature_id) - 1);
                    bin_id_t const b = bins[r];
                    bool const     left =
                        (b == last) ? s.default_left : b <= level_bins[lvl];
                    index = (index << 1U) | (left ? 0U : 1U);
                }
                values[r] = leaf_table[index];
            });
    }

    return {.tree   = Tree(std::move(level_splits), std::move(leaf_table)),
            .values = std::move(values)};
}

template class ObliviousGrower<HistogramLevelSplitFinder>;

template <NodeSplitFinder SplitterT>
LeafwiseGrower<SplitterT>::LeafwiseGrower(TreeConfig const &cfg)
    : config_(cfg), feature_rng_(cfg.feature_seed)
{
}

template <NodeSplitFinder SplitterT>
auto LeafwiseGrower<SplitterT>::grow(Dataset const &ds, floats_view grad,
                                     floats_view hess, row_index_view row_indices)
    -> GrowResult<Tree>
{
    Tree::Nodes       nodes;
    train_leaf_values values(ds.n_rows(), 0.0F);

    // Max-heap on gain; ties broken by lower node id so growth is deterministic.
    auto gain_less = [](Candidate const &a, Candidate const &b)
    {
        if (a.split.gain != b.split.gain)
        {
            return a.split.gain < b.split.gain;
        }
        return a.node.id > b.node.id;
    };

    std::vector<Candidate>  heap;
    std::vector<SplitInput> pending;
    std::vector<bin_id_t>   split_bins(1, 0);

    auto const selected =
        sample_features(ds.n_features(), config_.feature_fraction, feature_rng_);
    SplitInput root = make_root(ds, grad, hess, row_indices, selected);
    nodes.emplace_back(DenseTree::leaf(0.0F));

    size_t  n_leaves    = 0;
    size_t  live_leaves = 1;
    uint8_t depth       = 0;

    auto has_budget = [&]
    { return config_.max_leaves == 0 || live_leaves < config_.max_leaves; };

    SplitOutput const root_split = SplitterT::find(root, config_);
    if (root_split.valid && config_.max_depth > 0)
    {
        heap.push_back({std::move(root), root_split, 0});
    }
    else
    {
        pending.push_back(std::move(root));
    }

    while (!heap.empty() && has_budget())
    {
        std::pop_heap(heap.begin(), heap.end(), gain_less);
        Candidate c = std::move(heap.back());
        heap.pop_back();

        node_id_t const left_id = nodes.size();
        nodes.emplace_back(DenseTree::leaf(0.0F));
        node_id_t const right_id = nodes.size();
        nodes.emplace_back(DenseTree::leaf(0.0F));
        split_bins.resize(nodes.size(), 0);
        split_bins[c.node.id] = c.split.bin_id;

        float const threshold = ds.mappers()[c.split.feature_id].cuts()[c.split.bin_id];
        nodes[c.node.id] = DenseTree::internal(c.split.feature_id, threshold, left_id,
                                               right_id, c.split.default_left);

        auto [left, right] = split_node(ds, grad, hess, std::move(c.node), c.split,
                                        left_id, right_id, selected);
        ++live_leaves;

        uint8_t const child_depth = c.depth + 1;
        depth                     = std::max(depth, child_depth);
        for (SplitInput *child : {&left, &right})
        {
            SplitOutput const split = (child_depth < config_.max_depth)
                                          ? SplitterT::find(*child, config_)
                                          : SplitOutput{};
            if (split.valid)
            {
                heap.push_back({std::move(*child), split, child_depth});
                std::push_heap(heap.begin(), heap.end(), gain_less);
            }
            else
            {
                pending.push_back(std::move(*child));
            }
        }
    }

    for (auto const &c : heap)
    {
        finalize_as_leaf(nodes, c.node, config_.lambda_l2, n_leaves, values);
    }
    for (auto const &leaf : pending)
    {
        finalize_as_leaf(nodes, leaf, config_.lambda_l2, n_leaves, values);
    }
    route_unsampled(ds, nodes, split_bins, row_indices, values);

    return {.tree   = Tree(std::move(nodes), {.depth = depth, .n_leaves = n_leaves}),
            .values = std::move(values)};
}

template class LeafwiseGrower<HistogramNodeSplitFinder>;

} // namespace bonsai
