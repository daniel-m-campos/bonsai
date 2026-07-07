#pragma once

// Template definitions for the three growers, shared by every explicit
// instantiation TU: grower.cpp instantiates the CPU-engine variants and
// cuda/grower_cuda.cpp the CUDA one. Only src/ TUs include this header.

#include "bonsai/config/errors.hpp"
#include "bonsai/config/tree_config.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/parallel.hpp"
#include "bonsai/split.hpp"
#include "bonsai/tree.hpp"
#include "bonsai/types.hpp"
#include "level_step.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <numeric>
#include <print>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace bonsai::grower_detail
{

inline float leaf_value(double grad, double hess, TreeConfig const &config)
{
    return static_cast<float>(-l1_thresholded(grad, config.lambda_l1) /
                              (hess + config.lambda_l2));
}

// Children inherit the parent's leaf-value bounds; a split on a monotone
// feature additionally fences both sides at the midpoint of the child
// weights (xgboost's scheme), so descendant leaves can't cross.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
inline void propagate_monotone_bounds(double parent_lo, double parent_hi,
                                      SplitOutput const &s, TreeConfig const &config,
                                      SplitInput &left, SplitInput &right)
{
    left.lo      = parent_lo;
    left.hi      = parent_hi;
    right.lo     = parent_lo;
    right.hi     = parent_hi;
    int const mc = monotone_constraint_of(config, s.feature_id);
    if (mc == 0)
    {
        return;
    }
    double const wL  = bounded_leaf_weight(left.total_grad(), left.total_hess(), config,
                                           parent_lo, parent_hi);
    double const wR  = bounded_leaf_weight(right.total_grad(), right.total_hess(),
                                           config, parent_lo, parent_hi);
    double const mid = 0.5 * (wL + wR);
    if (mc > 0)
    {
        left.hi  = std::min(left.hi, mid);
        right.lo = std::max(right.lo, mid);
    }
    else
    {
        left.lo  = std::max(left.lo, mid);
        right.hi = std::min(right.hi, mid);
    }
}

using interaction_groups = std::vector<std::vector<feature_id_t>>;

// Parses TreeConfig::interaction_constraints ("0,1" or "0+1" per entry).
inline interaction_groups parse_interaction_groups(TreeConfig const &config)
{
    interaction_groups groups;
    for (auto const &entry : config.interaction_constraints)
    {
        std::vector<feature_id_t> group;
        size_t                    start = 0;
        while (start < entry.size())
        {
            size_t const sep = std::min(entry.find(',', start), entry.find('+', start));
            size_t const end = sep == std::string::npos ? entry.size() : sep;
            if (end > start)
            {
                group.push_back(static_cast<feature_id_t>(
                    std::stoul(entry.substr(start, end - start))));
            }
            start = end + 1;
        }
        if (!group.empty())
        {
            std::ranges::sort(group);
            groups.push_back(std::move(group));
        }
    }
    return groups;
}

// The features a node may split on, given the distinct features already on
// its path: every group that contains the whole path contributes its
// members, and a feature may always continue splitting alone. Empty result
// vector means "all allowed" (path empty or no constraints).
inline std::vector<char> allowed_features(interaction_groups const        &groups,
                                          std::vector<feature_id_t> const &path,
                                          size_t                           n_features)
{
    if (groups.empty() || path.empty())
    {
        return {};
    }
    std::vector<char> allowed(n_features, 0);
    for (auto const &group : groups)
    {
        bool const covers_path = std::ranges::includes(group, path);
        if (covers_path)
        {
            for (feature_id_t const f : group)
            {
                allowed[f] = 1;
            }
        }
    }
    if (path.size() == 1)
    {
        allowed[path.front()] = 1; // any feature may keep splitting alone
    }
    return allowed;
}

// Sets both children's interaction state after a split on `fid`.
inline void propagate_interaction_state(interaction_groups const        &groups,
                                        std::vector<feature_id_t> const &parent_path,
                                        feature_id_t fid, size_t n_features,
                                        SplitInput &left, SplitInput &right)
{
    if (groups.empty())
    {
        return;
    }
    std::vector<feature_id_t> path = parent_path;
    if (!std::ranges::binary_search(path, fid))
    {
        path.insert(std::ranges::upper_bound(path, fid), fid);
    }
    left.allowed  = allowed_features(groups, path, n_features);
    right.allowed = left.allowed;
    left.path     = path;
    right.path    = std::move(path);
}

// Per-tree feature subsample: a sorted draw of ceil(fraction * n) distinct
// feature ids. fraction >= 1 selects everything.
inline std::vector<feature_id_t> sample_features(size_t n_features, float fraction,
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

// Control plane, first half of a level: serial tree bookkeeping. Decides
// leaf-vs-split per frontier node, writes the tree's internal nodes and
// per-node stats, and defers the data-plane work (partition + histograms)
// into a LevelPlan the LevelStep executes level-wide.
inline LevelPlan
plan_level(Dataset const &ds, TreeConfig const &config,
           std::vector<SplitInput> &current, std::vector<SplitOutput> const &splits,
           std::vector<HistCell> const &child_sums, DenseTree::Nodes &nodes,
           size_t &n_leaves, train_leaf_values &values,
           std::vector<bin_id_t> &split_bins, std::vector<float> &split_gains,
           std::vector<float> &covers, std::vector<node_id_t> &leaf_ids)
{
    GrowProfiler::Lap lap;
    LevelPlan         plan;
    plan.splits.reserve(current.size());
    for (node_id_t i = 0; i < current.size(); ++i)
    {
        auto       &node  = current[i];
        auto const &split = splits[i];
        if (!split.valid) // assume valid incorporates all cfg parameter logic
        {
            plan.leaves.push_back({static_cast<uint32_t>(i), node.id});
            finalize_as_leaf(nodes, node, config, n_leaves, values, leaf_ids);
            continue;
        }
        node_id_t const left_id = nodes.size();
        nodes.emplace_back(DenseTree::leaf(0.0F));
        node_id_t const right_id = nodes.size();
        nodes.emplace_back(DenseTree::leaf(0.0F));
        split_bins.resize(nodes.size(), 0);
        split_bins[node.id] = split.bin_id;
        split_gains.resize(nodes.size(), 0.0F);
        split_gains[node.id] = static_cast<float>(split.gain);
        covers.resize(nodes.size(), 0.0F);

        float const threshold = ds.mappers()[split.feature_id].cuts()[split.bin_id];
        nodes[node.id] = DenseTree::internal(split.feature_id, threshold, left_id,
                                             right_id, split.default_left);

        double const   parent_lo   = node.lo;
        double const   parent_hi   = node.hi;
        auto           parent_path = std::move(node.path);
        HistCell const ls = child_sums.empty() ? HistCell{} : child_sums[2 * size_t{i}];
        HistCell const rs =
            child_sums.empty() ? HistCell{} : child_sums[(2 * size_t{i}) + 1];
        plan.splits.push_back({std::move(node), PendingSplit{}, split, left_id,
                               right_id, parent_lo, parent_hi, std::move(parent_path),
                               static_cast<uint32_t>(i), ls, rs});
    }
    lap(GrowProfiler::instance().bookkeep_s);
    return plan;
}

// Control plane, second half of a level: record covers, propagate the
// monotone bounds and interaction state both children need, and hand the
// children off as the next frontier in original node order.
inline void commit_children(Dataset const &ds, TreeConfig const &config,
                            interaction_groups const &groups, LevelPlan &plan,
                            std::vector<float>      &covers,
                            std::vector<SplitInput> &current,
                            std::vector<SplitInput> &next)
{
    GrowProfiler::Lap lap;
    for (auto &d : plan.splits)
    {
        SplitInput &left  = d.p.left;
        SplitInput &right = d.p.right;
        // Host children carry rows; device-resident children carry row_count.
        covers[d.left_id] =
            static_cast<float>(left.rows.empty() ? left.row_count : left.rows.size());
        covers[d.right_id] = static_cast<float>(right.rows.empty() ? right.row_count
                                                                   : right.rows.size());
        propagate_monotone_bounds(d.parent_lo, d.parent_hi, d.split, config, left,
                                  right);
        propagate_interaction_state(groups, d.parent_path, d.split.feature_id,
                                    ds.n_features(), left, right);
        next.push_back(std::move(left));
        next.push_back(std::move(right));
    }
    lap(GrowProfiler::instance().finalize_s);
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
                            row_index_view sampled, train_leaf_values &values,
                            std::vector<node_id_t> &leaf_ids)
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
                auto const &nd   = nodes[idx];
                auto const &bins = ds.feature_bins(nd.feature_id);
                auto const  last = static_cast<bin_id_t>(ds.n_bins(nd.feature_id) - 1);
                bin_id_t const b = bins[r];
                bool const left  = (b == last) ? nd.default_left : b <= split_bins[idx];
                idx              = left ? nd.left : nd.right;
            }
            values[r]   = nodes[idx].threshold_or_value;
            leaf_ids[r] = idx;
        });
}

} // namespace bonsai::grower_detail

namespace bonsai
{

template <HistogramEngine EngineT, NodeSplitFinder SplitterT>
DepthwiseGrower<EngineT, SplitterT>::DepthwiseGrower(TreeConfig const &cfg)
    : config_(cfg), feature_rng_(cfg.feature_seed),
      interaction_groups_(grower_detail::parse_interaction_groups(cfg))
{
}

template <HistogramEngine EngineT, NodeSplitFinder SplitterT>
auto DepthwiseGrower<EngineT, SplitterT>::grow(Dataset const &ds, floats_view grad,
                                               floats_view    hess,
                                               row_index_view row_indices)
    -> GrowResult<Tree>
{
    namespace gd = grower_detail;
    Tree::Nodes              nodes;
    train_leaf_values        values(ds.n_rows(), 0.0F);
    std::vector<node_id_t>   leaf_ids(ds.n_rows(), 0);
    std::vector<SplitInput>  current;
    std::vector<SplitInput>  next;
    std::vector<SplitOutput> splits;
    std::vector<HistCell>    child_sums;
    std::vector<bin_id_t>    split_bins(1, 0);
    std::vector<float>       split_gains(1, 0.0F);
    std::vector<float>       covers(1, static_cast<float>(row_indices.size()));
    auto const               selected =
        gd::sample_features(ds.n_features(), config_.feature_fraction, feature_rng_);

    // The LevelStep is the tree's data plane: host or GPU by engine type,
    // opened per tree so a GPU engine can decline back to the host mid-fit.
    gd::LevelStep<EngineT, SplitterT> step(engine_, ds, config_, grad, hess, selected);
    current.push_back(step.make_root(row_indices));
    nodes.emplace_back(DenseTree::leaf(0.0F));
    uint8_t depth    = 0;
    size_t  n_leaves = 0;
    while (depth < config_.max_depth)
    {
        step.find_splits(current, splits, child_sums);
        auto plan =
            gd::plan_level(ds, config_, current, splits, child_sums, nodes, n_leaves,
                           values, split_bins, split_gains, covers, leaf_ids);
        step.partition(plan);
        step.build_children(plan);
        gd::commit_children(ds, config_, interaction_groups_, plan, covers, current,
                            next);
        if (current.empty())
        {
            break;
        }
        ++depth;
    }
    step.finalize(current, nodes, n_leaves, values, leaf_ids, row_indices);
    gd::route_unsampled(ds, nodes, split_bins, row_indices, values, leaf_ids);
    split_gains.resize(nodes.size(), 0.0F);
    covers.resize(nodes.size(), 0.0F);

    return {.tree     = Tree(std::move(nodes), {.depth = depth, .n_leaves = n_leaves},
                             std::move(split_gains), std::move(covers)),
            .values   = std::move(values),
            .leaf_ids = std::move(leaf_ids)};
}

template <HistogramEngine EngineT, LevelSplitFinder SplitterT>
ObliviousGrower<EngineT, SplitterT>::ObliviousGrower(TreeConfig const &cfg)
    : config_(cfg), feature_rng_(cfg.feature_seed)
{
    for (int const mc : cfg.monotone_constraints)
    {
        if (mc != 0)
        {
            throw ConfigError(
                "monotone_constraints are not supported by the oblivious grower");
        }
    }
    if (!cfg.interaction_constraints.empty())
    {
        throw ConfigError(
            "interaction_constraints are not supported by the oblivious grower");
    }
}

template <HistogramEngine EngineT, LevelSplitFinder SplitterT>
auto ObliviousGrower<EngineT, SplitterT>::grow(Dataset const &ds, floats_view grad,
                                               floats_view    hess,
                                               row_index_view row_indices)
    -> GrowResult<Tree>
{
    namespace gd = grower_detail;
    Tree::LevelSplits level_splits;
    Tree::LeafTable   leaf_table;
    train_leaf_values values(ds.n_rows(), 0.0F);

    std::vector<SplitInput>  frontier;
    std::vector<SplitInput>  next;
    std::vector<SplitOutput> splits;
    std::vector<HistCell>    child_sums;
    std::vector<bin_id_t>    level_bins;
    std::vector<float>       level_gains;
    auto const               selected =
        gd::sample_features(ds.n_features(), config_.feature_fraction, feature_rng_);

    // Same data plane as depthwise; only the control plane differs (one split
    // per level, broadcast to every frontier node; ObliviousTree bookkeeping).
    gd::LevelStep<EngineT, SplitterT> step(engine_, ds, config_, grad, hess, selected);
    frontier.push_back(step.make_root(row_indices));

    size_t depth = 0;
    while (depth < config_.max_depth)
    {
        step.find_splits(frontier, splits, child_sums);
        SplitOutput const split = splits.front();
        if (!split.valid)
        {
            break;
        }
        float const threshold = ds.mappers()[split.feature_id].cuts()[split.bin_id];
        level_splits.push_back({.feature_id   = split.feature_id,
                                .threshold    = threshold,
                                .default_left = split.default_left});
        level_bins.push_back(split.bin_id);
        level_gains.push_back(static_cast<float>(split.gain));

        gd::LevelPlan plan;
        plan.splits.reserve(frontier.size());
        for (uint32_t i = 0; i < frontier.size(); ++i)
        {
            HistCell const ls = child_sums.empty() ? HistCell{} : child_sums[2 * i];
            HistCell const rs =
                child_sums.empty() ? HistCell{} : child_sums[(2 * i) + 1];
            plan.splits.push_back({std::move(frontier[i]),
                                   gd::PendingSplit{},
                                   split,
                                   0,
                                   0,
                                   0.0,
                                   0.0,
                                   {},
                                   i,
                                   ls,
                                   rs});
        }
        step.partition(plan);
        step.build_children(plan);
        next.reserve(plan.splits.size() * 2);
        for (auto &d : plan.splits)
        {
            next.push_back(std::move(d.p.left));
            next.push_back(std::move(d.p.right));
        }
        std::swap(frontier, next);
        next.clear();
        ++depth;
    }

    std::vector<node_id_t> leaf_ids(ds.n_rows(), 0);
    leaf_table.reserve(frontier.size());
    for (size_t li = 0; li < frontier.size(); ++li)
    {
        auto const &leaf = frontier[li];
        float const v = gd::leaf_value(leaf.total_grad(), leaf.total_hess(), config_);
        leaf_table.push_back(v);
        for (row_id_t r : leaf.rows) // device-empty on the GPU plane
        {
            values[r]   = v;
            leaf_ids[r] = static_cast<node_id_t>(li);
        }
    }
    // GPU plane: rows are device-resident, so stamp the final leaves and
    // download the per-row leaf assignment to fill values/leaf_ids.
    step.finalize_leaves(frontier, leaf_table, values, leaf_ids, row_indices);

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
                    auto const &s    = level_splits[lvl];
                    auto const &bins = ds.feature_bins(s.feature_id);
                    auto const  last =
                        static_cast<bin_id_t>(ds.n_bins(s.feature_id) - 1);
                    bin_id_t const b = bins[r];
                    bool const     left =
                        (b == last) ? s.default_left : b <= level_bins[lvl];
                    index = (index << 1U) | (left ? 0U : 1U);
                }
                values[r]   = leaf_table[index];
                leaf_ids[r] = static_cast<node_id_t>(index);
            });
    }

    return {.tree     = Tree(std::move(level_splits), std::move(leaf_table),
                             std::move(level_gains)),
            .values   = std::move(values),
            .leaf_ids = std::move(leaf_ids)};
}

template <HistogramEngine EngineT, NodeSplitFinder SplitterT>
LeafwiseGrower<EngineT, SplitterT>::LeafwiseGrower(TreeConfig const &cfg)
    : config_(cfg), feature_rng_(cfg.feature_seed),
      interaction_groups_(grower_detail::parse_interaction_groups(cfg))
{
}

template <HistogramEngine EngineT, NodeSplitFinder SplitterT>
auto LeafwiseGrower<EngineT, SplitterT>::grow(Dataset const &ds, floats_view grad,
                                              floats_view    hess,
                                              row_index_view row_indices)
    -> GrowResult<Tree>
{
    namespace gd = grower_detail;
    Tree::Nodes       nodes;
    train_leaf_values values(ds.n_rows(), 0.0F);

    // Max-heap on gain; ties broken by lower node id so growth is deterministic.
    auto gain_less = [](gd::Candidate const &a, gd::Candidate const &b)
    {
        if (a.split.gain != b.split.gain)
        {
            return a.split.gain < b.split.gain;
        }
        return a.node.id > b.node.id;
    };

    std::vector<gd::Candidate> heap;
    std::vector<SplitInput>    pending;
    std::vector<bin_id_t>      split_bins(1, 0);
    std::vector<float>         split_gains(1, 0.0F);
    std::vector<float>         covers(1, static_cast<float>(row_indices.size()));
    std::vector<node_id_t>     leaf_ids(ds.n_rows(), 0);

    auto const selected =
        gd::sample_features(ds.n_features(), config_.feature_fraction, feature_rng_);

    // Leafwise expands one node at a time (split_node) on host rows, so it does
    // not use the device-resident plane (which keeps rows/histograms on the GPU
    // and has no single-node best-first path). The root is built directly via
    // populate — the CUDA engine's populate delegates to its CPU member, so
    // this path is correct with either engine. begin_tree is skipped: it only
    // stages device gradients the CPU-side populate never reads.
    SplitInput root;
    root.id = 0;
    root.rows.assign(row_indices.begin(), row_indices.end());
    engine_.populate(ds, grad, hess, root, selected);
    root.sums      = root.totals();
    root.row_count = root.rows.size();
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
        gd::Candidate c = std::move(heap.back());
        heap.pop_back();

        node_id_t const left_id = nodes.size();
        nodes.emplace_back(DenseTree::leaf(0.0F));
        node_id_t const right_id = nodes.size();
        nodes.emplace_back(DenseTree::leaf(0.0F));
        split_bins.resize(nodes.size(), 0);
        split_bins[c.node.id] = c.split.bin_id;
        split_gains.resize(nodes.size(), 0.0F);
        split_gains[c.node.id] = static_cast<float>(c.split.gain);
        covers.resize(nodes.size(), 0.0F);

        float const threshold = ds.mappers()[c.split.feature_id].cuts()[c.split.bin_id];
        nodes[c.node.id] = DenseTree::internal(c.split.feature_id, threshold, left_id,
                                               right_id, c.split.default_left);

        double const parent_lo   = c.node.lo;
        double const parent_hi   = c.node.hi;
        auto const   parent_path = std::move(c.node.path);
        auto [left, right] = gd::split_node(ds, grad, hess, std::move(c.node), c.split,
                                            left_id, right_id, selected, engine_);
        covers[left_id]    = static_cast<float>(left.rows.size());
        covers[right_id]   = static_cast<float>(right.rows.size());
        gd::propagate_monotone_bounds(parent_lo, parent_hi, c.split, config_, left,
                                      right);
        gd::propagate_interaction_state(interaction_groups_, parent_path,
                                        c.split.feature_id, ds.n_features(), left,
                                        right);
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
        gd::finalize_as_leaf(nodes, c.node, config_, n_leaves, values, leaf_ids);
    }
    for (auto const &leaf : pending)
    {
        gd::finalize_as_leaf(nodes, leaf, config_, n_leaves, values, leaf_ids);
    }
    gd::route_unsampled(ds, nodes, split_bins, row_indices, values, leaf_ids);
    split_gains.resize(nodes.size(), 0.0F);
    covers.resize(nodes.size(), 0.0F);

    return {.tree     = Tree(std::move(nodes), {.depth = depth, .n_leaves = n_leaves},
                             std::move(split_gains), std::move(covers)),
            .values   = std::move(values),
            .leaf_ids = std::move(leaf_ids)};
}

} // namespace bonsai
