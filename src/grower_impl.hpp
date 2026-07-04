#pragma once

// Template definitions for the three growers, shared by every explicit
// instantiation TU: grower.cpp instantiates the CPU-builder variants and
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

// Host-side counterpart of the CUDA builder's ProfileCounters: attributes a
// fit's wall-clock across the grow loop's phases (BONSAI_GROW_PROFILE=1),
// printed once at process exit. Depthwise-only; drives phase-3 staging.
struct GrowProfiler
{
    bool const enabled = std::getenv("BONSAI_GROW_PROFILE") != nullptr;
    double find_s = 0, bookkeep_s = 0, partition_s = 0, populate_s = 0, finalize_s = 0;

    static GrowProfiler &instance()
    {
        static GrowProfiler prof;
        return prof;
    }

    ~GrowProfiler()
    {
        if (enabled &&
            find_s + bookkeep_s + partition_s + populate_s + finalize_s > 0.0)
        {
            std::println(stderr,
                         "grow-profile: find={:.2f}s bookkeep={:.2f}s "
                         "partition={:.2f}s populate={:.2f}s finalize={:.2f}s",
                         find_s, bookkeep_s, partition_s, populate_s, finalize_s);
        }
    }
};

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

using feature_view = std::span<feature_id_t const>;

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

inline void finalize_as_leaf(DenseTree::Nodes &nodes, SplitInput const &node,
                             TreeConfig const &config, size_t &n_leaves,
                             train_leaf_values      &values,
                             std::vector<node_id_t> &leaf_ids)
{
    auto const v   = static_cast<float>(bounded_leaf_weight(
        node.total_grad(), node.total_hess(), config, node.lo, node.hi));
    nodes[node.id] = DenseTree::leaf(v);
    for (row_id_t r : node.rows)
    {
        values[r]   = v;
        leaf_ids[r] = node.id;
    }
    ++n_leaves;
}

template <HistogramBuilder BuilderT>
SplitInput make_root(Dataset const &ds, floats_view grad, floats_view hess,
                     row_index_view row_indices, feature_view selected,
                     BuilderT &builder)
{
    SplitInput root;
    root.id = 0;
    root.rows.assign(row_indices.begin(), row_indices.end());
    builder.populate(ds, grad, hess, root, selected);
    root.sums      = root.totals();
    root.row_count = root.rows.size();
    return root;
}

// Uses the builder's batched hook when present, else a populate loop.
template <HistogramBuilder BuilderT>
inline void populate_nodes(Dataset const &ds, floats_view grad, floats_view hess,
                           split_input_refs nodes, feature_view selected,
                           BuilderT &builder)
{
    if constexpr (requires { builder.populate_many(ds, grad, hess, nodes, selected); })
    {
        builder.populate_many(ds, grad, hess, nodes, selected);
    }
    else
    {
        for (SplitInput &node : nodes)
        {
            builder.populate(ds, grad, hess, node, selected);
        }
    }
}

// One level's split search. Host path: the splitter policy per node. A
// builder exposing find_splits_many (device-resident histograms, phase 3)
// takes over here; the splitter remains the fallback and parity reference.
template <NodeSplitFinder SplitterT, HistogramBuilder BuilderT>
inline void
find_splits(std::vector<SplitInput> const &current, TreeConfig const &config,
            [[maybe_unused]] feature_view selected, [[maybe_unused]] BuilderT &builder,
            std::vector<SplitOutput> &out)
{
    out.clear();
    out.reserve(current.size());
    auto const t0 = std::chrono::steady_clock::now();
    for (auto const &input : current)
    {
        out.push_back(SplitterT::find(input, config));
    }
    GrowProfiler::instance().find_s +=
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// A split with rows partitioned and histograms pending: the smaller child
// populates, then finish_split derives the larger by subtraction.
struct PendingSplit
{
    SplitInput             left;
    SplitInput             right;
    std::vector<Histogram> parent_hists;
};

inline SplitInput &smaller_child(PendingSplit &p)
{
    return p.left.rows.size() <= p.right.rows.size() ? p.left : p.right;
}

// Scatters parent.rows into the children in one stable pass. Stability
// keeps every node's rows ascending (the root's are iota), so later
// per-feature bin lookups walk memory near-sequentially.
inline PendingSplit partition_rows(Dataset const &ds, SplitInput parent,
                                   SplitOutput const &s, node_id_t left_id,
                                   node_id_t right_id)
{
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

    PendingSplit p;
    p.left.id     = left_id;
    p.right.id    = right_id;
    size_t n_left = 0;
    for (row_id_t const r : parent.rows)
    {
        n_left += goes_left(r) ? 1 : 0;
    }
    p.left.rows.resize(n_left);
    p.right.rows.resize(parent.rows.size() - n_left);
    size_t li = 0;
    size_t ri = 0;
    for (row_id_t const r : parent.rows)
    {
        if (goes_left(r))
        {
            p.left.rows[li++] = r;
        }
        else
        {
            p.right.rows[ri++] = r;
        }
    }
    p.parent_hists = std::move(parent.hists);
    return p;
}

// Completes a partitioned split whose smaller child has been populated: the
// larger child takes the parent's histograms and subtracts the sibling.
inline void finish_split(Dataset const &ds, PendingSplit &p)
{
    bool const  left_smaller = p.left.rows.size() <= p.right.rows.size();
    SplitInput &small        = left_smaller ? p.left : p.right;
    SplitInput &large        = left_smaller ? p.right : p.left;
    large.hists.reserve(ds.n_features());
    for (feature_id_t f = 0; f < ds.n_features(); ++f)
    {
        large.hists.push_back(std::move(p.parent_hists[f]));
    }
    // Unselected slots are zero-binned on both sides: no-op subtraction.
    parallel::for_each_index(ds.n_features(),
                             [&](size_t f) { large.hists[f] -= small.hists[f]; });
    small.sums      = small.totals(); // row_count still 0: totals() scans hists
    large.sums      = large.totals();
    small.row_count = small.rows.size();
    large.row_count = large.rows.size();
}

template <HistogramBuilder BuilderT>
inline std::pair<SplitInput, SplitInput>
split_node(Dataset const &ds, floats_view grad, floats_view hess, SplitInput parent,
           SplitOutput const &s, node_id_t left_id, node_id_t right_id,
           feature_view selected, BuilderT &builder)
{
    PendingSplit p = partition_rows(ds, std::move(parent), s, left_id, right_id);
    builder.populate(ds, grad, hess, smaller_child(p), selected);
    finish_split(ds, p);
    return {std::move(p.left), std::move(p.right)};
}

template <HistogramBuilder BuilderT>
inline void
update_nodes(Dataset const &ds, floats_view grad, floats_view hess,
             TreeConfig const &config, std::vector<SplitInput> &current,
             std::vector<SplitInput> &next, std::vector<SplitOutput> const &splits,
             DenseTree::Nodes &nodes, size_t &n_leaves, train_leaf_values &values,
             feature_view selected, std::vector<bin_id_t> &split_bins,
             std::vector<float> &split_gains, std::vector<float> &covers,
             std::vector<node_id_t> &leaf_ids, interaction_groups const &groups,
             BuilderT &builder)
{
    // Pass 1: serial tree bookkeeping; partitions and histogram work are
    // deferred so both can run level-wide.
    struct Deferred
    {
        SplitInput                parent;
        PendingSplit              p;
        SplitOutput               split;
        node_id_t                 left_id;
        node_id_t                 right_id;
        double                    parent_lo;
        double                    parent_hi;
        std::vector<feature_id_t> parent_path;
    };
    auto &prof = GrowProfiler::instance();
    auto  mark = std::chrono::steady_clock::now();
    auto  lap  = [&mark](double &sink)
    {
        auto const now = std::chrono::steady_clock::now();
        sink += std::chrono::duration<double>(now - mark).count();
        mark = now;
    };
    std::vector<Deferred> deferred;
    deferred.reserve(current.size());
    for (node_id_t i = 0; i < current.size(); ++i)
    {
        auto       &node  = current[i];
        auto const &split = splits[i];
        if (!split.valid) // assume valid incorporates all cfg parameter logic
        {
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

        double const parent_lo   = node.lo;
        double const parent_hi   = node.hi;
        auto         parent_path = std::move(node.path);
        deferred.push_back({std::move(node), PendingSplit{}, split, left_id, right_id,
                            parent_lo, parent_hi, std::move(parent_path)});
    }

    lap(prof.bookkeep_s);

    // Pass 1b: partition each parent's rows, one node per worker (child
    // row order is scheduling-independent, so bit-identical to serial).
    parallel::for_each_index(deferred.size(),
                             [&](size_t i)
                             {
                                 Deferred &d = deferred[i];
                                 d.p = partition_rows(ds, std::move(d.parent), d.split,
                                                      d.left_id, d.right_id);
                             });
    for (auto &d : deferred)
    {
        covers[d.left_id]  = static_cast<float>(d.p.left.rows.size());
        covers[d.right_id] = static_cast<float>(d.p.right.rows.size());
    }

    lap(prof.partition_s);

    // Pass 2: populate every smaller sibling in one builder call.
    std::vector<std::reference_wrapper<SplitInput>> smalls;
    smalls.reserve(deferred.size());
    for (auto &d : deferred)
    {
        smalls.emplace_back(smaller_child(d.p));
    }
    populate_nodes(ds, grad, hess, smalls, selected, builder);
    lap(prof.populate_s);

    // Pass 3: subtraction, then constraint propagation (needs populated
    // child totals) and frontier hand-off in original node order.
    for (auto &d : deferred)
    {
        finish_split(ds, d.p);
        SplitInput &left  = d.p.left;
        SplitInput &right = d.p.right;
        propagate_monotone_bounds(d.parent_lo, d.parent_hi, d.split, config, left,
                                  right);
        propagate_interaction_state(groups, d.parent_path, d.split.feature_id,
                                    ds.n_features(), left, right);
        next.push_back(std::move(left));
        next.push_back(std::move(right));
    }
    lap(prof.finalize_s);
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

template <NodeSplitFinder SplitterT, HistogramBuilder BuilderT>
DepthwiseGrower<SplitterT, BuilderT>::DepthwiseGrower(TreeConfig const &cfg)
    : config_(cfg), feature_rng_(cfg.feature_seed),
      interaction_groups_(grower_detail::parse_interaction_groups(cfg))
{
}

template <NodeSplitFinder SplitterT, HistogramBuilder BuilderT>
auto DepthwiseGrower<SplitterT, BuilderT>::grow(Dataset const &ds, floats_view grad,
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
    std::vector<bin_id_t>    split_bins(1, 0);
    std::vector<float>       split_gains(1, 0.0F);
    std::vector<float>       covers(1, static_cast<float>(row_indices.size()));
    auto const               selected =
        gd::sample_features(ds.n_features(), config_.feature_fraction, feature_rng_);
    builder_.begin_tree(ds, grad, hess);
    current.push_back(gd::make_root(ds, grad, hess, row_indices, selected, builder_));
    nodes.emplace_back(DenseTree::leaf(0.0F));
    uint8_t depth    = 0;
    size_t  n_leaves = 0;
    while (depth < config_.max_depth)
    {
        gd::find_splits<SplitterT>(current, config_, selected, builder_, splits);
        gd::update_nodes(ds, grad, hess, config_, current, next, splits, nodes,
                         n_leaves, values, selected, split_bins, split_gains, covers,
                         leaf_ids, interaction_groups_, builder_);
        if (current.empty())
        {
            break;
        }
        ++depth;
    }

    for (auto const &node : current)
    {
        gd::finalize_as_leaf(nodes, node, config_, n_leaves, values, leaf_ids);
    }
    gd::route_unsampled(ds, nodes, split_bins, row_indices, values, leaf_ids);
    split_gains.resize(nodes.size(), 0.0F);
    covers.resize(nodes.size(), 0.0F);

    return {.tree     = Tree(std::move(nodes), {.depth = depth, .n_leaves = n_leaves},
                             std::move(split_gains), std::move(covers)),
            .values   = std::move(values),
            .leaf_ids = std::move(leaf_ids)};
}

template <LevelSplitFinder SplitterT, HistogramBuilder BuilderT>
ObliviousGrower<SplitterT, BuilderT>::ObliviousGrower(TreeConfig const &cfg)
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

template <LevelSplitFinder SplitterT, HistogramBuilder BuilderT>
auto ObliviousGrower<SplitterT, BuilderT>::grow(Dataset const &ds, floats_view grad,
                                                floats_view    hess,
                                                row_index_view row_indices)
    -> GrowResult<Tree>
{
    namespace gd = grower_detail;
    Tree::LevelSplits level_splits;
    Tree::LeafTable   leaf_table;
    train_leaf_values values(ds.n_rows(), 0.0F);

    std::vector<SplitInput> frontier;
    std::vector<SplitInput> next;
    std::vector<bin_id_t>   level_bins;
    std::vector<float>      level_gains;
    auto const              selected =
        gd::sample_features(ds.n_features(), config_.feature_fraction, feature_rng_);
    builder_.begin_tree(ds, grad, hess);
    frontier.push_back(gd::make_root(ds, grad, hess, row_indices, selected, builder_));

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
        level_gains.push_back(static_cast<float>(split.gain));
        // Same shape as the depthwise update: partition, batch-populate,
        // subtract.
        std::vector<gd::PendingSplit> pending(frontier.size());
        parallel::for_each_index(frontier.size(),
                                 [&](size_t i)
                                 {
                                     pending[i] = gd::partition_rows(
                                         ds, std::move(frontier[i]), split, 0, 0);
                                 });
        std::vector<std::reference_wrapper<SplitInput>> smalls;
        smalls.reserve(pending.size());
        for (auto &p : pending)
        {
            smalls.emplace_back(gd::smaller_child(p));
        }
        gd::populate_nodes(ds, grad, hess, smalls, selected, builder_);
        next.reserve(pending.size() * 2);
        for (auto &p : pending)
        {
            gd::finish_split(ds, p);
            next.push_back(std::move(p.left));
            next.push_back(std::move(p.right));
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
        for (row_id_t r : leaf.rows)
        {
            values[r]   = v;
            leaf_ids[r] = static_cast<node_id_t>(li);
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

template <NodeSplitFinder SplitterT, HistogramBuilder BuilderT>
LeafwiseGrower<SplitterT, BuilderT>::LeafwiseGrower(TreeConfig const &cfg)
    : config_(cfg), feature_rng_(cfg.feature_seed),
      interaction_groups_(grower_detail::parse_interaction_groups(cfg))
{
}

template <NodeSplitFinder SplitterT, HistogramBuilder BuilderT>
auto LeafwiseGrower<SplitterT, BuilderT>::grow(Dataset const &ds, floats_view grad,
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
    builder_.begin_tree(ds, grad, hess);
    SplitInput root = gd::make_root(ds, grad, hess, row_indices, selected, builder_);
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
                                            left_id, right_id, selected, builder_);
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
