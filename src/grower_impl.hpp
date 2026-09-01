#pragma once

#include "bonsai/config/errors.hpp"
#include "bonsai/config/tree_config.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/monotone.hpp"
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
#include <limits>
#include <numeric>
#include <print>
#include <random>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace bonsai::grower_detail
{

inline float leaf_value(double grad, double hess, TreeConfig const &config)
{
    constexpr double inf = std::numeric_limits<double>::infinity();
    return static_cast<float>(
        bounded_leaf_weight(grad, hess, config.lambda_l1, config.lambda_l2, -inf, inf));
}

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

template <typename SplitInputT>
void project_monotone_leaves(std::vector<SplitInputT> const   &frontier,
                             TreeConfig const                 &config,
                             ObliviousTree::LevelSplits const &level_splits,
                             ObliviousTree::LeafTable         &leaf_table)
{
    if (!has_monotone_constraint(config.monotone_constraints))
    {
        return;
    }
    std::vector<float> hessians;
    hessians.reserve(frontier.size());
    for (auto const &leaf : frontier)
    {
        hessians.push_back(static_cast<float>(leaf.total_hess()));
    }
    project_monotone(monotone_levels(level_splits, config.monotone_constraints),
                     hessians, leaf_table);
}

using interaction_groups = std::vector<std::vector<feature_id_t>>;

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
        allowed[path.front()] = 1;
    }
    return allowed;
}

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
    return selected;
}

inline std::pair<node_id_t, node_id_t>
commit_split_node(DenseTree::Nodes &nodes, std::vector<bin_id_t> &split_bins,
                  std::vector<float> &split_gains, std::vector<float> &covers,
                  Dataset const &ds, node_id_t id, SplitOutput const &out)
{
    node_id_t const left_id = nodes.size();
    nodes.emplace_back(DenseTree::leaf(0.0F));
    node_id_t const right_id = nodes.size();
    nodes.emplace_back(DenseTree::leaf(0.0F));
    split_bins.resize(nodes.size(), 0);
    split_bins[id] = out.bin_id;
    split_gains.resize(nodes.size(), 0.0F);
    split_gains[id] = static_cast<float>(out.gain);
    covers.resize(nodes.size(), 0.0F);
    float const threshold = ds.cuts(out.feature_id)[out.bin_id];
    nodes[id] = DenseTree::internal(out.feature_id, threshold, left_id, right_id,
                                    out.default_left);
    return {left_id, right_id};
}

inline LevelPlan
plan_level(Dataset const &ds, TreeConfig const &config,
           std::vector<SplitInput> &current, std::vector<SplitOutput> const &splits,
           std::vector<HistCell> const &child_sums, DenseTree::Nodes &nodes,
           size_t &n_leaves, train_leaf_values &values,
           std::vector<bin_id_t> &split_bins, std::vector<float> &split_gains,
           std::vector<float> &covers, std::vector<node_id_t> &leaf_ids)
{
    Phase<&GrowProfiler::bookkeep_s> phase;
    LevelPlan                        plan;
    plan.splits.reserve(current.size());
    for (node_id_t i = 0; i < current.size(); ++i)
    {
        auto       &node  = current[i];
        auto const &split = splits[i];
        if (!split.valid)
        {
            plan.leaves.push_back({static_cast<uint32_t>(i), node.id});
            finalize_as_leaf(nodes, node, config, n_leaves, values, leaf_ids);
            continue;
        }
        auto const [left_id, right_id] = commit_split_node(
            nodes, split_bins, split_gains, covers, ds, node.id, split);

        double const   parent_lo   = node.lo;
        double const   parent_hi   = node.hi;
        auto           parent_path = std::move(node.path);
        HistCell const ls = child_sums.empty() ? HistCell{} : child_sums[2 * size_t{i}];
        HistCell const rs =
            child_sums.empty() ? HistCell{} : child_sums[(2 * size_t{i}) + 1];
        plan.splits.push_back({.parent      = std::move(node),
                               .p           = {},
                               .split       = split,
                               .left_id     = left_id,
                               .right_id    = right_id,
                               .parent_lo   = parent_lo,
                               .parent_hi   = parent_hi,
                               .parent_path = std::move(parent_path),
                               .parent_slot = static_cast<uint32_t>(i),
                               .left_sums   = ls,
                               .right_sums  = rs});
    }
    return plan;
}

inline void demote_empty_splits(TreeConfig const &config, LevelPlan &plan,
                                DenseTree::Nodes &nodes, size_t &n_leaves,
                                train_leaf_values      &values,
                                std::vector<node_id_t> &leaf_ids,
                                std::vector<float>     &split_gains)
{
    std::erase_if(plan.splits,
                  [&](DeferredSplit &d)
                  {
                      if (d.p.left.rows.empty() && d.p.right.rows.empty())
                      {
                          return false;
                      }
                      bool const left_empty  = d.p.left.rows.empty();
                      bool const right_empty = d.p.right.rows.empty();
                      if (!left_empty && !right_empty)
                      {
                          return false;
                      }
                      SplitInput &survivor = left_empty ? d.p.right : d.p.left;
                      d.parent.rows        = std::move(survivor.rows);
                      d.parent.hists       = std::move(d.p.parent_hists);
                      finalize_as_leaf(nodes, d.parent, config, n_leaves, values,
                                       leaf_ids);
                      split_gains[d.parent.id] = 0.0F;
                      return true;
                  });
}

inline void commit_children(Dataset const &ds, TreeConfig const &config,
                            interaction_groups const &groups, LevelPlan &plan,
                            std::vector<float>      &covers,
                            std::vector<SplitInput> &current,
                            std::vector<SplitInput> &next)
{
    for (auto &d : plan.splits)
    {
        SplitInput &left   = d.p.left;
        SplitInput &right  = d.p.right;
        covers[d.left_id]  = static_cast<float>(row_count_of(left));
        covers[d.right_id] = static_cast<float>(row_count_of(right));
        propagate_monotone_bounds(d.parent_lo, d.parent_hi, d.split, config, left,
                                  right);
        propagate_interaction_state(groups, d.parent_path, d.split.feature_id,
                                    ds.n_features(), left, right);
        next.push_back(std::move(left));
        next.push_back(std::move(right));
    }
    std::swap(current, next);
    next.clear();
}

struct PoppedSplit
{
    Candidate                 c;
    node_id_t                 left_id  = 0;
    node_id_t                 right_id = 0;
    double                    parent_lo;
    double                    parent_hi;
    std::vector<feature_id_t> parent_path;
};

template <typename LessT>
inline PoppedSplit
pop_split(std::vector<Candidate> &heap, LessT gain_less, DenseTree::Nodes &nodes,
          Dataset const &ds, std::vector<bin_id_t> &split_bins,
          std::vector<float> &split_gains, std::vector<float> &covers)
{
    Phase<&GrowProfiler::bookkeep_s> phase;
    std::pop_heap(heap.begin(), heap.end(), gain_less);
    Candidate c = std::move(heap.back());
    heap.pop_back();
    auto const [left_id, right_id] = commit_split_node(nodes, split_bins, split_gains,
                                                       covers, ds, c.node.id, c.split);
    double const parent_lo         = c.node.lo;
    double const parent_hi         = c.node.hi;
    auto         parent_path       = std::move(c.node.path);
    return {.c           = std::move(c),
            .left_id     = left_id,
            .right_id    = right_id,
            .parent_lo   = parent_lo,
            .parent_hi   = parent_hi,
            .parent_path = std::move(parent_path)};
}

// NOLINTNEXTLINE(readability-function-size)
template <typename StepT>
inline bool commit_pop(StepT &step, Dataset const &ds, TreeConfig const &config,
                       interaction_groups const &groups, PoppedSplit &ps,
                       ChildPair &pair, DenseTree::Nodes &nodes,
                       std::vector<float> &covers, std::vector<float> &split_gains,
                       size_t &n_leaves, size_t &live_leaves, uint8_t &depth,
                       train_leaf_values &values, std::vector<node_id_t> &leaf_ids)
{
    Phase<&GrowProfiler::commit_s> phase;
    SplitInput                    &left  = pair.nodes[0];
    SplitInput                    &right = pair.nodes[1];
    if (is_empty_child(left) || is_empty_child(right))
    {
        SplitInput const demoted =
            step.demoted_leaf(ps.c, pair, ps.parent_lo, ps.parent_hi);
        step.leaf(demoted.id, ps.c.slot);
        finalize_as_leaf(nodes, demoted, config, n_leaves, values, leaf_ids);
        split_gains[ps.c.node.id] = 0.0F;
        return false;
    }
    covers[ps.left_id]  = static_cast<float>(row_count_of(left));
    covers[ps.right_id] = static_cast<float>(row_count_of(right));
    propagate_monotone_bounds(ps.parent_lo, ps.parent_hi, ps.c.split, config, left,
                              right);
    propagate_interaction_state(groups, ps.parent_path, ps.c.split.feature_id,
                                ds.n_features(), left, right);
    ++live_leaves;
    depth = std::max(depth, pair.depth);
    return true;
}

template <typename LessT>
inline void queue_children(ChildPair &pair, LessT gain_less,
                           std::vector<Candidate> &heap,
                           std::vector<Candidate> &pending)
{
    Phase<&GrowProfiler::commit_s> phase;
    for (size_t i = 0; i < pair.nodes.size(); ++i)
    {
        Candidate child{.node       = std::move(pair.nodes[i]),
                        .split      = pair.splits[i],
                        .depth      = pair.depth,
                        .slot       = pair.slots[i],
                        .left_sums  = pair.child_sums[2 * i],
                        .right_sums = pair.child_sums[(2 * i) + 1]};
        if (child.split.valid)
        {
            heap.push_back(std::move(child));
            std::push_heap(heap.begin(), heap.end(), gain_less);
        }
        else
        {
            pending.push_back(std::move(child));
        }
    }
}

template <typename F>
void for_each_unsampled(RowView const &view, row_index_view sampled, F &&fn)
{
    if (sampled.size() == view.size())
    {
        return;
    }
    std::vector<row_id_t> oob;
    oob.reserve(view.size() - sampled.size());
    size_t j        = 0;
    auto   consider = [&](row_id_t r)
    {
        if (j < sampled.size() && sampled[j] == r)
        {
            ++j;
            return;
        }
        oob.push_back(r);
    };
    if (view.is_identity())
    {
        for (row_id_t r = 0; r < view.size(); ++r)
        {
            consider(r);
        }
    }
    else
    {
        for (row_id_t const r : view.materialize())
        {
            consider(r);
        }
    }
    parallel::for_each_index(oob.size(), [&](size_t k) { fn(oob[k]); });
}

inline void route_unsampled(Dataset const &ds, DenseTree::Nodes const &nodes,
                            std::vector<bin_id_t> const &split_bins,
                            row_index_view sampled, train_leaf_values &values,
                            std::vector<node_id_t> &leaf_ids)
{
    for_each_unsampled(ds.row_view(), sampled,
                       [&](row_id_t r)
                       {
                           node_id_t idx = 0;
                           while (!DenseTree::is_leaf(nodes[idx]))
                           {
                               auto const &nd = nodes[idx];
                               auto const  last =
                                   static_cast<bin_id_t>(ds.n_bins(nd.feature_id) - 1);
                               bin_id_t const b = ds.bin_at(nd.feature_id, r);
                               bool const left  = routes_left(b, last, split_bins[idx],
                                                              nd.default_left);
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
    : Host(cfg), interaction_groups_(grower_detail::parse_interaction_groups(cfg))
{
}

template <HistogramEngine EngineT, NodeSplitFinder SplitterT>
auto DepthwiseGrower<EngineT, SplitterT>::grow(Dataset const &ds, floats_view grad,
                                               floats_view hess, RowSelection selection)
    -> GrowResult<Tree>
{
    namespace gd                    = grower_detail;
    bool const             resident = this->resident();
    gd::GrowProfiler::Lap  slap;
    Tree::Nodes            nodes;
    train_leaf_values      values   = std::move(recycled().values);
    std::vector<node_id_t> leaf_ids = std::move(recycled().leaf_ids);
    if (!resident)
    {
        values.resize(ds.plane_n_rows(), 0.0F);
        leaf_ids.resize(ds.plane_n_rows(), 0);
    } // namespace gd
    std::vector<SplitInput> current;
    std::vector<SplitInput> next;
    gd::LevelOutputs        level_out;
    std::vector<bin_id_t>   split_bins(1, 0);
    std::vector<float>      split_gains(1, 0.0F);
    std::vector<float>      covers(1, static_cast<float>(selection.rows.size()));
    auto const              selected =
        gd::sample_features(ds.n_features(), config().feature_fraction, feature_rng());

    gd::LevelStep<EngineT, SplitterT> step(seam().engine(), ds, config(), grad, hess,
                                           selected);
    slap(gd::GrowProfiler::instance().setup_s);
    current.push_back(step.make_root(selection));
    nodes.emplace_back(DenseTree::leaf(0.0F));
    uint8_t depth    = 0;
    size_t  n_leaves = 0;
    while (depth < config().max_depth)
    {
        step.open_level(current, level_out);
        auto plan = gd::plan_level(ds, config(), current, level_out.splits,
                                   level_out.child_sums, nodes, n_leaves, values,
                                   split_bins, split_gains, covers, leaf_ids);
        step.apply_level(plan);
        gd::GrowProfiler::Lap clap;
        gd::demote_empty_splits(config(), plan, nodes, n_leaves, values, leaf_ids,
                                split_gains);
        clap(gd::GrowProfiler::instance().commit_s);
        step.build_children(plan, depth + 1 >= config().max_depth);
        gd::GrowProfiler::Lap clap2;
        gd::commit_children(ds, config(), interaction_groups_, plan, covers, current,
                            next);
        clap2(gd::GrowProfiler::instance().commit_s);
        if (current.empty())
        {
            break;
        }
        ++depth;
    }
    {
        gd::Phase<&gd::GrowProfiler::finalize_s> phase;
        step.end_tree(current, nodes, n_leaves, values, leaf_ids, selection.rows);
        if (!resident)
        {
            gd::route_unsampled(ds, nodes, split_bins, selection.rows, values,
                                leaf_ids);
        }
    }
    gd::GrowProfiler::Lap alap;
    split_gains.resize(nodes.size(), 0.0F);
    covers.resize(nodes.size(), 0.0F);
    alap(gd::GrowProfiler::instance().assemble_s);

    return {.tree     = Tree(std::move(nodes), {.depth = depth, .n_leaves = n_leaves},
                             std::move(split_gains), std::move(covers)),
            .values   = std::move(values),
            .leaf_ids = std::move(leaf_ids)};
}

template <HistogramEngine EngineT, LevelSplitFinder SplitterT>
ObliviousGrower<EngineT, SplitterT>::ObliviousGrower(TreeConfig const &cfg) : Host(cfg)
{
    if (!cfg.interaction_constraints.empty())
    {
        throw ConfigError(
            "interaction_constraints are not supported by the levelwise grower");
    }
}

template <HistogramEngine EngineT, LevelSplitFinder SplitterT>
auto ObliviousGrower<EngineT, SplitterT>::grow(Dataset const &ds, floats_view grad,
                                               floats_view hess, RowSelection selection)
    -> GrowResult<Tree>
{
    namespace gd                   = grower_detail;
    bool const            resident = this->resident();
    gd::GrowProfiler::Lap slap;
    Tree::LevelSplits     level_splits;
    Tree::LeafTable       leaf_table;
    train_leaf_values     values = std::move(recycled().values);
    if (!resident)
    {
        values.resize(ds.plane_n_rows(), 0.0F);
    } // namespace gd

    std::vector<SplitInput> frontier;
    std::vector<SplitInput> next;
    gd::LevelOutputs        level_out;
    std::vector<bin_id_t>   level_bins;
    std::vector<float>      level_gains;
    auto const              selected =
        gd::sample_features(ds.n_features(), config().feature_fraction, feature_rng());

    gd::LevelStep<EngineT, SplitterT> step(seam().engine(), ds, config(), grad, hess,
                                           selected);
    slap(gd::GrowProfiler::instance().setup_s);
    frontier.push_back(step.make_root(selection));

    size_t depth = 0;
    while (depth < config().max_depth)
    {
        step.open_level(frontier, level_out);
        SplitOutput const split = level_out.splits.front();
        if (!split.valid)
        {
            break;
        }
        float const threshold = ds.cuts(split.feature_id)[split.bin_id];
        level_splits.push_back({.feature_id   = split.feature_id,
                                .threshold    = threshold,
                                .default_left = split.default_left});
        level_bins.push_back(split.bin_id);
        level_gains.push_back(static_cast<float>(split.gain));

        gd::LevelPlan plan;
        plan.splits.reserve(frontier.size());
        for (uint32_t i = 0; i < frontier.size(); ++i)
        {
            HistCell const ls =
                level_out.child_sums.empty() ? HistCell{} : level_out.child_sums[2 * i];
            HistCell const rs = level_out.child_sums.empty()
                                    ? HistCell{}
                                    : level_out.child_sums[(2 * i) + 1];
            plan.splits.push_back({.parent      = std::move(frontier[i]),
                                   .p           = {},
                                   .split       = split,
                                   .left_id     = 0,
                                   .right_id    = 0,
                                   .parent_lo   = 0.0,
                                   .parent_hi   = 0.0,
                                   .parent_path = {},
                                   .parent_slot = i,
                                   .left_sums   = ls,
                                   .right_sums  = rs});
        }
        step.apply_level(plan);
        step.build_children(plan, depth + 1 >= config().max_depth);
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

    std::vector<node_id_t> leaf_ids = std::move(recycled().leaf_ids);
    if (!resident)
    {
        leaf_ids.resize(ds.plane_n_rows(), 0);
    }
    leaf_table.reserve(frontier.size());
    std::vector<float> leaf_covers;
    leaf_covers.reserve(frontier.size());
    for (size_t li = 0; li < frontier.size(); ++li)
    {
        auto const &leaf = frontier[li];
        float const v = gd::leaf_value(leaf.total_grad(), leaf.total_hess(), config());
        leaf_table.push_back(v);
        leaf_covers.push_back(static_cast<float>(gd::row_count_of(leaf)));
    }
    gd::project_monotone_leaves(frontier, config(), level_splits, leaf_table);
    // perf: Host plane stamps each leaf's rows; device plane stamps the resident
    // segments and downloads the per-row assignment. Lapped as finalize:
    // a 16M CPU decomposition found ~15s of stamping hiding in levelwise's
    // conservation gap because only depthwise lapped it.
    gd::GrowProfiler::Lap flap;
    step.finalize_leaves(frontier, leaf_table, values, leaf_ids, selection.rows,
                         std::span<ObliviousTree::LevelSplit const>{level_splits},
                         std::span<bin_id_t const>{level_bins});

    if (!resident)
    {
        gd::for_each_unsampled(
            ds.row_view(), selection.rows,
            [&](row_id_t r)
            {
                size_t index = 0;
                for (size_t lvl = 0; lvl < level_splits.size(); ++lvl)
                {
                    auto const &s = level_splits[lvl];
                    auto const  last =
                        static_cast<bin_id_t>(ds.n_bins(s.feature_id) - 1);
                    bin_id_t const b = ds.bin_at(s.feature_id, r);
                    bool const     left =
                        routes_left(b, last, level_bins[lvl], s.default_left);
                    index = (index << 1U) | (left ? 0U : 1U);
                }
                values[r]   = leaf_table[index];
                leaf_ids[r] = static_cast<node_id_t>(index);
            });
    }
    flap(gd::GrowProfiler::instance().finalize_s);

    return {.tree     = Tree(std::move(level_splits), std::move(leaf_table),
                             std::move(level_gains), std::move(leaf_covers)),
            .values   = std::move(values),
            .leaf_ids = std::move(leaf_ids)};
}

template <HistogramEngine EngineT, ParallelNodeSplitFinder SplitterT>
LeafwiseGrower<EngineT, SplitterT>::LeafwiseGrower(TreeConfig const &cfg)
    : Host(cfg), interaction_groups_(grower_detail::parse_interaction_groups(cfg))
{
}

template <HistogramEngine EngineT, ParallelNodeSplitFinder SplitterT>
auto LeafwiseGrower<EngineT, SplitterT>::grow(Dataset const &ds, floats_view grad,
                                              floats_view hess, RowSelection selection)
    -> GrowResult<Tree>
{
    namespace gd                   = grower_detail;
    bool const            resident = this->resident();
    gd::GrowProfiler::Lap slap;
    Tree::Nodes           nodes;
    train_leaf_values     values = std::move(recycled().values);
    if (!resident)
    {
        values.resize(ds.plane_n_rows(), 0.0F);
    } // namespace gd

    auto gain_less = [](gd::Candidate const &a, gd::Candidate const &b)
    {
        if (a.split.gain != b.split.gain)
        {
            return a.split.gain < b.split.gain;
        }
        return a.node.id > b.node.id;
    };

    std::vector<gd::Candidate> heap;
    std::vector<gd::Candidate> pending;
    std::vector<bin_id_t>      split_bins(1, 0);
    std::vector<float>         split_gains(1, 0.0F);
    std::vector<float>         covers(1, static_cast<float>(selection.rows.size()));
    std::vector<node_id_t>     leaf_ids = std::move(recycled().leaf_ids);
    if (!resident)
    {
        leaf_ids.resize(ds.plane_n_rows(), 0);
    }

    auto const selected =
        gd::sample_features(ds.n_features(), config().feature_fraction, feature_rng());

    gd::LeafStep<EngineT, SplitterT> step(seam().engine(), ds, config(), grad, hess,
                                          selected);
    slap(gd::GrowProfiler::instance().setup_s);
    gd::Candidate root = step.open_root(selection);
    nodes.emplace_back(DenseTree::leaf(0.0F));

    size_t  n_leaves    = 0;
    size_t  live_leaves = 1;
    uint8_t depth       = 0;

    auto has_budget = [&]
    { return config().max_leaves == 0 || live_leaves < config().max_leaves; };

    if (root.split.valid && config().max_depth > 0)
    {
        heap.push_back(std::move(root));
    }
    else
    {
        pending.push_back(std::move(root));
    }

    while (!heap.empty() && has_budget())
    {
        gd::PoppedSplit ps =
            gd::pop_split(heap, gain_less, nodes, ds, split_bins, split_gains, covers);
        gd::ChildPair pair = step.split_children(ps.c, ps.left_id, ps.right_id);
        if (!gd::commit_pop(step, ds, config(), interaction_groups_, ps, pair, nodes,
                            covers, split_gains, n_leaves, live_leaves, depth, values,
                            leaf_ids))
        {
            continue;
        }
        step.find_children(pair, pair.depth < config().max_depth);
        gd::queue_children(pair, gain_less, heap, pending);
    }

    {
        gd::Phase<&gd::GrowProfiler::finalize_s> phase;
        heap.insert(heap.end(), std::make_move_iterator(pending.begin()),
                    std::make_move_iterator(pending.end()));
        for (auto const &c : heap)
        {
            step.leaf(c.node.id, c.slot);
            gd::write_leaf(nodes, c.node, config(), n_leaves);
        }
        gd::stamp_leaf_rows(nodes, heap | std::views::transform(&gd::Candidate::node),
                            values, leaf_ids);
        step.end_tree(nodes, values, leaf_ids);
        if (!resident)
        {
            gd::route_unsampled(ds, nodes, split_bins, selection.rows, values,
                                leaf_ids);
        }
    }
    gd::GrowProfiler::Lap alap;
    split_gains.resize(nodes.size(), 0.0F);
    covers.resize(nodes.size(), 0.0F);
    alap(gd::GrowProfiler::instance().assemble_s);

    return {.tree     = Tree(std::move(nodes), {.depth = depth, .n_leaves = n_leaves},
                             std::move(split_gains), std::move(covers)),
            .values   = std::move(values),
            .leaf_ids = std::move(leaf_ids)};
}

template <typename EngineT, typename TableFn>
bool eval_walk(EngineT &engine, bool armed, TableFn &&table, float lr,
               std::span<float> scores_out, std::optional<float> &loss)
{
    if constexpr (requires { typename EngineT::ResidentNode; })
    {
        if (!armed)
        {
            return false;
        }
        loss = engine.eval_accumulate(
            table.template operator()<typename EngineT::ResidentNode>(), lr,
            scores_out);
        return true;
    }
    else
    {
        return false;
    }
}

template <HistogramEngine EngineT, NodeSplitFinder SplitterT>
bool DepthwiseGrower<EngineT, SplitterT>::eval_accumulate(Tree const      &tree,
                                                          Dataset const   &valid,
                                                          float            lr,
                                                          std::span<float> scores_out,
                                                          std::optional<float> &loss)
{
    return eval_walk(
        seam().engine(), seam().eval_armed(), [&]<typename NodeT>()
        { return grower_detail::resident_node_table<NodeT>(tree.nodes(), valid); }, lr,
        scores_out, loss);
}

template <HistogramEngine EngineT, LevelSplitFinder SplitterT>
bool ObliviousGrower<EngineT, SplitterT>::eval_accumulate(Tree const      &tree,
                                                          Dataset const   &valid,
                                                          float            lr,
                                                          std::span<float> scores_out,
                                                          std::optional<float> &loss)
{
    return eval_walk(
        seam().engine(), seam().eval_armed(), [&]<typename NodeT>()
        { return grower_detail::oblivious_node_table<NodeT>(tree, valid); }, lr,
        scores_out, loss);
}

template <HistogramEngine EngineT, ParallelNodeSplitFinder SplitterT>
bool LeafwiseGrower<EngineT, SplitterT>::eval_accumulate(Tree const    &tree,
                                                         Dataset const &valid, float lr,
                                                         std::span<float> scores_out,
                                                         std::optional<float> &loss)
{
    return eval_walk(
        seam().engine(), seam().eval_armed(), [&]<typename NodeT>()
        { return grower_detail::resident_node_table<NodeT>(tree.nodes(), valid); }, lr,
        scores_out, loss);
}

} // namespace bonsai
