#pragma once

#include "bonsai/config/tree_config.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/detail/perf.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/parallel.hpp"
#include "bonsai/split.hpp"
#include "bonsai/tree.hpp"
#include "bonsai/types.hpp"
#include "step/primitives.hpp"
#include "step/tree.hpp"
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace bonsai::grower_detail
{

template <HistogramEngine EngineT, typename SplitterT>
class LevelStep : public TreeStep<EngineT>
{
    using Step = TreeStep<EngineT>;
    using Step::config_;
    using Step::ds_;
    using Step::engine_;
    using Step::grad_;
    using Step::hess_;
    using Step::selected_;

  public:
    using Step::Step;

    SplitInput make_root(RowSelection const &sel)
    {
        Phase<&GrowProfiler::populate_s> phase;
        return populated_root(ds_, grad_, hess_, sel, selected_, engine_);
    }

    void open_level(std::vector<SplitInput> const &frontier, LevelOutputs &out)
    {
        Phase<&GrowProfiler::find_s> phase;
        out.splits.clear();
        out.child_sums.clear();
        if constexpr (LevelSplitFinder<SplitterT>)
        {
            SplitOutput const split = SplitterT::find(frontier, config_);
            out.splits.assign(frontier.size(), split);
        }
        else
        {
            out.splits.resize(frontier.size());
            SplitInput const *cur = frontier.data();
            SplitOutput      *op  = out.splits.data();
            parallel::for_each_index(frontier.size(), [&, cur, op](size_t i)
                                     { op[i] = SplitterT::find(cur[i], config_); });
        }
    }

    void apply_level(LevelPlan &plan)
    {
        Phase<&GrowProfiler::partition_s> phase;
        host_partition(ds_, plan);
    }

    void build_children(LevelPlan &plan, bool /*last*/ = false)
    {
        Phase<&GrowProfiler::populate_s>                phase;
        std::vector<std::reference_wrapper<SplitInput>> smalls;
        smalls.reserve(plan.splits.size());
        for (auto &d : plan.splits)
        {
            smalls.emplace_back(smaller_child(d.p));
        }
        populate_nodes(ds_, grad_, hess_, smalls, selected_, engine_);
        for (auto &d : plan.splits)
        {
            finish_split(ds_, d.p);
        }
    }

    void end_tree(std::vector<SplitInput> const &current, DenseTree::Nodes &nodes,
                  size_t &n_leaves, train_leaf_values &values,
                  std::vector<node_id_t> &leaf_ids, row_index_view /*row_indices*/)
    {
        for (auto const &input : current)
        {
            write_leaf(nodes, input, config_, n_leaves);
        }
        stamp_leaf_rows(nodes, current, values, leaf_ids);
    }

    void finalize_leaves(std::vector<SplitInput> const &frontier,
                         std::vector<float> const      &leaf_table,
                         train_leaf_values &values, std::vector<node_id_t> &leaf_ids,
                         row_index_view /*row_indices*/,
                         std::span<ObliviousTree::LevelSplit const> /*level_splits*/,
                         std::span<bin_id_t const> /*level_bins*/)
    {
        parallel::for_each_index(frontier.size(),
                                 [&](size_t li)
                                 {
                                     for (row_id_t const r : frontier[li].rows)
                                     {
                                         values[r]   = leaf_table[li];
                                         leaf_ids[r] = static_cast<node_id_t>(li);
                                     }
                                 });
    }
};

template <GPULevelEngine EngineT, typename SplitterT>
class LevelStep<EngineT, SplitterT> : public TreeStep<EngineT>
{
    using Step = TreeStep<EngineT>;
    using Step::config_;
    using Step::ds_;
    using Step::engine_;
    using Step::grad_;
    using Step::hess_;
    using Step::selected_;

  public:
    using Step::Step;

    SplitInput make_root(RowSelection const &sel)
    {
        Phase<&GrowProfiler::populate_s> phase;
        SplitInput                       root = device_root(sel);
        engine_.begin_root(ds_, grad_, hess_, root, selected_);
        return root;
    }

    void open_level(std::vector<SplitInput> const &frontier, LevelOutputs &lout)
    {
        Phase<&GrowProfiler::find_s> phase;
        auto                        &out        = lout.splits;
        auto                        &child_sums = lout.child_sums;
        auto const                  &current    = frontier;
        out.clear();
        child_sums.clear();
        out.resize(current.size());
        child_sums.resize(2 * current.size());
        static_assert(std::same_as<SplitterT, HistogramLevelSplitFinder> ||
                      std::same_as<SplitterT, HistogramNodeSplitFinder>);
        if constexpr (LevelSplitFinder<SplitterT>)
        {
            engine_.find_level_split(ds_, config_, current, out, child_sums);
        }
        else
        {
            engine_.find_splits_many(ds_, config_, current, out, child_sums);
        }
    }

    void apply_level(LevelPlan &plan)
    {
        Phase<&GrowProfiler::partition_s>        phase;
        std::vector<typename EngineT::LeafStamp> stamps;
        stamps.reserve(plan.leaves.size());
        for (SlotLeaf const &sl : plan.leaves)
        {
            stamps.push_back({sl.slot, sl.node_id});
        }
        engine_.stamp_leaves(stamps);

        std::vector<typename EngineT::PartitionOp> pops;
        pops.reserve(plan.splits.size());
        for (uint32_t k = 0; k < plan.splits.size(); ++k)
        {
            DeferredSplit const &d = plan.splits[k];
            pops.push_back({d.parent_slot, 2 * k, (2 * k) + 1, d.split.feature_id,
                            d.split.bin_id, d.split.default_left});
        }
        std::vector<uint32_t> counts(2 * plan.splits.size(), 0);
        engine_.partition_level(ds_, pops, counts);
        for (uint32_t k = 0; k < plan.splits.size(); ++k)
        {
            DeferredSplit &d    = plan.splits[k];
            d.p.left.id         = d.left_id;
            d.p.right.id        = d.right_id;
            d.p.left.sums       = d.left_sums;
            d.p.right.sums      = d.right_sums;
            d.p.left.row_count  = counts[2 * k];
            d.p.right.row_count = counts[(2 * k) + 1];
        }
    }

    void build_children(LevelPlan &plan, bool last = false)
    {
        Phase<&GrowProfiler::populate_s> phase;
        if (last)
        {
            engine_.advance_layout_only();
            return;
        }
        std::vector<typename EngineT::LevelOp> ops;
        ops.reserve(plan.splits.size());
        for (uint32_t k = 0; k < plan.splits.size(); ++k)
        {
            DeferredSplit const &d          = plan.splits[k];
            bool const           left_small = d.p.left.row_count <= d.p.right.row_count;
            ops.push_back({d.parent_slot, (2 * k) + (left_small ? 0U : 1U),
                           (2 * k) + (left_small ? 1U : 0U)});
        }
        engine_.advance_level(ds_, ops);
    }

    void end_tree(std::vector<SplitInput> const &current, DenseTree::Nodes &nodes,
                  size_t &n_leaves, train_leaf_values &values,
                  std::vector<node_id_t> &leaf_ids, row_index_view /*row_indices*/)
    {
        bool const resident = engine_.resident_armed();
        if (!resident)
        {
            engine_.stamp_leaves(leaf_stamps<typename EngineT::LeafStamp>(
                current.size(), [&](uint32_t i) { return current[i].id; }));
        }
        for (auto const &input : current)
        {
            finalize_as_leaf(nodes, input, config_, n_leaves, values, leaf_ids);
        }
        if (resident)
        {
            engine_.resident_finalize(
                resident_node_table<typename EngineT::ResidentNode>(nodes, ds_));
            return;
        }
        engine_.finalize_tree(node_values(nodes), values, leaf_ids);
    }

    void finalize_leaves(std::vector<SplitInput> const &frontier,
                         std::vector<float> const      &leaf_table,
                         train_leaf_values &values, std::vector<node_id_t> &leaf_ids,
                         row_index_view /*row_indices*/,
                         std::span<ObliviousTree::LevelSplit const> level_splits,
                         std::span<bin_id_t const>                  level_bins)
    {
        if (engine_.resident_armed())
        {
            engine_.resident_finalize(
                grower_detail::perfect_tree_table<typename EngineT::ResidentNode>(
                    level_splits.size(),
                    [&](size_t lvl)
                    {
                        return grower_detail::LevelSplitBins{
                            level_splits[lvl].feature_id, level_bins[lvl],
                            level_splits[lvl].default_left};
                    },
                    leaf_table));
            return;
        }
        engine_.stamp_leaves(leaf_stamps<typename EngineT::LeafStamp>(
            frontier.size(), [](uint32_t i) { return static_cast<node_id_t>(i); }));
        engine_.finalize_tree(leaf_table, values, leaf_ids);
    }
};

} // namespace bonsai::grower_detail
