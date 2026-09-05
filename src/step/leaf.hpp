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
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace bonsai::grower_detail
{

template <HistogramEngine EngineT, ParallelNodeSplitFinder SplitterT>
class LeafStep : public TreeStep<EngineT>
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

    Candidate open_root(RowSelection const &sel)
    {
        GrowProfiler::Lap lap;
        SplitInput root = populated_root(ds_, grad_, hess_, sel, selected_, engine_);
        lap(GrowProfiler::instance().populate_s);
        std::array<SplitOutput, 1> split{};
        SplitterT::find_parallel({&root, 1}, config_, split);
        lap(GrowProfiler::instance().find_s);
        return {.node = std::move(root), .split = split[0], .depth = 0};
    }

    ChildPair split_children(Candidate &c, node_id_t left_id, node_id_t right_id)
    {
        auto [left, right] = split_node(ds_, grad_, hess_, std::move(c.node), c.split,
                                        left_id, right_id, selected_, engine_);
        return {.nodes = {std::move(left), std::move(right)},
                .depth = static_cast<uint8_t>(c.depth + 1)};
    }

    SplitInput demoted_leaf(Candidate const &c, ChildPair &pair, double parent_lo,
                            double parent_hi)
    {
        SplitInput &survivor =
            pair.nodes[0].rows.empty() ? pair.nodes[1] : pair.nodes[0];
        survivor.id = c.node.id;
        survivor.lo = parent_lo;
        survivor.hi = parent_hi;
        return std::move(survivor);
    }

    void find_children(ChildPair &pair, bool may_split)
    {
        Phase<&GrowProfiler::find_s> phase;
        if (!may_split)
        {
            pair.splits = {};
            return;
        }
        SplitterT::find_parallel(pair.nodes, config_, pair.splits);
    }

    void leaf(node_id_t /*id*/, uint32_t /*slot*/) {}

    void end_tree(DenseTree::Nodes const & /*nodes*/, train_leaf_values & /*values*/,
                  std::vector<node_id_t> & /*leaf_ids*/)
    {
    }
};

template <GPULeafEngine EngineT, ParallelNodeSplitFinder SplitterT>
class LeafStep<EngineT, SplitterT> : public TreeStep<EngineT>
{
    using Step = TreeStep<EngineT>;
    using Step::config_;
    using Step::ds_;
    using Step::engine_;
    using Step::grad_;
    using Step::hess_;
    using Step::selected_;

  public:
    LeafStep(EngineT &engine, Dataset const &ds, TreeConfig const &config,
             floats_view grad, floats_view hess, feature_view selected)
        : Step(engine, ds, config, grad, hess, selected),
          resident_(engine_.resident_armed())
    {
    }

    Candidate open_root(RowSelection const &sel)
    {
        GrowProfiler::Lap lap;
        SplitInput        root = device_root(sel);
        engine_.leaf_begin_root(ds_, config_, grad_, hess_, root, selected_);
        release_staged_rows(root);
        lap(GrowProfiler::instance().populate_s);
        std::array<uint32_t, 1> const slots{0};
        std::array<SplitOutput, 1>    out{};
        std::array<NodeTotals, 2>     sums{};
        engine_.leaf_find(ds_, config_, std::span<SplitInput const>{&root, 1}, slots,
                          out, sums);
        lap(GrowProfiler::instance().find_s);
        return {.node       = std::move(root),
                .split      = out[0],
                .depth      = 0,
                .slot       = 0,
                .left_sums  = sums[0],
                .right_sums = sums[1]};
    }

    ChildPair split_children(Candidate &c, node_id_t left_id, node_id_t right_id)
    {
        Phase<&GrowProfiler::partition_s>  phase;
        typename EngineT::LeafPartOp const op{c.slot, c.split.feature_id,
                                              c.split.bin_id, c.split.default_left};
        auto const                         round = engine_.leaf_split(ds_, op);
        ChildPair                          pair;
        pair.depth              = static_cast<uint8_t>(c.depth + 1);
        pair.nodes[0].id        = left_id;
        pair.nodes[0].row_count = round.left_count;
        pair.nodes[0].sums      = c.left_sums;
        pair.nodes[1].id        = right_id;
        pair.nodes[1].row_count = round.right_count;
        pair.nodes[1].sums      = c.right_sums;
        pair.slots              = {round.left_slot, round.right_slot};
        return pair;
    }

    SplitInput demoted_leaf(Candidate &c, ChildPair & /*pair*/, double /*parent_lo*/,
                            double /*parent_hi*/)
    {
        return std::move(c.node);
    }

    void find_children(ChildPair &pair, bool may_split)
    {
        if (!may_split)
        {
            pair.splits = {};
            return;
        }
        GrowProfiler::Lap lap;
        bool const     left_small = pair.nodes[0].row_count <= pair.nodes[1].row_count;
        uint32_t const small_slot = left_small ? pair.slots[0] : pair.slots[1];
        uint32_t const large_slot = left_small ? pair.slots[1] : pair.slots[0];
        engine_.leaf_build(ds_, small_slot, large_slot);
        lap(GrowProfiler::instance().populate_s);
        engine_.leaf_find(ds_, config_, pair.nodes, pair.slots, pair.splits,
                          pair.child_sums);
        lap(GrowProfiler::instance().find_s);
    }

    void leaf(node_id_t id, uint32_t slot)
    {
        if (!resident_)
        {
            stamps_.push_back({slot, id});
        }
    }

    void end_tree(DenseTree::Nodes const &nodes, train_leaf_values &values,
                  std::vector<node_id_t> &leaf_ids)
    {
        if (resident_)
        {
            engine_.resident_finalize(
                resident_node_table<typename EngineT::ResidentNode>(nodes, ds_));
            return;
        }
        engine_.leaf_stamp(stamps_);
        stamps_.clear();
        engine_.finalize_tree(node_values(nodes), values, leaf_ids);
    }

  private:
    std::vector<typename EngineT::LeafStamp> stamps_;
    bool                                     resident_;
};

} // namespace bonsai::grower_detail
