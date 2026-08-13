#pragma once

// LeafStep: the data plane for best-first growth, host plane and device
// plane. One node at a time, so there is no level to batch.
//
// Included into src/level_step.hpp, which is the only include of this
// header: the data plane stays in one translation unit so nothing here
// crosses a call the optimizer cannot see.

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
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace bonsai::grower_detail
{

// Host data plane for best-first growth: the gain heap expands one node at a
// time, so there is no level to batch and no LevelPlan — the same
// partition/populate/subtract primitives compose one node deep.
// Branch-free: no GPU concept appears below.
template <HistogramEngine EngineT, ParallelNodeSplitFinder SplitterT> class LeafStep
{
  public:
    LeafStep(EngineT &engine, Dataset const &ds, TreeConfig const &config,
             floats_view grad, floats_view hess, feature_view selected)
        : engine_(engine), ds_(ds), config_(config), grad_(grad), hess_(hess),
          selected_(selected)
    {
        engine_.begin_tree(ds_, grad_, hess_);
    }

    // Opens the tree and seeds the heap: the root node plus its best split.
    Candidate open_root(row_index_view row_indices)
    {
        return host_open_root(engine_, ds_, config_, grad_, hess_, selected_,
                              row_indices);
    }

    // Routes the popped candidate's rows into its two children: they come back
    // with rows, histograms, and totals, their splits still empty.
    ChildPair split_children(Candidate &c, node_id_t left_id, node_id_t right_id)
    {
        return host_split_children(engine_, ds_, grad_, hess_, selected_, c, left_id,
                                   right_id);
    }

    // A split whose partition emptied one child demotes back to a leaf: the
    // surviving child holds every one of the parent's rows and takes over the
    // parent's id and bounds.
    SplitInput demoted_leaf(Candidate const &c, ChildPair &pair, double parent_lo,
                            double parent_hi)
    {
        return host_demoted_leaf(c, pair, parent_lo, parent_hi);
    }

    // Each child's best split, once its histograms are ready. At the depth cap
    // the children can only be leaves, so no split is looked for.
    void find_children(ChildPair &pair, bool may_split)
    {
        host_find_children(config_, pair, may_split);
    }

    // This node is final; the host plane already stamped its rows.
    void leaf(node_id_t /*id*/, uint32_t /*slot*/) {}

    void end_tree(DenseTree::Nodes const & /*nodes*/, train_leaf_values & /*values*/,
                  std::vector<node_id_t> & /*leaf_ids*/)
    {
    }

    // --- shared host ops (the GPU specialization's fallback calls these) ----

    template <HistogramEngine E>
    static Candidate host_open_root(E &engine, Dataset const &ds,
                                    TreeConfig const &config, floats_view grad,
                                    floats_view hess, feature_view selected,
                                    row_index_view row_indices)
    {
        GrowProfiler::Lap lap;
        SplitInput        root;
        root.id = 0;
        root.rows.assign(row_indices.begin(), row_indices.end());
        engine.populate(ds, grad, hess, root, selected);
        root.sums      = root.totals();
        root.row_count = root.rows.size();
        lap(GrowProfiler::instance().populate_s);
        std::array<SplitOutput, 1> split{};
        SplitterT::find_parallel({&root, 1}, config, split);
        lap(GrowProfiler::instance().find_s);
        return {.node = std::move(root), .split = split[0], .depth = 0};
    }

    template <HistogramEngine E>
    static ChildPair host_split_children(E &engine, Dataset const &ds, floats_view grad,
                                         floats_view hess, feature_view selected,
                                         Candidate &c, node_id_t left_id,
                                         node_id_t right_id)
    {
        auto [left, right] = split_node(ds, grad, hess, std::move(c.node), c.split,
                                        left_id, right_id, selected, engine);
        return {.nodes = {std::move(left), std::move(right)},
                .depth = static_cast<uint8_t>(c.depth + 1)};
    }

    static SplitInput host_demoted_leaf(Candidate const &c, ChildPair &pair,
                                        double parent_lo, double parent_hi)
    {
        SplitInput &survivor =
            pair.nodes[0].rows.empty() ? pair.nodes[1] : pair.nodes[0];
        survivor.id = c.node.id;
        survivor.lo = parent_lo;
        survivor.hi = parent_hi;
        return std::move(survivor);
    }

    // Best-first growth has no frontier to spread across workers, so a node
    // takes its parallelism over features. A split's two children go in
    // together: one parallel region, not two.
    static void host_find_children(TreeConfig const &config, ChildPair &pair,
                                   bool may_split)
    {
        Phase<&GrowProfiler::find_s> phase;
        if (!may_split)
        {
            pair.splits = {};
            return;
        }
        SplitterT::find_parallel(pair.nodes, config, pair.splits);
    }

  protected:
    EngineT          &engine_;
    Dataset const    &ds_;
    TreeConfig const &config_;
    floats_view       grad_;
    floats_view       hess_;
    feature_view      selected_;
};

// ---------------------------------------------------------------------------
// GPU leaf plane (docs/architecture/20-cuda-leafwise.md): one histogram slot
// pool per tree, one partition per heap pop, and the same one runtime fork the
// level plane has — leaf_begin_root declines (oversized max_bin, pool won't
// fit) and the step falls back to the host ops above for the rest of the tree.
// There is no per-node fallback: a tree is device-resident or it is not.
template <GPULeafEngine EngineT, ParallelNodeSplitFinder SplitterT>
class LeafStep<EngineT, SplitterT>
{
  public:
    LeafStep(EngineT &engine, Dataset const &ds, TreeConfig const &config,
             floats_view grad, floats_view hess, feature_view selected)
        : engine_(engine), ds_(ds), config_(config), grad_(grad), hess_(hess),
          selected_(selected), resident_(engine_.resident_armed())
    {
        engine_.begin_tree(ds_, grad_, hess_);
    }

    Candidate open_root(row_index_view row_indices)
    {
        GrowProfiler::Lap lap;
        SplitInput        root;
        root.id = 0;
        // Identity contract, as on the level plane: a full-data fit passes
        // empty rows + row_count and the permutation never crosses the bus.
        bool const identity = row_indices.size() == ds_.n_rows();
        if (identity)
        {
            root.row_count = row_indices.size();
        }
        else
        {
            root.rows.assign(row_indices.begin(), row_indices.end());
        }
        on_device_ =
            engine_.leaf_begin_root(ds_, config_, grad_, hess_, root, selected_);
        if (!on_device_)
        {
            if (identity)
            {
                // The host fallback walks explicit row lists; row_count must
                // drop back to 0 or totals() would return the zeroed cached
                // sums instead of the histogram totals populate is about to
                // build (SplitInput's cached-statistics contract).
                root.rows.assign(row_indices.begin(), row_indices.end());
                root.row_count = 0;
            }
            engine_.populate(ds_, grad_, hess_, root, selected_);
            root.sums      = root.totals();
            root.row_count = root.rows.size();
            lap(GrowProfiler::instance().populate_s);
            SplitOutput const split = SplitterT::find(root, config_);
            lap(GrowProfiler::instance().find_s);
            return {.node = std::move(root), .split = split, .depth = 0};
        }
        lap(GrowProfiler::instance().populate_s);
        std::array<uint32_t, 1> const slots{0};
        std::array<SplitOutput, 1>    out{};
        std::array<HistCell, 2>       sums{};
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

    // Device children carry row counts, pool slots, and the sums the parent's
    // find already produced; rows never cross the bus.
    ChildPair split_children(Candidate &c, node_id_t left_id, node_id_t right_id)
    {
        if (!on_device_)
        {
            return HostStep::host_split_children(engine_, ds_, grad_, hess_, selected_,
                                                 c, left_id, right_id);
        }
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

    // Device demotion: the partition took no slot, so the parent keeps its
    // histogram, its segment, and its totals, and becomes the leaf itself.
    SplitInput demoted_leaf(Candidate &c, ChildPair &pair, double parent_lo,
                            double parent_hi)
    {
        if (!on_device_)
        {
            return HostStep::host_demoted_leaf(c, pair, parent_lo, parent_hi);
        }
        return std::move(c.node);
    }

    // The smaller child builds from its device row segment, the larger derives
    // in place in the slot it inherited, and one find covers both.
    void find_children(ChildPair &pair, bool may_split)
    {
        if (!on_device_)
        {
            HostStep::host_find_children(config_, pair, may_split);
            return;
        }
        if (!may_split)
        {
            // The children are leaves at the depth cap: their histograms would
            // never be read, so only their segments live on.
            pair.splits = {};
            return;
        }
        GrowProfiler::Lap lap;
        // The smaller child built the fresh slot in split_children; size tie:
        // left wins (smaller_child, matched by leaf_split on the device).
        bool const     left_small = pair.nodes[0].row_count <= pair.nodes[1].row_count;
        uint32_t const small_slot = left_small ? pair.slots[0] : pair.slots[1];
        uint32_t const large_slot = left_small ? pair.slots[1] : pair.slots[0];
        engine_.leaf_build(ds_, small_slot, large_slot);
        lap(GrowProfiler::instance().populate_s);
        engine_.leaf_find(ds_, config_, pair.nodes, pair.slots, pair.splits,
                          pair.child_sums);
        lap(GrowProfiler::instance().find_s);
    }

    // A finalized leaf's segment never moves again, so the stamps accumulate
    // and the tree epilogue replays them in one launch. The resident epilogue
    // routes every row through the finished tree instead, so it reads no leaf
    // assignment and nothing is stamped.
    void leaf(node_id_t id, uint32_t slot)
    {
        if (on_device_ && !resident_)
        {
            stamps_.push_back({slot, id});
        }
    }

    // End of tree: stamp every leaf's segment, then hand the node value table
    // to the engine, which maps every resident row to its leaf value on device
    // and returns values/leaf_ids in two bulk copies. In resident mode neither
    // copy happens: the same route+add epilogue the level plane uses fuses the
    // score update on device, so the tree's per-row output never exists.
    void end_tree(DenseTree::Nodes const &nodes, train_leaf_values &values,
                  std::vector<node_id_t> &leaf_ids)
    {
        if (!on_device_)
        {
            return;
        }
        if (resident_)
        {
            engine_.resident_finalize(
                resident_node_table<typename EngineT::ResidentNode>(nodes, ds_));
            return;
        }
        engine_.leaf_stamp(stamps_);
        stamps_.clear();
        std::vector<float> node_vals(nodes.size());
        for (size_t i = 0; i < nodes.size(); ++i)
        {
            node_vals[i] = nodes[i].threshold_or_value;
        }
        engine_.finalize_tree(node_vals, values, leaf_ids);
    }

  private:
    // The host ops are static on the primary template; name it with a host
    // engine stand-in so the fallback arms reuse them without duplication.
    using HostStep = LeafStep<CpuHistogramEngine, SplitterT>;

    EngineT                                 &engine_;
    Dataset const                           &ds_;
    TreeConfig const                        &config_;
    floats_view                              grad_;
    floats_view                              hess_;
    feature_view                             selected_;
    std::vector<typename EngineT::LeafStamp> stamps_;
    bool                                     on_device_ = false;
    // Captured once at construction: arming happens per fit before any tree
    // opens (see LeafwiseGrower::resident_begin), so it cannot change while
    // this step is alive.
    bool resident_;
};

} // namespace bonsai::grower_detail
