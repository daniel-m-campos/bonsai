#pragma once

// LevelStep: the data plane for level-batched growth (depthwise and
// levelwise), host plane and device plane.
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
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace bonsai::grower_detail
{

// Host data plane: serves every HistogramEngine. Branch-free: no GPU concept
// appears below.
template <HistogramEngine EngineT, typename SplitterT> class LevelStep
{
  public:
    LevelStep(EngineT &engine, Dataset const &ds, TreeConfig const &config,
              floats_view grad, floats_view hess, feature_view selected)
        : engine_(engine), ds_(ds), config_(config), grad_(grad), hess_(hess),
          selected_(selected)
    {
        engine_.begin_tree(ds_, grad_, hess_);
    }

    SplitInput make_root(row_index_view row_indices)
    {
        Phase<&GrowProfiler::populate_s> phase;
        SplitInput                       root;
        root.id = 0;
        root.rows.assign(row_indices.begin(), row_indices.end());
        // Checked once per tree, with an early exit on the first row that is
        // not its own index, against a fill that walks these rows once per
        // feature per level.
        root.rows_identity = rows_are_identity(root.rows, ds_.n_rows());
        engine_.populate(ds_, grad_, hess_, root, selected_);
        root.sums      = root.totals();
        root.row_count = root.rows.size();
        return root;
    }

    // Per-node splitter, or one level-wide find broadcast to every node when
    // the splitter is level-granular (the levelwise growth shape).
    // Level transaction, phase 1: split decisions for the whole frontier
    // (decision 53). The frontier is the transaction's input; outputs are
    // caller-owned and reused across levels.
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
            // One worker per node: the finder's inner feature loops nest
            // serial and its scratch is thread_local, so per-node results
            // are identical to the serial walk. A region per level replaces
            // a region per node.
            out.splits.resize(frontier.size());
            SplitInput const *cur = frontier.data();
            SplitOutput      *op  = out.splits.data();
            parallel::for_each_index(frontier.size(), [&, cur, op](size_t i)
                                     { op[i] = SplitterT::find(cur[i], config_); });
        }
    }

    // Routes every split parent's rows into its children, one node per worker
    // (each partition touches only its own parent's rows: bit-identical to
    // serial at any thread count).
    void apply_level(LevelPlan &plan)
    {
        Phase<&GrowProfiler::partition_s> phase;
        host_partition(ds_, plan);
    }

    // Fills every smaller child's histograms in one engine call; the larger
    // sibling derives by subtraction.
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

    // End of tree: the surviving frontier becomes leaves. Values and the
    // node table are serial (tiny); the row stamping takes one region.
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

    // Levelwise leaf finalize, host plane: each frontier node is a leaf,
    // indexed by position into leaf_table; stamp its rows directly. The level
    // split/bin spans feed only the GPU resident finalize; the host ignores
    // them.
    void finalize_leaves(std::vector<SplitInput> const &frontier,
                         std::vector<float> const      &leaf_table,
                         train_leaf_values &values, std::vector<node_id_t> &leaf_ids,
                         row_index_view /*row_indices*/,
                         std::span<ObliviousTree::LevelSplit const> /*level_splits*/,
                         std::span<bin_id_t const> /*level_bins*/)
    {
        // Leaf-parallel: leaves partition the sampled rows, so writes never
        // collide; dynamic scheduling absorbs the uneven leaf sizes.
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

  protected:
    EngineT          &engine_;
    Dataset const    &ds_;
    TreeConfig const &config_;
    floats_view       grad_;
    floats_view       hess_;
    feature_view      selected_;
};

// GPU data plane: histograms live in device slot buffers and rows in device
// segments; only decisions, child sums, and counts cross the bus. Every tree
// runs there: begin_root refuses a tree the device cannot hold rather than
// moving it to the host plane, so this step has no runtime fork.
template <GPULevelEngine EngineT, typename SplitterT>
class LevelStep<EngineT, SplitterT>
{
  public:
    LevelStep(EngineT &engine, Dataset const &ds, TreeConfig const &config,
              floats_view grad, floats_view hess, feature_view selected)
        : engine_(engine), ds_(ds), config_(config), grad_(grad), hess_(hess),
          selected_(selected)
    {
        engine_.begin_tree(ds_, grad_, hess_);
    }

    SplitInput make_root(row_index_view row_indices)
    {
        Phase<&GrowProfiler::populate_s> phase;
        SplitInput                       root;
        root.id = 0;
        // Full-data fits pass the identity by contract (empty rows +
        // row_count): the 64MB host copy and its upload never happen; the
        // engine builds/caches the identity on device. Identity, not
        // cardinality: any other full-length list takes the general path.
        if (rows_are_identity(row_indices, ds_.n_rows()))
        {
            root.row_count = row_indices.size();
        }
        else
        {
            root.rows.assign(row_indices.begin(), row_indices.end());
        }
        engine_.begin_root(ds_, grad_, hess_, root, selected_);
        return root; // hists/rows stay device-resident; root carries sums
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
        // Levelwise (LevelSplitFinder) picks one split for the whole frontier;
        // depthwise/leafwise pick one per node. The engine kernels hardcode
        // histogram-gain scoring, so only the histogram finders may select
        // this plane.
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

    // Leaves stamp their device segments, splits partition on the device
    // (only child counts return), and each child's ids/sums/row_count fill in
    // from the counts — SplitInput degrades to node metadata on this plane.
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

    // Smaller children build from their device row segments; the larger
    // derive on-device as parent minus smaller, then the child level becomes
    // current.
    void build_children(LevelPlan &plan, bool last = false)
    {
        Phase<&GrowProfiler::populate_s> phase;
        if (last)
        {
            // The last level's children are leaves: their histograms are never
            // read, so skip the build and keep the layout flip stamping
            // depends on (decision 71).
            engine_.advance_layout_only();
            return;
        }
        std::vector<typename EngineT::LevelOp> ops;
        ops.reserve(plan.splits.size());
        for (uint32_t k = 0; k < plan.splits.size(); ++k)
        {
            DeferredSplit const &d = plan.splits[k];
            // Size tie: left wins (smaller_child).
            bool const left_small = d.p.left.row_count <= d.p.right.row_count;
            ops.push_back({d.parent_slot, (2 * k) + (left_small ? 0U : 1U),
                           (2 * k) + (left_small ? 1U : 0U)});
        }
        engine_.advance_level(ds_, ops);
    }

    // End of tree: stamp the surviving frontier's device segments, then hand
    // the node value table to the engine, which maps every resident row to
    // its leaf value on device and returns values/leaf_ids in two bulk copies
    // (host finalize_as_leaf writes the nodes; its row loop no-ops on empty
    // device-mode rows). Rows outside the sampled set get placeholder stamps
    // here and are overwritten by route_unsampled.
    void end_tree(std::vector<SplitInput> const &current, DenseTree::Nodes &nodes,
                  size_t &n_leaves, train_leaf_values &values,
                  std::vector<node_id_t> &leaf_ids, row_index_view /*row_indices*/)
    {
        bool const resident = engine_.resident_armed();
        if (!resident)
        {
            std::vector<typename EngineT::LeafStamp> stamps;
            stamps.reserve(current.size());
            for (uint32_t i = 0; i < current.size(); ++i)
            {
                stamps.push_back({i, current[i].id});
            }
            engine_.stamp_leaves(stamps);
        }
        for (auto const &input : current)
        {
            finalize_as_leaf(nodes, input, config_, n_leaves, values, leaf_ids);
        }
        if (resident)
        {
            // The finished dense tree routes every row on device; no stamp, no
            // values/leaf_ids D2H.
            engine_.resident_finalize(
                resident_node_table<typename EngineT::ResidentNode>(nodes, ds_));
            return;
        }
        std::vector<float> node_vals(nodes.size());
        for (size_t i = 0; i < nodes.size(); ++i)
        {
            node_vals[i] = nodes[i].threshold_or_value;
        }
        engine_.finalize_tree(node_vals, values, leaf_ids);
    }

    // Levelwise leaf finalize: the frontier nodes are the leaves, indexed by
    // position into leaf_table. The rows are device-resident, so stamp each
    // final slot with its leaf index and download the per-row assignment.
    void finalize_leaves(std::vector<SplitInput> const &frontier,
                         std::vector<float> const      &leaf_table,
                         train_leaf_values &values, std::vector<node_id_t> &leaf_ids,
                         row_index_view /*row_indices*/,
                         std::span<ObliviousTree::LevelSplit const> level_splits,
                         std::span<bin_id_t const>                  level_bins)
    {
        if (engine_.resident_armed())
        {
            // The perfect-tree numbering (grower_detail::perfect_tree_table)
            // lets the one device route+add kernel serve both tree shapes;
            // this path's per-level bins are already computed, so its split
            // accessor just indexes.
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
        std::vector<typename EngineT::LeafStamp> stamps;
        stamps.reserve(frontier.size());
        for (uint32_t i = 0; i < frontier.size(); ++i)
        {
            stamps.push_back({i, static_cast<node_id_t>(i)});
        }
        engine_.stamp_leaves(stamps);
        engine_.finalize_tree(leaf_table, values, leaf_ids);
    }

  private:
    EngineT          &engine_;
    Dataset const    &ds_;
    TreeConfig const &config_;
    floats_view       grad_;
    floats_view       hess_;
    feature_view      selected_;
};

} // namespace bonsai::grower_detail
