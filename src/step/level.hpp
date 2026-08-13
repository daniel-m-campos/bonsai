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

// Host data plane: serves every HistogramEngine — the CPU engine, and any
// engine's CPU fallback. Branch-free: no GPU concept appears below.
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
        host_find<SplitterT>(frontier, config_, out.splits, out.child_sums);
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
        Phase<&GrowProfiler::populate_s> phase;
        host_build_children(engine_, ds_, grad_, hess_, selected_, plan);
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
        host_finalize_leaves(frontier, leaf_table, values, leaf_ids);
    }

    static void host_finalize_leaves(std::vector<SplitInput> const &frontier,
                                     std::vector<float> const      &leaf_table,
                                     train_leaf_values             &values,
                                     std::vector<node_id_t>        &leaf_ids)
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

    // --- shared host ops (the GPU specialization's fallback calls these) ----

    template <typename S>
    static void host_find(std::vector<SplitInput> const &current,
                          TreeConfig const &config, std::vector<SplitOutput> &out,
                          std::vector<HistCell> &child_sums)
    {
        out.clear();
        child_sums.clear();
        if constexpr (LevelSplitFinder<S>)
        {
            SplitOutput const split = S::find(current, config);
            out.assign(current.size(), split);
        }
        else
        {
            // One worker per node: the finder's inner feature loops nest
            // serial and its scratch is thread_local, so per-node results
            // are identical to the serial walk. A region per level replaces
            // a region per node.
            out.resize(current.size());
            SplitInput const *cur = current.data();
            SplitOutput      *op  = out.data();
            parallel::for_each_index(current.size(), [&, cur, op](size_t i)
                                     { op[i] = S::find(cur[i], config); });
        }
    }

    template <HistogramEngine E>
    static void host_build_children(E &engine, Dataset const &ds, floats_view grad,
                                    floats_view hess, feature_view selected,
                                    LevelPlan &plan)
    {
        std::vector<std::reference_wrapper<SplitInput>> smalls;
        smalls.reserve(plan.splits.size());
        for (auto &d : plan.splits)
        {
            smalls.emplace_back(smaller_child(d.p));
        }
        populate_nodes(ds, grad, hess, smalls, selected, engine);
        for (auto &d : plan.splits)
        {
            finish_split(ds, d.p);
        }
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
// segments; only decisions, child sums, and counts cross the bus. Holds the
// per-tree mode — the ONE runtime fork in the design: begin_root declines
// (oversized max_bin, buffers won't fit) and the step falls back to the host
// ops above for the rest of the tree.
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
        // engine builds/caches the identity on device (decision 71).
        bool const identity = row_indices.size() == ds_.n_rows();
        if (identity)
        {
            root.row_count = row_indices.size();
        }
        else
        {
            root.rows.assign(row_indices.begin(), row_indices.end());
        }
        on_device_ = engine_.begin_root(ds_, grad_, hess_, root, selected_);
        if (on_device_)
        {
            return root; // hists/rows stay device-resident; root carries sums
        }
        if (identity)
        {
            // The host fallback walks explicit row lists; materialize the
            // identity only on this cold path. row_count must drop back to 0
            // or totals() would return the zeroed cached sums instead of the
            // histogram totals populate is about to build (SplitInput's
            // cached-statistics contract).
            root.rows.assign(row_indices.begin(), row_indices.end());
            root.row_count = 0;
        }
        engine_.populate(ds_, grad_, hess_, root, selected_);
        root.sums      = root.totals();
        root.row_count = root.rows.size();
        return root;
    }

    void open_level(std::vector<SplitInput> const &frontier, LevelOutputs &lout)
    {
        Phase<&GrowProfiler::find_s> phase;
        auto                        &out        = lout.splits;
        auto                        &child_sums = lout.child_sums;
        auto const                  &current    = frontier;
        if (on_device_)
        {
            out.clear();
            child_sums.clear();
            out.resize(current.size());
            child_sums.resize(2 * current.size());
            // Levelwise (LevelSplitFinder) picks one split for the whole
            // frontier; depthwise/leafwise pick one per node. The engine
            // kernels hardcode histogram-gain scoring, so only the histogram
            // finders may select this plane.
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
        else
        {
            HostStep::template host_find<SplitterT>(current, config_, out, child_sums);
        }
    }

    // Leaves stamp their device segments, splits partition on the device
    // (only child counts return), and each child's ids/sums/row_count fill in
    // from the counts — SplitInput degrades to node metadata on this plane.
    void apply_level(LevelPlan &plan)
    {
        Phase<&GrowProfiler::partition_s> phase;
        if (on_device_)
        {
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
        else
        {
            host_partition(ds_, plan);
        }
    }

    // Smaller children build from their device row segments; the larger
    // derive on-device as parent minus smaller, then the child level becomes
    // current.
    void build_children(LevelPlan &plan, bool last = false)
    {
        Phase<&GrowProfiler::populate_s> phase;
        if (on_device_)
        {
            if (last)
            {
                // The last level's children are leaves: their histograms are
                // never read, so skip the build and keep the layout flip
                // stamping depends on (decision 71).
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
        else
        {
            HostStep::host_build_children(engine_, ds_, grad_, hess_, selected_, plan);
        }
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
        bool const resident = on_device_ && engine_.resident_armed();
        if (on_device_ && !resident)
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
        if (on_device_)
        {
            std::vector<float> node_vals(nodes.size());
            for (size_t i = 0; i < nodes.size(); ++i)
            {
                node_vals[i] = nodes[i].threshold_or_value;
            }
            engine_.finalize_tree(node_vals, values, leaf_ids);
        }
    }

    // Levelwise leaf finalize: the frontier nodes are the leaves, indexed by
    // position into leaf_table. On device the rows are resident, so stamp each
    // final slot with its leaf index and download the per-row assignment; in
    // fallback mode the rows live on the host and must be stamped here — the
    // early return that assumed otherwise trained silent garbage (issue #12).
    void finalize_leaves(std::vector<SplitInput> const &frontier,
                         std::vector<float> const      &leaf_table,
                         train_leaf_values &values, std::vector<node_id_t> &leaf_ids,
                         row_index_view /*row_indices*/,
                         std::span<ObliviousTree::LevelSplit const> level_splits,
                         std::span<bin_id_t const>                  level_bins)
    {
        if (!on_device_)
        {
            HostStep::host_finalize_leaves(frontier, leaf_table, values, leaf_ids);
            return;
        }
        if (engine_.resident_armed())
        {
            // Synthesize the perfect-tree node numbering (children 2i+1 / 2i+2)
            // from the per-level splits so the one device route+add kernel
            // serves both tree shapes; leaves 2^D-1 .. hold leaf_table in
            // left-to-right order (the oblivious index is exactly that order).
            size_t const depth      = level_splits.size();
            size_t const n_internal = (size_t{1} << depth) - 1;
            size_t const n_leaves   = size_t{1} << depth;
            std::vector<typename EngineT::ResidentNode> table(n_internal + n_leaves);
            for (size_t i = 0; i < n_internal; ++i)
            {
                size_t lvl = 0;
                size_t cap = 1;
                size_t off = i;
                while (off >= cap)
                {
                    off -= cap;
                    cap <<= 1U;
                    ++lvl;
                }
                typename EngineT::ResidentNode &rn = table[i];
                rn.feature_id                      = level_splits[lvl].feature_id;
                rn.split_bin                       = level_bins[lvl];
                rn.default_left                    = level_splits[lvl].default_left;
                rn.left  = static_cast<node_id_t>((2 * i) + 1);
                rn.right = static_cast<node_id_t>((2 * i) + 2);
            }
            for (size_t j = 0; j < n_leaves; ++j)
            {
                typename EngineT::ResidentNode &rn = table[n_internal + j];
                rn.is_leaf                         = true;
                rn.value                           = leaf_table[j];
            }
            engine_.resident_finalize(table);
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
    // The host ops are static on the primary template; name it with a host
    // engine stand-in so the fallback arms reuse them without duplication.
    using HostStep = LevelStep<CpuHistogramEngine, SplitterT>;

    EngineT          &engine_;
    Dataset const    &ds_;
    TreeConfig const &config_;
    floats_view       grad_;
    floats_view       hess_;
    feature_view      selected_;
    bool              on_device_ = false;
};

} // namespace bonsai::grower_detail
