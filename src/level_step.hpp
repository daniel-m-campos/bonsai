#pragma once

// The grower's data plane (docs/architecture/12-grower-backend.md, decision
// 41). LevelStep groups the per-tree data-plane steps — root setup, split
// finding, row partitioning, child histogram construction, leaf finalize —
// behind one interface, selected by engine type: the primary template is the
// host plane (the CPU engine, and any engine's CPU fallback); the
// GPULevelEngine specialization is the device plane and holds the one runtime
// fork (on_device vs fallback) the design allows. The grow loops stay the
// control plane: every decision (leaf-vs-split, smaller-child pairing,
// constraint propagation) happens in grower_impl.hpp.

#include "bonsai/config/tree_config.hpp"
#include "bonsai/dataset.hpp"
#include "bonsai/detail/perf.hpp"
#include "bonsai/grower.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/parallel.hpp"
#include "bonsai/split.hpp"
#include "bonsai/tree.hpp"
#include "bonsai/types.hpp"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <print>
#include <span>
#include <utility>
#include <vector>

namespace bonsai::grower_detail
{

using feature_view = std::span<feature_id_t const>;

using detail::GrowProfiler; // definitions live in bonsai/detail/perf.hpp

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
    auto const last_bin = static_cast<bin_id_t>(ds.n_bins(s.feature_id) - 1);

    PendingSplit p;
    p.left.id  = left_id;
    p.right.id = right_id;
    ds.visit_bins(s.feature_id,
                  [&](auto bins)
                  {
                      auto goes_left = [&](row_id_t r)
                      {
                          bin_id_t const b = bins[r];
                          if (b == last_bin)
                          {
                              return s.default_left;
                          }
                          return b <= s.bin_id;
                      };
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
                  });
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

template <HistogramEngine EngineT>
inline void populate_nodes(Dataset const &ds, floats_view grad, floats_view hess,
                           split_input_refs nodes, feature_view selected,
                           EngineT &engine)
{
    // Engines that batch a level's fills (CPU row-wise units) take the whole
    // span; others fill node by node.
    if constexpr (requires { engine.populate_many(ds, grad, hess, nodes, selected); })
    {
        engine.populate_many(ds, grad, hess, nodes, selected);
        return;
    }
    else
    {
        for (SplitInput &node : nodes)
        {
            engine.populate(ds, grad, hess, node, selected);
        }
    }
}

// The data plane for a single node — the leafwise grower's unit of work: its
// gain heap expands one node at a time, so there is no level to batch and no
// LevelPlan; the same partition/populate/subtract primitives compose directly.
template <HistogramEngine EngineT>
inline std::pair<SplitInput, SplitInput>
split_node(Dataset const &ds, floats_view grad, floats_view hess, SplitInput parent,
           SplitOutput const &s, node_id_t left_id, node_id_t right_id,
           feature_view selected, EngineT &engine)
{
    PendingSplit p = partition_rows(ds, std::move(parent), s, left_id, right_id);
    engine.populate(ds, grad, hess, smaller_child(p), selected);
    finish_split(ds, p);
    return {std::move(p.left), std::move(p.right)};
}

// One split node's deferred work, produced by the control plane (plan_level)
// and executed by the LevelStep: partition fills p's children, build_children
// fills their histograms (host) or slots (device).
struct DeferredSplit
{
    SplitInput                parent;
    PendingSplit              p;
    SplitOutput               split;
    node_id_t                 left_id;
    node_id_t                 right_id;
    double                    parent_lo;
    double                    parent_hi;
    std::vector<feature_id_t> parent_path;
    uint32_t                  parent_slot = 0; // index in the frontier
    HistCell                  left_sums{};     // from find (device mode only)
    HistCell                  right_sums{};
};

// A frontier node the control plane finalized as a leaf this level; the
// device plane stamps its row segment before the level advances past it.
struct SlotLeaf
{
    uint32_t  slot;
    node_id_t node_id;
};

// One level's planned work: what splits, what stays a leaf.
struct LevelPlan
{
    std::vector<DeferredSplit> splits;
    std::vector<SlotLeaf>      leaves;
};

// ---------------------------------------------------------------------------
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
        GrowProfiler::Lap lap;
        SplitInput        root;
        root.id = 0;
        root.rows.assign(row_indices.begin(), row_indices.end());
        engine_.populate(ds_, grad_, hess_, root, selected_);
        root.sums      = root.totals();
        root.row_count = root.rows.size();
        lap(GrowProfiler::instance().populate_s);
        return root;
    }

    // Per-node splitter, or one level-wide find broadcast to every node when
    // the splitter is level-granular (the oblivious growth shape).
    void find_splits(std::vector<SplitInput> const &current,
                     std::vector<SplitOutput> &out, std::vector<HistCell> &child_sums)
    {
        GrowProfiler::Lap lap;
        host_find<SplitterT>(current, config_, out, child_sums);
        lap(GrowProfiler::instance().find_s);
    }

    // Routes every split parent's rows into its children, one node per worker
    // (each partition touches only its own parent's rows: bit-identical to
    // serial at any thread count).
    void partition(LevelPlan &plan)
    {
        GrowProfiler::Lap lap;
        host_partition(ds_, plan);
        lap(GrowProfiler::instance().partition_s);
    }

    // Fills every smaller child's histograms in one engine call; the larger
    // sibling derives by subtraction.
    void build_children(LevelPlan &plan)
    {
        GrowProfiler::Lap lap;
        host_build_children(engine_, ds_, grad_, hess_, selected_, plan);
        lap(GrowProfiler::instance().populate_s);
    }

    // End of tree: the surviving frontier becomes leaves (values and row ids
    // stamped on the host).
    void finalize(std::vector<SplitInput> const &current, DenseTree::Nodes &nodes,
                  size_t &n_leaves, train_leaf_values &values,
                  std::vector<node_id_t> &leaf_ids, row_index_view /*row_indices*/)
    {
        for (auto const &input : current)
        {
            finalize_as_leaf(nodes, input, config_, n_leaves, values, leaf_ids);
        }
    }

    // Oblivious leaf finalize, host plane: each frontier node is a leaf,
    // indexed by position into leaf_table; stamp its rows directly.
    void finalize_leaves(std::vector<SplitInput> const &frontier,
                         std::vector<float> const      &leaf_table,
                         train_leaf_values &values, std::vector<node_id_t> &leaf_ids,
                         row_index_view /*row_indices*/)
    {
        host_finalize_leaves(frontier, leaf_table, values, leaf_ids);
    }

    static void host_finalize_leaves(std::vector<SplitInput> const &frontier,
                                     std::vector<float> const      &leaf_table,
                                     train_leaf_values             &values,
                                     std::vector<node_id_t>        &leaf_ids)
    {
        for (size_t li = 0; li < frontier.size(); ++li)
        {
            for (row_id_t const r : frontier[li].rows)
            {
                values[r]   = leaf_table[li];
                leaf_ids[r] = static_cast<node_id_t>(li);
            }
        }
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
            out.reserve(current.size());
            for (auto const &input : current)
            {
                out.push_back(S::find(input, config));
            }
        }
    }

    // Level-wide blocked partition: nodes decompose into fixed-size row
    // blocks so one huge parent (the root) and many small deep nodes both
    // fill every worker. Per block, count goes-left; a serial scan turns
    // counts into stable scatter offsets; blocks then scatter concurrently.
    // The output is the exact stable order of a serial pass — bit-identical
    // at any thread count (integers only, no reductions).
    static void host_partition(Dataset const &ds, LevelPlan &plan)
    {
        struct Block
        {
            size_t split_idx, k0, k1;
            size_t n_left = 0, left0 = 0, right0 = 0;
        };
        constexpr size_t                       block_rows = 65536;
        static thread_local std::vector<Block> blocks;
        blocks.clear();
        for (size_t i = 0; i < plan.splits.size(); ++i)
        {
            size_t const n = plan.splits[i].parent.rows.size();
            for (size_t k0 = 0; k0 < n; k0 += block_rows)
            {
                blocks.push_back({i, k0, std::min(k0 + block_rows, n)});
            }
        }
        // Capture raw pointers: naming a thread_local inside the parallel
        // regions would resolve to each worker's own (empty) vector.
        Block *const         blk          = blocks.data();
        DeferredSplit *const splits       = plan.splits.data();
        auto const           goes_left_of = [&ds](DeferredSplit const &d)
        {
            auto const last_bin =
                static_cast<bin_id_t>(ds.n_bins(d.split.feature_id) - 1);
            return [&d, last_bin](bin_id_t b)
            { return b == last_bin ? d.split.default_left : b <= d.split.bin_id; };
        };
        parallel::for_each_index(
            blocks.size(),
            [&, blk, splits](size_t u)
            {
                Block               &b         = blk[u];
                DeferredSplit const &d         = splits[b.split_idx];
                auto const           goes_left = goes_left_of(d);
                row_id_t const      *rows      = d.parent.rows.data();
                ds.visit_bins(d.split.feature_id,
                              [&](auto bins)
                              {
                                  size_t n_left = 0;
                                  for (size_t k = b.k0; k < b.k1; ++k)
                                  {
                                      n_left += goes_left(bins[rows[k]]) ? 1 : 0;
                                  }
                                  b.n_left = n_left;
                              });
            });
        size_t prev_split = static_cast<size_t>(-1);
        size_t li         = 0;
        size_t ri         = 0;
        for (Block &b : blocks)
        {
            if (b.split_idx != prev_split)
            {
                if (prev_split != static_cast<size_t>(-1))
                {
                    finish_sizes(splits[prev_split], li, ri);
                }
                prev_split = b.split_idx;
                li         = 0;
                ri         = 0;
            }
            b.left0  = li;
            b.right0 = ri;
            li += b.n_left;
            ri += (b.k1 - b.k0) - b.n_left;
        }
        if (prev_split != static_cast<size_t>(-1))
        {
            finish_sizes(splits[prev_split], li, ri);
        }
        parallel::for_each_index(
            blocks.size(),
            [&, blk, splits](size_t u)
            {
                Block const    &b         = blk[u];
                DeferredSplit  &d         = splits[b.split_idx];
                auto const      goes_left = goes_left_of(d);
                row_id_t const *rows      = d.parent.rows.data();
                row_id_t *const left      = d.p.left.rows.data() + b.left0;
                row_id_t *const right     = d.p.right.rows.data() + b.right0;
                ds.visit_bins(d.split.feature_id,
                              [&](auto bins)
                              {
                                  size_t li2 = 0;
                                  size_t ri2 = 0;
                                  for (size_t k = b.k0; k < b.k1; ++k)
                                  {
                                      row_id_t const r = rows[k];
                                      if (goes_left(bins[r]))
                                      {
                                          left[li2++] = r;
                                      }
                                      else
                                      {
                                          right[ri2++] = r;
                                      }
                                  }
                              });
            });
        for (DeferredSplit &d : plan.splits)
        {
            d.p.left.id      = d.left_id;
            d.p.right.id     = d.right_id;
            d.p.parent_hists = std::move(d.parent.hists);
            d.parent.rows.clear();
            d.parent.rows.shrink_to_fit();
        }
    }

    static void finish_sizes(DeferredSplit &d, size_t n_left, size_t n_right)
    {
        d.p.left.rows.resize(n_left);
        d.p.right.rows.resize(n_right);
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

// ---------------------------------------------------------------------------
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
        SplitInput root;
        root.id = 0;
        root.rows.assign(row_indices.begin(), row_indices.end());
        on_device_ = engine_.begin_root(ds_, grad_, hess_, root, selected_);
        if (on_device_)
        {
            return root; // hists/rows stay device-resident; root carries sums
        }
        engine_.populate(ds_, grad_, hess_, root, selected_);
        root.sums      = root.totals();
        root.row_count = root.rows.size();
        return root;
    }

    void find_splits(std::vector<SplitInput> const &current,
                     std::vector<SplitOutput> &out, std::vector<HistCell> &child_sums)
    {
        GrowProfiler::Lap lap;
        if (on_device_)
        {
            out.clear();
            child_sums.clear();
            out.resize(current.size());
            child_sums.resize(2 * current.size());
            // Oblivious (LevelSplitFinder) picks one split for the whole
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
        lap(GrowProfiler::instance().find_s);
    }

    // Leaves stamp their device segments, splits partition on the device
    // (only child counts return), and each child's ids/sums/row_count fill in
    // from the counts — SplitInput degrades to node metadata on this plane.
    void partition(LevelPlan &plan)
    {
        GrowProfiler::Lap lap;
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
            HostStep::host_partition(ds_, plan);
        }
        lap(GrowProfiler::instance().partition_s);
    }

    // Smaller children build from their device row segments; the larger
    // derive on-device as parent minus smaller, then the child level becomes
    // current.
    void build_children(LevelPlan &plan)
    {
        GrowProfiler::Lap lap;
        if (on_device_)
        {
            std::vector<typename EngineT::LevelOp> ops;
            ops.reserve(plan.splits.size());
            for (uint32_t k = 0; k < plan.splits.size(); ++k)
            {
                DeferredSplit const &d = plan.splits[k];
                bool const left_small  = d.p.left.row_count <= d.p.right.row_count;
                ops.push_back({d.parent_slot, (2 * k) + (left_small ? 0U : 1U),
                               (2 * k) + (left_small ? 1U : 0U)});
            }
            engine_.advance_level(ds_, ops);
        }
        else
        {
            HostStep::host_build_children(engine_, ds_, grad_, hess_, selected_, plan);
        }
        lap(GrowProfiler::instance().populate_s);
    }

    // End of tree: stamp the surviving frontier's device segments, then pull
    // the per-row leaf assignment home once and stamp values/leaf_ids from it
    // (host finalize_as_leaf writes the nodes; its row loop no-ops on empty
    // device-mode rows).
    void finalize(std::vector<SplitInput> const &current, DenseTree::Nodes &nodes,
                  size_t &n_leaves, train_leaf_values &values,
                  std::vector<node_id_t> &leaf_ids, row_index_view row_indices)
    {
        if (on_device_)
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
        if (on_device_)
        {
            std::vector<node_id_t> by_row(ds_.n_rows(), 0);
            engine_.finalize_rows(by_row);
            for (row_id_t const r : row_indices)
            {
                leaf_ids[r] = by_row[r];
                values[r]   = nodes[by_row[r]].threshold_or_value;
            }
            // BONSAI_EXP_DEVICE_GRAD probe (decision 52 phase A): hand the
            // per-node leaf values to the engine's fused device update.
            static bool const exp_device_grad =
                std::getenv("BONSAI_EXP_DEVICE_GRAD") != nullptr;
            if constexpr (requires {
                              engine_.exp_end_tree(ds_, std::span<float const>{});
                          })
            {
                if (exp_device_grad)
                {
                    std::vector<float> node_vals(nodes.size(), 0.0F);
                    for (size_t i = 0; i < nodes.size(); ++i)
                    {
                        node_vals[i] = nodes[i].threshold_or_value;
                    }
                    engine_.exp_end_tree(ds_, node_vals);
                }
            }
        }
    }

    // Oblivious leaf finalize: the frontier nodes are the leaves, indexed by
    // position into leaf_table. On device the rows are resident, so stamp each
    // final slot with its leaf index and download the per-row assignment; in
    // fallback mode the rows live on the host and must be stamped here — the
    // early return that assumed otherwise trained silent garbage (issue #12).
    void finalize_leaves(std::vector<SplitInput> const &frontier,
                         std::vector<float> const      &leaf_table,
                         train_leaf_values &values, std::vector<node_id_t> &leaf_ids,
                         row_index_view row_indices)
    {
        if (!on_device_)
        {
            HostStep::host_finalize_leaves(frontier, leaf_table, values, leaf_ids);
            return;
        }
        std::vector<typename EngineT::LeafStamp> stamps;
        stamps.reserve(frontier.size());
        for (uint32_t i = 0; i < frontier.size(); ++i)
        {
            stamps.push_back({i, static_cast<node_id_t>(i)});
        }
        engine_.stamp_leaves(stamps);
        std::vector<node_id_t> by_row(ds_.n_rows(), 0);
        engine_.finalize_rows(by_row);
        for (row_id_t const r : row_indices)
        {
            leaf_ids[r] = by_row[r];
            values[r]   = leaf_table[by_row[r]];
        }
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
