#pragma once

// The grower's data plane (docs/architecture/12-grower-backend.md, decision
// 41; transaction vocabulary from docs/architecture/14-engine-narrative.md,
// decision 53). LevelStep groups the per-tree data-plane steps — root setup,
// open_level (split finding), apply_level (row partitioning), child
// histogram construction, end_tree (leaf finalize) — behind one interface,
// selected by engine type: the primary template is the
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
#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <print>
#include <ranges>
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
    // Row-parallel: each row is written exactly once with the same value,
    // so the order is immaterial (byte-identical at any thread count). The
    // stamping loops were ~17s of the 16M CPU fit on dual-EPYC hosts
    // (issue #46's decomposition) — scattered writes that want bandwidth.
    parallel::for_each_index(node.rows.size(),
                             [&](size_t k)
                             {
                                 row_id_t const r = node.rows[k];
                                 values[r]        = v;
                                 leaf_ids[r]      = node.id;
                             });
    ++n_leaves;
}

// A split with rows partitioned and histograms pending: the smaller child
// populates, then finish_split derives the larger by subtraction.
struct PendingSplit
{
    SplitInput             left;
    SplitInput             right;
    std::vector<Histogram> parent_hists;
    // The parent's arena backs parent_hists' views; it travels with them.
    std::vector<HistCell, detail::PoolAllocator<HistCell>> parent_arena;
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
    ds.visit_bins(
        s.feature_id,
        [&](auto bins)
        {
            auto goes_left = [&](row_id_t r)
            { return routes_left(bins[r], last_bin, s.bin_id, s.default_left); };
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
    p.parent_arena = std::move(parent.arena);
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
    large.arena = std::move(p.parent_arena);
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

// A leaf awaiting expansion: its histograms/rows, its best split (heap key),
// and its depth (to enforce the max_depth cap on children). On the device leaf
// plane the node's statistics are metadata only: slot names the pool slot
// holding its histogram, and the two sums are what its split implies for the
// children it would produce.
struct Candidate
{
    SplitInput  node;
    SplitOutput split;
    uint8_t     depth = 0;
    uint32_t    slot  = 0;
    HistCell    left_sums{};
    HistCell    right_sums{};
};

// One split's two children as the leaf plane hands them back. Contiguous by
// design: the device find takes the pair in a single launch, so the nodes must
// be adjacent and their outputs indexable by child.
struct ChildPair
{
    std::array<SplitInput, 2>  nodes;
    std::array<SplitOutput, 2> splits{};
    std::array<uint32_t, 2>    slots{};
    std::array<HistCell, 4>    child_sums{}; // (left, right) per child
    uint8_t                    depth = 0;
};

// A child with no rows. Host-plane children carry row lists; device-plane
// children carry counts only.
inline bool is_empty_child(SplitInput const &child)
{
    return child.rows.empty() && child.row_count == 0;
}

// A node's row count under either plane's contract: host-plane nodes carry
// rows and a zero row_count, device-plane nodes carry row_count and empty
// rows. The sanctioned reader alongside is_empty_child, replacing the
// repeated ternary at every covers[] site.
inline uint32_t row_count_of(SplitInput const &node)
{
    return static_cast<uint32_t>(node.rows.empty() ? node.row_count : node.rows.size());
}

// The level-transaction vocabulary (decision 53): the same narrative on
// both planes, with the backend an implementation detail. Step 1 of the
// migration introduces the types and speaks them from the grow loops;
// buffers stay caller-owned and reused across levels exactly as before.
struct LevelOutputs
{
    std::vector<SplitOutput> splits;
    // (left, right) sums per node from the find plane, 2 cells per node;
    // empty on the host plane, which derives child sums by subtraction.
    std::vector<HistCell> child_sums;
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
    // the splitter is level-granular (the levelwise growth shape).
    // Level transaction, phase 1: split decisions for the whole frontier
    // (decision 53). The frontier is the transaction's input; outputs are
    // caller-owned and reused across levels.
    void open_level(std::vector<SplitInput> const &frontier, LevelOutputs &out)
    {
        GrowProfiler::Lap lap;
        host_find<SplitterT>(frontier, config_, out.splits, out.child_sums);
        lap(GrowProfiler::instance().find_s);
    }

    // Routes every split parent's rows into its children, one node per worker
    // (each partition touches only its own parent's rows: bit-identical to
    // serial at any thread count).
    void apply_level(LevelPlan &plan)
    {
        GrowProfiler::Lap lap;
        host_partition(ds_, plan);
        lap(GrowProfiler::instance().partition_s);
    }

    // Fills every smaller child's histograms in one engine call; the larger
    // sibling derives by subtraction.
    void build_children(LevelPlan &plan, bool /*last*/ = false)
    {
        GrowProfiler::Lap lap;
        host_build_children(engine_, ds_, grad_, hess_, selected_, plan);
        lap(GrowProfiler::instance().populate_s);
    }

    // End of tree: the surviving frontier becomes leaves. Values and the
    // node table are serial (tiny); the row stamping runs one region for
    // the whole frontier, since per-leaf regions cost more than the writes.
    void end_tree(std::vector<SplitInput> const &current, DenseTree::Nodes &nodes,
                  size_t &n_leaves, train_leaf_values &values,
                  std::vector<node_id_t> &leaf_ids, row_index_view /*row_indices*/)
    {
        static thread_local std::vector<float> leaf_values;
        leaf_values.resize(current.size());
        for (size_t li = 0; li < current.size(); ++li)
        {
            auto const &input = current[li];
            auto const  v     = static_cast<float>(bounded_leaf_weight(
                input.total_grad(), input.total_hess(), config_, input.lo, input.hi));
            nodes[input.id]   = DenseTree::leaf(v);
            leaf_values[li]   = v;
            ++n_leaves;
        }
        float const *lv = leaf_values.data();
        parallel::for_each_index(current.size(),
                                 [&, lv](size_t li)
                                 {
                                     SplitInput const &input = current[li];
                                     float const       v     = lv[li];
                                     for (row_id_t const r : input.rows)
                                     {
                                         values[r]   = v;
                                         leaf_ids[r] = input.id;
                                     }
                                 });
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
            { return routes_left(b, last_bin, d.split.bin_id, d.split.default_left); };
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
        // Per-split prefix and child sizing runs one worker per split: the
        // sizing resize is a value-init of the children's row storage, and
        // serial it is the same Amdahl chunk the placeholder build was.
        static thread_local std::vector<size_t> split_b0;
        split_b0.assign(plan.splits.size() + 1, blocks.size());
        for (size_t u = blocks.size(); u-- > 0;)
        {
            split_b0[blk[u].split_idx] = u;
        }
        split_b0[plan.splits.size()] = blocks.size();
        for (size_t i = plan.splits.size(); i-- > 0;)
        {
            if (split_b0[i] == blocks.size())
            {
                split_b0[i] = split_b0[i + 1];
            }
        }
        size_t const *b0 = split_b0.data();
        parallel::for_each_index(plan.splits.size(),
                                 [&, blk, splits, b0](size_t i)
                                 {
                                     size_t li = 0;
                                     size_t ri = 0;
                                     for (size_t u = b0[i]; u < b0[i + 1]; ++u)
                                     {
                                         blk[u].left0  = li;
                                         blk[u].right0 = ri;
                                         li += blk[u].n_left;
                                         ri += (blk[u].k1 - blk[u].k0) - blk[u].n_left;
                                     }
                                     finish_sizes(splits[i], li, ri);
                                 });
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
        // Hand-off and parent-row release are independent per split; the
        // shrink's deallocation otherwise serializes on the orchestrator.
        parallel::for_each_index(plan.splits.size(),
                                 [&, splits](size_t i)
                                 {
                                     DeferredSplit &d = splits[i];
                                     d.p.left.id      = d.left_id;
                                     d.p.right.id     = d.right_id;
                                     d.p.parent_hists = std::move(d.parent.hists);
                                     d.p.parent_arena = std::move(d.parent.arena);
                                     d.parent.rows.clear();
                                     d.parent.rows.shrink_to_fit();
                                 });
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

// The finished dense tree flattened for the device-resident epilogue, which
// walks it in bin space: split_bin reconstructs exactly from the threshold
// (the grower set threshold = cuts[bin]). Shared by both device planes, whose
// trees are the same shape.
template <typename ResidentNodeT>
std::vector<ResidentNodeT> resident_node_table(DenseTree::Nodes const &nodes,
                                               Dataset const          &ds)
{
    std::vector<ResidentNodeT> table(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i)
    {
        DenseTree::Node const &nd = nodes[i];
        ResidentNodeT         &rn = table[i];
        if (DenseTree::is_leaf(nd))
        {
            rn.is_leaf = true;
            rn.value   = nd.threshold_or_value;
            continue;
        }
        rn.feature_id   = nd.feature_id;
        rn.left         = nd.left;
        rn.right        = nd.right;
        rn.default_left = nd.default_left;
        rn.split_bin    = ds.bin_of_threshold(nd.feature_id, nd.threshold_or_value);
    }
    return table;
}

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
        GrowProfiler::Lap lap;
        SplitInput        root;
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
        lap(GrowProfiler::instance().assign_s);
        on_device_ = engine_.begin_root(ds_, grad_, hess_, root, selected_);
        if (on_device_)
        {
            lap(GrowProfiler::instance().populate_s);
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
            lap(GrowProfiler::instance().assign_s);
        }
        engine_.populate(ds_, grad_, hess_, root, selected_);
        root.sums      = root.totals();
        root.row_count = root.rows.size();
        lap(GrowProfiler::instance().populate_s);
        return root;
    }

    void open_level(std::vector<SplitInput> const &frontier, LevelOutputs &lout)
    {
        auto             &out        = lout.splits;
        auto             &child_sums = lout.child_sums;
        auto const       &current    = frontier;
        GrowProfiler::Lap lap;
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
        lap(GrowProfiler::instance().find_s);
    }

    // Leaves stamp their device segments, splits partition on the device
    // (only child counts return), and each child's ids/sums/row_count fill in
    // from the counts — SplitInput degrades to node metadata on this plane.
    void apply_level(LevelPlan &plan)
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
    void build_children(LevelPlan &plan, bool last = false)
    {
        GrowProfiler::Lap lap;
        if (on_device_)
        {
            if (last)
            {
                // The last level's children are leaves: their histograms are
                // never read, so skip the build and keep the layout flip
                // stamping depends on (decision 71).
                engine_.advance_layout_only();
                lap(GrowProfiler::instance().populate_s);
                return;
            }
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

// ---------------------------------------------------------------------------
// Host data plane for best-first growth: the gain heap expands one node at a
// time, so there is no level to batch and no LevelPlan — the same
// partition/populate/subtract primitives compose one node deep.
// Branch-free: no GPU concept appears below.
template <HistogramEngine EngineT, typename SplitterT> class LeafStep
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
        SplitOutput const split = SplitterT::find(root, config);
        lap(GrowProfiler::instance().find_s);
        return {.node = std::move(root), .split = split, .depth = 0};
    }

    template <HistogramEngine E>
    static ChildPair host_split_children(E &engine, Dataset const &ds, floats_view grad,
                                         floats_view hess, feature_view selected,
                                         Candidate &c, node_id_t left_id,
                                         node_id_t right_id)
    {
        GrowProfiler::Lap lap;
        auto [left, right] = split_node(ds, grad, hess, std::move(c.node), c.split,
                                        left_id, right_id, selected, engine);
        lap(GrowProfiler::instance().partition_s);
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

    static void host_find_children(TreeConfig const &config, ChildPair &pair,
                                   bool may_split)
    {
        GrowProfiler::Lap lap;
        for (size_t i = 0; i < pair.nodes.size(); ++i)
        {
            pair.splits[i] =
                may_split ? SplitterT::find(pair.nodes[i], config) : SplitOutput{};
        }
        lap(GrowProfiler::instance().find_s);
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
template <GPULeafEngine EngineT, typename SplitterT> class LeafStep<EngineT, SplitterT>
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
        lap(GrowProfiler::instance().assign_s);
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
                lap(GrowProfiler::instance().assign_s);
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
        GrowProfiler::Lap                  lap;
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
        lap(GrowProfiler::instance().partition_s);
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
        // The smaller child built the fresh slot in split_children; tie goes
        // to the left child, matching leaf_split's device-side choice
        // (CudaDeviceContext::leaf_split: left_small = left_count <= right_count).
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
