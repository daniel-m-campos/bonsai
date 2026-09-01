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
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

namespace bonsai::grower_detail
{

using feature_view = std::span<feature_id_t const>;

using detail::GrowProfiler;
using detail::Phase;

inline float write_leaf(DenseTree::Nodes &nodes, SplitInput const &node,
                        TreeConfig const &config, size_t &n_leaves)
{
    auto const v   = static_cast<float>(bounded_leaf_weight(
        node.total_grad(), node.total_hess(), config, node.lo, node.hi));
    nodes[node.id] = DenseTree::leaf(v);
    ++n_leaves;
    return v;
}

inline void finalize_as_leaf(DenseTree::Nodes &nodes, SplitInput const &node,
                             TreeConfig const &config, size_t &n_leaves,
                             train_leaf_values      &values,
                             std::vector<node_id_t> &leaf_ids)
{
    float const v = write_leaf(nodes, node, config, n_leaves);
    // perf: Row-parallel: each row is written exactly once with the same value,
    // so the order is immaterial (byte-identical at any thread count). The
    // stamping loops were ~17s of the 16M CPU fit on dual-EPYC hosts:
    // scattered writes that want bandwidth.
    parallel::for_each_index(node.rows.size(),
                             [&](size_t k)
                             {
                                 row_id_t const r = node.rows[k];
                                 values[r]        = v;
                                 leaf_ids[r]      = node.id;
                             });
}

template <typename R>
inline void stamp_leaf_rows(DenseTree::Nodes const &nodes, R &&frontier,
                            train_leaf_values &values, std::vector<node_id_t> &leaf_ids)
{
    parallel::for_each_index(std::ranges::size(frontier),
                             [&](size_t li)
                             {
                                 SplitInput const &input = frontier[li];
                                 float const v = nodes[input.id].threshold_or_value;
                                 for (row_id_t const r : input.rows)
                                 {
                                     values[r]   = v;
                                     leaf_ids[r] = input.id;
                                 }
                             });
}

struct PendingSplit
{
    SplitInput     left;
    SplitInput     right;
    NodeHistograms parent_hists;
};

inline SplitInput &smaller_child(PendingSplit &p)
{
    return p.left.rows.size() <= p.right.rows.size() ? p.left : p.right;
}

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
    return p;
}

inline void finish_split(Dataset const &ds, PendingSplit &p, bool fused = false)
{
    SplitInput &small = smaller_child(p);
    SplitInput &large = &small == &p.left ? p.right : p.left;
    large.hists       = std::move(p.parent_hists);
    if (!fused)
    {
        parallel::for_each_index(ds.n_features(),
                                 [&](size_t f) { large.hists[f] -= small.hists[f]; });
    }
    small.sums      = small.totals();
    large.sums      = large.totals();
    small.row_count = small.rows.size();
    large.row_count = large.rows.size();
}

template <HistogramEngine EngineT>
inline void populate_nodes(Dataset const &ds, floats_view grad, floats_view hess,
                           split_input_refs nodes, feature_view selected,
                           EngineT &engine)
{
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

template <HistogramEngine EngineT>
inline SplitInput populated_root(Dataset const &ds, floats_view grad, floats_view hess,
                                 RowSelection const &sel, feature_view selected,
                                 EngineT &engine)
{
    SplitInput root;
    root.rows.assign(sel.rows.begin(), sel.rows.end());
    assert(sel.shape.runs.empty() || rows_in(sel.shape.runs) == sel.rows.size());
    root.shape = sel.shape;
    engine.populate(ds, grad, hess, root, selected);
    root.sums      = root.totals();
    root.row_count = root.rows.size();
    return root;
}

// perf: Full-data fits pass the identity by contract (empty rows +
// row_count): the 64MB host copy and its upload never happen; the
// engine builds/caches the identity on device. Identity, not
// cardinality: any other full-length list takes the general path.
inline SplitInput device_root(RowSelection const &sel)
{
    SplitInput root;
    if (sel.shape.identity)
    {
        root.row_count = sel.rows.size();
    }
    else
    {
        root.rows.assign(sel.rows.begin(), sel.rows.end());
    }
    return root;
}

inline void release_staged_rows(SplitInput &root)
{
    root.rows = {};
}

template <typename StampT, typename IdOfFn>
inline std::vector<StampT> leaf_stamps(size_t n, IdOfFn &&id_of)
{
    std::vector<StampT> stamps;
    stamps.reserve(n);
    for (uint32_t i = 0; i < n; ++i)
    {
        stamps.push_back({i, id_of(i)});
    }
    return stamps;
}

inline std::vector<float> node_values(DenseTree::Nodes const &nodes)
{
    std::vector<float> out(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i)
    {
        out[i] = nodes[i].threshold_or_value;
    }
    return out;
}

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
    uint32_t                  parent_slot = 0;
    HistCell                  left_sums{};
    HistCell                  right_sums{};
};

struct SlotLeaf
{
    uint32_t  slot;
    node_id_t node_id;
};

struct LevelPlan
{
    std::vector<DeferredSplit> splits;
    std::vector<SlotLeaf>      leaves;
};

inline constexpr size_t partition_block_rows = 65536;

// perf: Rows one partition worker must own to earn its share of the region entry:
// below it the entry costs more than the scan it splits (measured).
inline constexpr size_t partition_rows_per_worker = 4096;

// perf: Blocks per worker: an equal cut strands the slowest core with a whole
// share, so every worker takes several blocks and the dynamic schedule
// evens the rest out (measured on asymmetric cores).
inline constexpr size_t partition_blocks_per_worker = 4;

// perf: Workers one parent's partition earns: the whole configured team, or none
// of it. Every other region in a fit runs on the configured team, so a
// partition region of a different size between them rebuilds the team the
// runtime keeps hot, and the fills around it pay more than the partition
// saves (measured at 12 threads). A parent that cannot give every
// worker its floor of rows takes the serial scan instead.
inline int partition_workers(size_t n_rows, int threads)
{
    return n_rows / partition_rows_per_worker >= static_cast<size_t>(threads) ? threads
                                                                              : 1;
}

inline void host_partition(Dataset const &ds, LevelPlan &plan,
                           size_t block_rows = partition_block_rows,
                           int    workers    = parallel::n_threads())
{
    struct Block
    {
        size_t split_idx, k0, k1;
        size_t n_left = 0, left0 = 0, right0 = 0;
    };
    static thread_local std::vector<Block>  blocks;
    static thread_local std::vector<size_t> split_b0;
    blocks.clear();
    split_b0.assign(plan.splits.size() + 1, 0);
    for (size_t i = 0; i < plan.splits.size(); ++i)
    {
        split_b0[i]    = blocks.size();
        size_t const n = plan.splits[i].parent.rows.size();
        for (size_t k0 = 0; k0 < n; k0 += block_rows)
        {
            blocks.push_back({i, k0, std::min(k0 + block_rows, n)});
        }
    }
    split_b0[plan.splits.size()]      = blocks.size();
    Block *const         blk          = blocks.data();
    DeferredSplit *const splits       = plan.splits.data();
    auto const           goes_left_of = [&ds](DeferredSplit const &d)
    {
        auto const last_bin = static_cast<bin_id_t>(ds.n_bins(d.split.feature_id) - 1);
        return [&d, last_bin](bin_id_t b)
        { return routes_left(b, last_bin, d.split.bin_id, d.split.default_left); };
    };
    parallel::for_each_index_on(
        workers, blocks.size(),
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
    size_t const *b0 = split_b0.data();
    parallel::for_each_index_on(workers, plan.splits.size(),
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
                                    splits[i].p.left.rows.resize(li);
                                    splits[i].p.right.rows.resize(ri);
                                });
    parallel::for_each_index_on(
        workers, blocks.size(),
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
    parallel::for_each_index_on(workers, plan.splits.size(),
                                [&, splits](size_t i)
                                {
                                    DeferredSplit &d = splits[i];
                                    d.p.left.id      = d.left_id;
                                    d.p.right.id     = d.right_id;
                                    d.p.parent_hists = std::move(d.parent.hists);
                                    d.parent.rows.clear();
                                    d.parent.rows.shrink_to_fit();
                                });
}

template <HistogramEngine EngineT>
inline std::pair<SplitInput, SplitInput>
split_node(Dataset const &ds, floats_view grad, floats_view hess, SplitInput parent,
           SplitOutput const &s, node_id_t left_id, node_id_t right_id,
           feature_view selected, EngineT &engine)
{
    GrowProfiler::Lap lap;
    PendingSplit      p;
    size_t const      n_rows  = parent.rows.size();
    int const         workers = partition_workers(n_rows, parallel::n_threads());
    if (workers > 1)
    {
        LevelPlan      plan;
        DeferredSplit &d = plan.splits.emplace_back();
        d.parent         = std::move(parent);
        d.split          = s;
        d.left_id        = left_id;
        d.right_id       = right_id;
        size_t const blocks =
            static_cast<size_t>(workers) * partition_blocks_per_worker;
        host_partition(ds, plan, (n_rows + blocks - 1) / blocks, workers);
        p = std::move(plan.splits.front().p);
    }
    else
    {
        p = partition_rows(ds, std::move(parent), s, left_id, right_id);
    }
    lap(GrowProfiler::instance().partition_s);
    SplitInput &small = smaller_child(p);
    bool        fused = false;
    if constexpr (requires {
                      engine.populate_lone(ds, grad, hess, small, selected,
                                           p.parent_hists);
                  })
    {
        fused = engine.populate_lone(ds, grad, hess, small, selected, p.parent_hists);
    }
    else
    {
        engine.populate(ds, grad, hess, small, selected);
    }
    finish_split(ds, p, fused);
    lap(GrowProfiler::instance().populate_s);
    return {std::move(p.left), std::move(p.right)};
}

struct Candidate
{
    SplitInput  node;
    SplitOutput split;
    uint8_t     depth = 0;
    uint32_t    slot  = 0;
    HistCell    left_sums{};
    HistCell    right_sums{};
};

struct ChildPair
{
    std::array<SplitInput, 2>  nodes;
    std::array<SplitOutput, 2> splits{};
    std::array<uint32_t, 2>    slots{};
    std::array<HistCell, 4>    child_sums{};
    uint8_t                    depth = 0;
};

inline bool is_empty_child(SplitInput const &child)
{
    return child.rows.empty() && child.row_count == 0;
}

inline uint32_t row_count_of(SplitInput const &node)
{
    return static_cast<uint32_t>(node.rows.empty() ? node.row_count : node.rows.size());
}

struct LevelOutputs
{
    std::vector<SplitOutput> splits;
    std::vector<HistCell>    child_sums;
};

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

struct LevelSplitBins
{
    feature_id_t feature_id;
    bin_id_t     split_bin;
    bool         default_left;
};

template <typename ResidentNodeT, typename SplitAtFn>
std::vector<ResidentNodeT> perfect_tree_table(size_t depth, SplitAtFn &&split_at,
                                              std::span<float const> leaf_table)
{
    size_t const               n_internal = (size_t{1} << depth) - 1;
    size_t const               n_leaves   = size_t{1} << depth;
    std::vector<ResidentNodeT> table(n_internal + n_leaves);
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
        LevelSplitBins const s  = split_at(lvl);
        ResidentNodeT       &rn = table[i];
        rn.feature_id           = s.feature_id;
        rn.split_bin            = s.split_bin;
        rn.default_left         = s.default_left;
        rn.left                 = static_cast<node_id_t>((2 * i) + 1);
        rn.right                = static_cast<node_id_t>((2 * i) + 2);
    }
    for (size_t j = 0; j < n_leaves; ++j)
    {
        ResidentNodeT &rn = table[n_internal + j];
        rn.is_leaf        = true;
        rn.value          = leaf_table[j];
    }
    return table;
}

template <typename ResidentNodeT>
std::vector<ResidentNodeT> oblivious_node_table(ObliviousTree const &tree,
                                                Dataset const       &ds)
{
    return perfect_tree_table<ResidentNodeT>(
        tree.splits().size(),
        [&](size_t lvl)
        {
            auto const &s = tree.splits()[lvl];
            return LevelSplitBins{s.feature_id,
                                  ds.bin_of_threshold(s.feature_id, s.threshold),
                                  s.default_left};
        },
        tree.leaf_table());
}

} // namespace bonsai::grower_detail
