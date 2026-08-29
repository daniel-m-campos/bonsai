#pragma once

#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "bonsai/types.hpp"

namespace bonsai
{

class DenseWalk;
class ObliviousWalk;

// The one NaN-routing rule, shared by every tree walk and both predict
// packs: v > t and !(v <= t) are the same predicate on non-NaN values and
// disagree exactly on NaN (the first sends it left, the second right), so
// default_left selects the comparison form and no isnan test is needed.
// Equivalence requires IEEE comparison semantics: -ffinite-math-only would
// let a compiler collapse the two forms and silently break NaN routing
// (no such flag is set; -ffp-contract=off is, see the top-level CMake).
inline bool routes_right(float v, float threshold, bool default_left)
{
    return default_left ? v > threshold : !(v <= threshold);
}

template <typename T>
concept Tree = requires(T const t, features_view X, floats_out out, row_id_t i) {
    { t.params() } -> std::same_as<typename T::Params const &>;
    { t.predict(X, out) } -> std::same_as<void>;
    { t.value_for(X, i) } -> std::same_as<float>;
};

// Heterogeneous (depth-wise) tree stored as a flat array of nodes.
//
// A single Node struct represents both internal and leaf nodes:
//   - Internal: feature_id < k_leaf_flag; threshold_or_value is the split
//     threshold; left/right are child ids; default_left routes NaN.
//   - Leaf:     feature_id == k_leaf_flag; threshold_or_value is the leaf
//     contribution; left/right/default_left are unused.
//
// 20-byte node (vs the prior 24-byte std::variant<InternalNode, LeafNode>).
// The smaller footprint and the absence of a variant tag check shrink the
// predict-path hot loop.
class DenseTree
{
  public:
    static constexpr feature_id_t k_leaf_flag =
        std::numeric_limits<feature_id_t>::max();

    struct Node
    {
        feature_id_t feature_id         = k_leaf_flag;
        float        threshold_or_value = 0.0F;
        node_id_t    left               = 0;
        node_id_t    right              = 0;
        bool         default_left       = false;
    };

    using Nodes     = std::vector<Node>;
    using walk_type = DenseWalk;

    struct Params
    {
        size_t depth{};
        size_t n_leaves{};
    };

    // split_gains: per-node split gain, indexed by node id (0 for leaves).
    // covers: per-node training row count. Both optional so hand-built test
    // trees stay terse; empty means "unknown" (importance reports 0, SHAP
    // refuses).
    DenseTree(Nodes nodes, Params params, std::vector<float> split_gains = {},
              std::vector<float> covers = {});

    static Node leaf(float value)
    {
        return Node{.feature_id = k_leaf_flag, .threshold_or_value = value};
    }

    static Node internal(feature_id_t feature_id, float threshold, node_id_t left,
                         node_id_t right, bool default_left)
    {
        return Node{.feature_id         = feature_id,
                    .threshold_or_value = threshold,
                    .left               = left,
                    .right              = right,
                    .default_left       = default_left};
    }

    static bool is_leaf(Node const &n)
    {
        return n.feature_id == k_leaf_flag;
    }

    // DART normalization: multiply every leaf contribution by `factor`.
    void scale_leaves(float factor)
    {
        for (auto &n : nodes_)
        {
            if (is_leaf(n))
            {
                n.threshold_or_value *= factor;
            }
        }
    }

    // Leaf renewal: overwrite one leaf's contribution in place.
    void set_leaf_value(node_id_t id, float value)
    {
        assert(is_leaf(nodes_[id]));
        nodes_[id].threshold_or_value = value;
    }

    // Accumulates into out; caller initializes (e.g. to zero or to a bias).
    void predict(features_view X, floats_out out) const;

    // The leaf node id row i of X lands in (pred_leaf support).
    node_id_t leaf_for(features_view X, row_id_t i) const;

    // Row i's leaf contribution; predict's per-row body, exposed so a caller
    // that scales it can skip predict's accumulate buffer.
    float value_for(features_view X, row_id_t i) const;

    Params const &params() const
    {
        return params_;
    }

    Nodes const &nodes() const
    {
        return nodes_;
    }

    std::vector<float> const &split_gains() const
    {
        return split_gains_;
    }

    std::vector<float> const &covers() const
    {
        return covers_;
    }

  private:
    Nodes              nodes_;
    Params             params_;
    std::vector<float> split_gains_;
    std::vector<float> covers_;
};

class ObliviousTree
{
  public:
    struct LevelSplit
    {
        feature_id_t feature_id;
        float        threshold;
        bool         default_left;
    };

    using LevelSplits = std::vector<LevelSplit>;
    using LeafTable   = std::vector<float>;
    using walk_type   = ObliviousWalk;

    struct Params
    {
        size_t depth{};
        size_t n_leaves{};
    };

    // level_gains: split gain per level; leaf_covers: training rows per
    // leaf slot (2^depth entries). Both optional, empty = unknown (see
    // DenseTree) — leaf covers exist so TreeSHAP has its background
    // distribution; models saved before they were recorded load with
    // covers empty and pred_contribs explains why it can't run.
    ObliviousTree(LevelSplits splits, LeafTable values,
                  std::vector<float> level_gains = {},
                  std::vector<float> leaf_covers = {});

    // DART normalization: multiply every leaf contribution by `factor`.
    void scale_leaves(float factor)
    {
        for (auto &v : leaf_table_)
        {
            v *= factor;
        }
    }

    // Leaf renewal: overwrite one leaf-table entry in place.
    void set_leaf_value(size_t index, float value)
    {
        leaf_table_[index] = value;
    }

    // Accumulates into out; caller initializes (e.g. to zero or to a bias).
    void predict(features_view X, floats_out out) const;

    // The leaf-table index row i of X lands in (pred_leaf support).
    node_id_t leaf_for(features_view X, row_id_t i) const;

    // Row i's leaf contribution; predict's per-row body, exposed so a caller
    // that scales it can skip predict's accumulate buffer.
    float value_for(features_view X, row_id_t i) const;

    Params const &params() const
    {
        return params_;
    }

    LevelSplits const &splits() const
    {
        return splits_;
    }

    LeafTable const &leaf_table() const
    {
        return leaf_table_;
    }

    std::vector<float> const &level_gains() const
    {
        return level_gains_;
    }

    std::vector<float> const &leaf_covers() const
    {
        return leaf_covers_;
    }

  private:
    LevelSplits        splits_;
    LeafTable          leaf_table_;
    std::vector<float> leaf_covers_;
    Params             params_;
    std::vector<float> level_gains_;
};

// SHAP support: an oblivious tree is a perfect binary tree with one split
// broadcast per level; expand it into the DenseTree shape (internal covers
// aggregated bottom-up from the leaf covers) so TreeSHAP's cover-weighted
// walk applies unchanged. Throws std::invalid_argument without leaf covers.
DenseTree dense_equivalent(ObliviousTree const &tree);

// The dense-tree predict pack, for the depthwise and leafwise ensembles.
// Rows-outer inversion alone buys a dense tree nothing, because its cost is
// the chain of data-dependent node loads, not the loop shape (measured 2231
// vs 2206 ns/row); the lever is walking eight trees per row in lockstep so
// the core always has independent load chains in flight. Leaves point to
// themselves and a leaf's threshold slot holds its value, so a walk padded
// to its group's deepest tree self-loops harmlessly and the sum reads the
// node the walk ends on; the padding tax is bounded by the group's deepest
// member (a shallow tree still steps that many times), and groups bound
// their own depth so leafwise's ragged best-first trees tax only their own
// group.
//
// What it guarantees: accumulate adds each row's tree-order value sum into
// out, bit-identical to calling tree.predict per tree (the depthwise eval
// baseline pin never moves), NaN routing per routes_right included.
//
// What breaks it: the pack snapshots the trees it was built from; a booster
// mutation invalidates it, which the booster's epoch-keyed cache enforces.
// Nothing else may hold one across a mutation. Tree depth is trusted from
// Params::depth, which every construction site stamps from the grown
// structure.
//
// perf: measured on M2 (1 thread, 64 cols, depth 8, 100 trees) 2206 ->
// 1004 ns/row at batch against xgboost inplace_predict at 1015.
//
// Pinned by tests/unit/test_predict_walk.cpp.
class DenseWalk
{
  public:
    explicit DenseWalk(std::span<DenseTree const> trees);

    // Adds the first n_trees trees' values to out, row-parallel above a
    // small-batch floor and serial below it so single-row callers never
    // pay a parallel-region entry.
    void accumulate(features_view X, size_t n_trees, floats_out out) const;

  private:
    struct PackedNode
    {
        float        threshold;
        feature_id_t feature;
        uint32_t     left;
        uint32_t     right;
    };

    std::vector<PackedNode> nodes_;
    std::vector<uint8_t>    default_left_;
    std::vector<uint32_t>   roots_;
    std::vector<uint32_t>   group_depth_;
};

// The oblivious ensemble's predict pack: every tree's level splits packed
// contiguous and every leaf table concatenated, so accumulate() inverts the
// ensemble loop (rows outer in one parallel section, trees inner over
// cache-resident arrays) instead of opening a parallel region per tree over
// each tree's own split vector.
//
// Full 64-row blocks additionally walk transposed: the block is flipped
// into feature-major scratch once, so every level step is one broadcast
// compare over 64 contiguous floats, which vectorizes at baseline SIMD
// widths with no gathers. Row-wise blocking is the one vectorization the
// bit-identity contract permits: each row's accumulator remains its own
// tree-order fold, the lanes just answer each level's question together.
// The sub-block tail walks the scalar path.
//
// What it guarantees: accumulate adds each row's tree-order leaf sum into
// out, with the same per-row float fold as calling tree.predict per tree,
// so results are bit-identical to the per-tree walk, NaN routing per
// routes_right included. Trees of unequal depth pack per-tree ranges, so
// early-stopped levelwise models walk unchanged.
//
// What breaks it: same staleness contract as DenseWalk above; the
// booster's epoch-keyed cache is the only legitimate holder across a
// mutation.
//
// perf: measured at 1 thread (64 cols, depth 8, 100 trees): single-row
// 1053 -> 610ns on M2, and the blocked batch walk reaches 181 ns/row on
// M2 and 179 on EPYC 9654 (from 505 and 874 scalar), against catboost's
// same-pod 784. The two hosts converging says the block walk is memory
// bound, not ISA bound. A scalar baked-constant codegen probe measured
// 293, so row blocking beats that ceiling along the dimension the probe
// did not price, and this is still not a code generator.
//
// Pinned by tests/unit/test_predict_walk.cpp (exact-equality parity vs the
// per-tree walk, NaN injection included).
class ObliviousWalk
{
  public:
    explicit ObliviousWalk(std::span<ObliviousTree const> trees);

    // Adds the first n_trees trees' leaf values to out, row-parallel above
    // a small-batch floor and serial below it so single-row callers never
    // pay a parallel-region entry.
    void accumulate(features_view X, size_t n_trees, floats_out out) const;

  private:
    void walk_rows(features_view X, size_t n_trees, size_t row0, size_t count,
                   floats_out out) const;
    void walk_block(features_view X, size_t n_trees, size_t row0, float *scratch,
                    floats_out out) const;

    std::vector<feature_id_t> feat_;
    std::vector<float>        thr_;
    std::vector<uint8_t>      default_left_;
    std::vector<uint32_t>     split_off_;
    std::vector<uint32_t>     leaf_off_;
    std::vector<float>        leaf_;
};

} // namespace bonsai
