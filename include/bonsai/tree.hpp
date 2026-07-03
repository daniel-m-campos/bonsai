#pragma once

#include <cassert>
#include <concepts>
#include <cstddef>
#include <limits>
#include <vector>

#include "bonsai/types.hpp"

namespace bonsai
{

template <typename T>
concept Tree = requires(T const t, features_view X, floats_out out) {
    { t.params() } -> std::same_as<typename T::Params const &>;
    { t.predict(X, out) } -> std::same_as<void>;
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

    using Nodes = std::vector<Node>;

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
    float walk_row(features_view X, row_id_t i) const;

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

    struct Params
    {
        size_t depth{};
        size_t n_leaves{};
    };

    // level_gains: split gain per level; empty = unknown (see DenseTree).
    ObliviousTree(LevelSplits splits, LeafTable values,
                  std::vector<float> level_gains = {});

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

  private:
    float walk_row(features_view X, row_id_t i) const;

    LevelSplits        splits_;
    LeafTable          leaf_table_;
    Params             params_;
    std::vector<float> level_gains_;
};

} // namespace bonsai
