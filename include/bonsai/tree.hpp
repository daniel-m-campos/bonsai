#pragma once

#include <concepts>
#include <cstddef>
#include <variant>
#include <vector>

#include "bonsai/types.hpp"

namespace bonsai
{

template <typename T>
concept Tree = requires(T const t, features_view X, floats_out out) {
    { t.params() } -> std::same_as<typename T::Params const &>;
    { t.predict(X, out) } -> std::same_as<void>;
};

class DenseTree
{
  public:
    struct InternalNode
    {
        feature_id_t feature_id;
        float threshold;
        node_id_t left;
        node_id_t right;
        bool default_left;
    };

    struct LeafNode
    {
        float value;
    };

    using Node  = std::variant<InternalNode, LeafNode>;
    using Nodes = std::vector<Node>;

    struct Params
    {
        size_t depth{};
        size_t n_leaves{};
    };

    DenseTree(Nodes nodes, Params params);

    // Accumulates into out; caller initializes (e.g. to zero or to a bias).
    void predict(features_view X, floats_out out) const;

    Params const &params() const
    {
        return params_;
    }

    Nodes const &nodes() const
    {
        return nodes_;
    }

  private:
    float walk_row(features_view X, row_id_t i) const;

    Nodes nodes_;
    Params params_;
};

class ObliviousTree
{
  public:
    struct LevelSplit
    {
        feature_id_t feature_id;
        float threshold;
        bool default_left;
    };

    using LevelSplits = std::vector<LevelSplit>;
    using LeafValues  = std::vector<float>;

    struct Params
    {
        size_t depth{};
        size_t n_leaves{};
    };

    ObliviousTree(LevelSplits splits, LeafValues values);

    // Accumulates into out; caller initializes (e.g. to zero or to a bias).
    void predict(features_view X, floats_out out) const;

    Params const &params() const
    {
        return params_;
    }

  private:
    float walk_row(features_view X, row_id_t i) const;

    LevelSplits splits_;
    LeafValues leaf_values_;
    Params params_;
};

} // namespace bonsai
