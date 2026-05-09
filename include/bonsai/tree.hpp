#pragma once

#include <concepts>
#include <cstddef>
#include <variant>
#include <vector>

#include "bonsai/types.hpp"

namespace bonsai
{

template <typename T>
concept Tree = requires(T const t, floats_view row, floats_view rows, size_t n_features,
                        floats_out out) {
    { t.params() } -> std::same_as<typename T::Params const &>;
    { t.predict(row) } -> std::same_as<float>;
    { t.predict(rows, n_features, out) } -> std::same_as<void>;
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

    float predict(floats_view row) const;

    void predict(floats_view rows, size_t n_features, floats_out out) const;

    Params const &params() const
    {
        return params_;
    }

  private:
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

    float predict(floats_view row) const;

    void predict(floats_view rows, size_t n_features, floats_out out) const;

    Params const &params() const
    {
        return params_;
    }

  private:
    LevelSplits splits_;
    LeafValues leaf_values_;
    Params params_;
};

} // namespace bonsai
