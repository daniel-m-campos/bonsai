#include "bonsai/shap.hpp"

#include "bonsai/tree.hpp"
#include "bonsai/types.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>

namespace bonsai
{

namespace
{

// One entry of the "unique path" of Algorithm 2: the fractions of subsets
// that flow down this branch with the feature blocked (zero) or active
// (one), and the permutation weight accumulated so far.
struct PathElement
{
    int    feature      = -1;
    double zero_fraction = 0.0;
    double one_fraction  = 0.0;
    double pweight       = 0.0;
};

// Extend the path with a new (zero, one, feature) split condition.
void extend(std::vector<PathElement> &path, size_t length, double zero_fraction,
            double one_fraction, int feature)
{
    path[length] = {.feature       = feature,
                    .zero_fraction = zero_fraction,
                    .one_fraction  = one_fraction,
                    .pweight       = length == 0 ? 1.0 : 0.0};
    for (size_t i = length; i-- > 0;)
    {
        path[i + 1].pweight += one_fraction * path[i].pweight *
                               (static_cast<double>(i) + 1.0) /
                               (static_cast<double>(length) + 1.0);
        path[i].pweight = zero_fraction * path[i].pweight *
                          (static_cast<double>(length - i)) /
                          (static_cast<double>(length) + 1.0);
    }
}

// Undo an extend at position `index` (feature reappears on the path).
void unwind(std::vector<PathElement> &path, size_t length, size_t index)
{
    double const one_fraction  = path[index].one_fraction;
    double const zero_fraction = path[index].zero_fraction;
    double       next          = path[length].pweight;
    for (size_t i = length; i-- > 0;)
    {
        if (one_fraction != 0.0)
        {
            double const tmp = path[i].pweight;
            path[i].pweight  = next * (static_cast<double>(length) + 1.0) /
                              ((static_cast<double>(i) + 1.0) * one_fraction);
            next = tmp - path[i].pweight * zero_fraction *
                             (static_cast<double>(length - i)) /
                             (static_cast<double>(length) + 1.0);
        }
        else
        {
            path[i].pweight = path[i].pweight *
                              (static_cast<double>(length) + 1.0) /
                              (zero_fraction * static_cast<double>(length - i));
        }
    }
    for (size_t i = index; i < length; ++i)
    {
        path[i].feature       = path[i + 1].feature;
        path[i].zero_fraction = path[i + 1].zero_fraction;
        path[i].one_fraction  = path[i + 1].one_fraction;
    }
}

// Sum of pweights if the element at `index` were unwound, without mutating.
double unwound_path_sum(std::vector<PathElement> const &path, size_t length,
                        size_t index)
{
    double const one_fraction  = path[index].one_fraction;
    double const zero_fraction = path[index].zero_fraction;
    double       next          = path[length].pweight;
    double       total         = 0.0;
    for (size_t i = length; i-- > 0;)
    {
        if (one_fraction != 0.0)
        {
            double const tmp = next * (static_cast<double>(length) + 1.0) /
                               ((static_cast<double>(i) + 1.0) * one_fraction);
            total += tmp;
            next = path[i].pweight - tmp * zero_fraction *
                                         (static_cast<double>(length - i)) /
                                         (static_cast<double>(length) + 1.0);
        }
        else
        {
            total += path[i].pweight * (static_cast<double>(length) + 1.0) /
                     (zero_fraction * static_cast<double>(length - i));
        }
    }
    return total;
}

struct ShapContext
{
    DenseTree::Nodes const   &nodes;
    std::vector<float> const &covers;
    features_view             X;
    row_id_t                  row;
    std::span<double>         phi;
};

// Which child does the instance follow at an internal node?
node_id_t hot_child(DenseTree::Node const &n, features_view X, row_id_t row)
{
    float const v      = X[row, n.feature_id];
    bool const  is_nan = std::isnan(v);
    bool const  left   = (!is_nan && v <= n.threshold_or_value) ||
                       (is_nan && n.default_left);
    return left ? n.left : n.right;
}

// NOLINTNEXTLINE(misc-no-recursion)
void recurse(ShapContext const &ctx, node_id_t node_id,
             std::vector<PathElement> &path, size_t length,
             double parent_zero_fraction, double parent_one_fraction,
             int parent_feature)
{
    // Work on a copy of the parent's path (Algorithm 2 duplicates it).
    extend(path, length, parent_zero_fraction, parent_one_fraction,
           parent_feature);
    ++length;

    auto const &n = ctx.nodes[node_id];
    if (DenseTree::is_leaf(n))
    {
        for (size_t i = 1; i < length; ++i)
        {
            double const w = unwound_path_sum(path, length - 1, i);
            ctx.phi[static_cast<size_t>(path[i].feature)] +=
                w * (path[i].one_fraction - path[i].zero_fraction) *
                n.threshold_or_value;
        }
        return;
    }

    node_id_t const hot  = hot_child(n, ctx.X, ctx.row);
    node_id_t const cold = hot == n.left ? n.right : n.left;

    double const cover      = ctx.covers[node_id];
    double const hot_frac   = cover > 0.0 ? ctx.covers[hot] / cover : 0.0;
    double const cold_frac  = cover > 0.0 ? ctx.covers[cold] / cover : 0.0;

    // If this feature already conditioned the path, undo that entry and
    // fold its fractions into the new one (a feature contributes once).
    double incoming_zero = 1.0;
    double incoming_one  = 1.0;
    for (size_t i = 1; i < length; ++i)
    {
        if (path[i].feature == static_cast<int>(n.feature_id))
        {
            incoming_zero = path[i].zero_fraction;
            incoming_one  = path[i].one_fraction;
            unwind(path, length - 1, i);
            --length;
            break;
        }
    }

    auto const feature = static_cast<int>(n.feature_id);
    {
        auto hot_path = path; // Algorithm 2 duplicates the path per branch
        recurse(ctx, hot, hot_path, length, incoming_zero * hot_frac,
                incoming_one, feature);
    }
    {
        auto cold_path = path;
        recurse(ctx, cold, cold_path, length, incoming_zero * cold_frac, 0.0,
                feature);
    }
}

} // namespace

void tree_shap(DenseTree const &tree, features_view X, row_id_t row,
               std::span<double> phi)
{
    auto const &covers = tree.covers();
    if (covers.size() != tree.nodes().size())
    {
        throw std::invalid_argument(
            "tree_shap: tree carries no per-node covers (model predates "
            "format v6 or was hand-built)");
    }
    // Bias: the tree's expected value over the training distribution.
    phi[phi.size() - 1] += tree_expected_value(tree, X, row, {});

    std::vector<PathElement> path(tree.params().depth + 2);
    ShapContext const        ctx{.nodes  = tree.nodes(),
                                 .covers = covers,
                                 .X      = X,
                                 .row    = row,
                                 .phi    = phi};
    recurse(ctx, 0, path, 0, 1.0, 1.0, -1);
}

namespace
{
// NOLINTNEXTLINE(misc-no-recursion)
double expected_value_impl(DenseTree const &tree, features_view X,
                                  row_id_t row, std::span<bool const> in_subset,
                                  node_id_t id)
{
    auto const &n = tree.nodes()[id];
    if (DenseTree::is_leaf(n))
    {
        return n.threshold_or_value;
    }
    bool const conditioned =
        n.feature_id < in_subset.size() && in_subset[n.feature_id];
    if (conditioned)
    {
        return expected_value_impl(tree, X, row, in_subset,
                                   hot_child(n, X, row));
    }
    auto const  &covers = tree.covers();
    double const cover  = covers[id];
    if (cover <= 0.0)
    {
        return 0.0;
    }
    return (covers[n.left] *
                expected_value_impl(tree, X, row, in_subset, n.left) +
            covers[n.right] *
                expected_value_impl(tree, X, row, in_subset, n.right)) /
           cover;
}

} // namespace

double tree_expected_value(DenseTree const &tree, features_view X, row_id_t row,
                           std::span<bool const> in_subset)
{
    return expected_value_impl(tree, X, row, in_subset, 0);
}

} // namespace bonsai
