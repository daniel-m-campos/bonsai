#pragma once

#include "bonsai/config/tree_config.hpp"
#include "bonsai/histogram.hpp"
#include "bonsai/row_view.hpp"
#include "bonsai/types.hpp"
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <limits>
#include <ranges>
#include <span>
#include <vector>

namespace bonsai
{

// Whether a row list is exactly [0, n_rows). Cardinality alone does not say
// so: a permutation and a with-replacement bootstrap both fill n slots.
inline bool rows_are_identity(std::span<row_id_t const> rows, size_t n_rows)
{
    return rows.size() == n_rows &&
           std::ranges::equal(
               rows, std::views::iota(row_id_t{0}, static_cast<row_id_t>(n_rows)));
}

struct SplitInput
{
    NodeHistograms        hists;
    std::vector<row_id_t> rows;
    // What the fills may assume about `rows`. Root builders set it from the
    // caller's RowSelection; a child's rows are a strict subset, so the
    // default (no assumptions) is right for them.
    RowShape  shape = {};
    node_id_t id    = 0;
    // Leaf-value bounds inherited down the tree by monotone constraints.
    double lo = -std::numeric_limits<double>::infinity();
    double hi = std::numeric_limits<double>::infinity();
    // Features this node may split on under interaction constraints;
    // empty = all allowed. Indexed by feature id. The default member
    // initializers here keep SplitInput a designated-init aggregate.
    // NOLINTNEXTLINE(readability-redundant-member-init)
    std::vector<char> allowed = {};
    // Distinct features used on the path from the root to this node.
    // NOLINTNEXTLINE(readability-redundant-member-init)
    std::vector<feature_id_t> path = {};
    // Cached node totals + row count. A device-resident engine leaves
    // hists/rows empty and sets these as the node's only host statistics.
    HistCell sums      = {};
    size_t   row_count = 0;

    // Node-level totals: the cached sums when set, else from the first
    // populated histogram (every populated feature sums the same rows;
    // unselected features are zero-binned placeholders and are skipped).
    HistCell totals() const
    {
        if (row_count > 0)
        {
            return sums;
        }
        for (auto const &h : hists)
        {
            if (h.size() != 0)
            {
                return h.totals();
            }
        }
        return {};
    }
    double total_grad() const
    {
        return totals().sum_grad;
    }
    double total_hess() const
    {
        return totals().sum_hess;
    }
};

// A split proposal. Only `valid` says the other fields mean anything: an
// invalid output carries zeroed defaults, not a split at bin 0.
struct SplitOutput
{
    double       gain         = 0.0;
    feature_id_t feature_id   = 0;
    bin_id_t     bin_id       = 0;
    bool         default_left = true;
    bool         valid        = false;
};

template <typename T>
concept NodeSplitFinder = requires(SplitInput const &input, TreeConfig const &config) {
    { T::find(input, config) } -> std::same_as<SplitOutput>;
};

// What the leaf plane needs on top of find: best-first growth has no frontier
// to spread across workers, so it drives the batched find instead of the
// serial one.
template <typename T>
concept ParallelNodeSplitFinder =
    NodeSplitFinder<T> &&
    requires(std::span<SplitInput const> nodes, TreeConfig const &config,
             std::span<SplitOutput> out) {
        { T::find_parallel(nodes, config, out) } -> std::same_as<void>;
    };

struct HistogramNodeSplitFinder
{
    static SplitOutput find(SplitInput const &input, TreeConfig const &config);
    static void        find_parallel(std::span<SplitInput const> nodes,
                                     TreeConfig const &config, std::span<SplitOutput> out);
};

using FrontierInput = std::span<SplitInput const>;

template <typename T>
concept LevelSplitFinder = requires(FrontierInput frontier, TreeConfig const &config) {
    { T::find(frontier, config) } -> std::same_as<SplitOutput>;
};

struct HistogramLevelSplitFinder
{
    static SplitOutput find(FrontierInput frontier, TreeConfig const &config);
};

// Soft-threshold on the gradient sum: XGBoost's L1 treatment.
// constexpr: callable from CUDA device code (clang treats constexpr as
// implicitly host+device), keeping one definition of the gain math.
constexpr double l1_thresholded(double g, double l1)
{
    if (g > l1)
    {
        return g - l1;
    }
    if (g < -l1)
    {
        return g + l1;
    }
    return 0.0;
}

// A zero denominator is an empty node side under lambda_l2 = 0 (an oblivious
// level mints empty children structurally): no evidence means 0, not 0/0.
constexpr double score(double g, double h, double lambda)
{
    double const d = h + lambda;
    return d > 0.0 ? (g * g) / d : 0.0;
}

// g and h are a fixed gradient/hessian pair, not interchangeable operands.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
constexpr double score(double g, double h, double l1, double l2)
{
    double const t = l1_thresholded(g, l1);
    double const d = h + l2;
    return d > 0.0 ? (t * t) / d : 0.0;
}

// Newton leaf weight for the given sums, clamped to the node's monotone
// bounds. The scalar overload exists because TreeConfig holds vectors and
// cannot cross to the device.
constexpr double bounded_leaf_weight(double g, double h, double l1, double l2,
                                     double lo, double hi)
{
    double const d = h + l2;
    double const w = d > 0.0 ? -l1_thresholded(g, l1) / d : 0.0;
    return std::clamp(w, lo, hi);
}

inline double bounded_leaf_weight(double g, double h, TreeConfig const &config,
                                  double lo, double hi)
{
    return bounded_leaf_weight(g, h, config.lambda_l1, config.lambda_l2, lo, hi);
}

// Per-feature monotone direction; features beyond the configured list are
// unconstrained.
inline int monotone_constraint_of(TreeConfig const &config, feature_id_t fid)
{
    return fid < config.monotone_constraints.size() ? config.monotone_constraints[fid]
                                                    : 0;
}

} // namespace bonsai
