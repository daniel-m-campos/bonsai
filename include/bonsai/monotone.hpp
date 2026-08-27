#pragma once

#include <algorithm>
#include <span>
#include <vector>

#include "bonsai/tree.hpp"

namespace bonsai
{

// Projects an oblivious tree's finished leaf values onto the nearest table
// that honours every monotone constraint, and is the whole of how the
// levelwise grower supports them: there is no split veto, so the tree's
// structure is chosen exactly as it would be unconstrained.
//
// What it guarantees. `leaf_table` holds 2^D leaves indexed by the bit vector
// of level outcomes, level 0 the most significant bit and 0 meaning the left
// (below-threshold) child (INVARIANT perfect-tree-numbering-one-scheme).
// `level_directions[lvl]` is +1, -1 or 0 for the feature that level cut. On
// return every pair of leaves differing only in a constrained level's bit is
// ordered by that level's sign. Because monotone functions are closed under
// addition and positive scaling and boosting is a positively weighted sum of
// trees, a per-tree projection makes the whole ensemble monotone exactly,
// which is why the end-to-end violation is 0 and not merely small.
//
// What breaks it. Anything that rewrites leaf values after the projection
// runs: leaf renewal (MAE, Huber, Quantile) does exactly that, so the booster
// projects again once renewal has finished. `weights` must be parallel to
// `leaf_table`; pass the per-leaf hessians during growth, since weighted
// isotonic regression on the Newton step is the constrained minimiser of the
// second-order objective, and the row counts after renewal, where the leaf
// values are no longer Newton steps.
//
// What it is not. With two or more constrained levels the leaves form a
// partial order rather than a chain, and this runs isotonic regression along
// one linear extension of it. Every constraint is satisfied, but the result
// need not be the L2-nearest monotone table; a single constrained level is
// exact. CatBoost's SymmetricTree path accepts the same approximation.
//
// Pinned by tests/unit/test_monotone.cpp, and end to end by INVARIANT
// levelwise-monotone-holds.
void project_monotone(std::span<int const>   level_directions,
                      std::span<float const> weights, std::span<float> leaf_table);

inline std::vector<int>
monotone_levels(std::span<ObliviousTree::LevelSplit const> splits,
                std::span<int const>                       constraints)
{
    std::vector<int> directions;
    directions.reserve(splits.size());
    for (auto const &split : splits)
    {
        directions.push_back(
            split.feature_id < constraints.size() ? constraints[split.feature_id] : 0);
    }
    return directions;
}

inline bool has_monotone_constraint(std::span<int const> constraints)
{
    return std::ranges::any_of(constraints,
                               [](int direction) { return direction != 0; });
}

} // namespace bonsai
