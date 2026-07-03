#pragma once

#include "bonsai/tree.hpp"
#include "bonsai/types.hpp"
#include <span>

namespace bonsai
{

// TreeSHAP (Lundberg et al., "Consistent Individualized Feature Attribution
// for Tree Ensembles", Algorithm 2): exact Shapley values for one tree in
// O(leaves * depth^2) per row, using per-node training covers as the
// background distribution.
//
// Accumulates into `phi` (size n_features + 1): phi[f] += contribution of
// feature f, phi[n_features] += the tree's expected value (bias). The
// efficiency property holds exactly: sum(phi) == tree prediction for x.
// Throws std::invalid_argument if the tree carries no covers.
void tree_shap(DenseTree const &tree, features_view X, row_id_t row,
               std::span<double> phi);

// Conditional expectation of the tree given the feature subset S (features
// present follow x; absent features average children by cover). This is the
// quantity TreeSHAP attributes; exposed for the brute-force Shapley
// reference used in tests.
double tree_expected_value(DenseTree const &tree, features_view X, row_id_t row,
                           std::span<bool const> in_subset);

} // namespace bonsai
