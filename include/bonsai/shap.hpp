#pragma once

#include "bonsai/dataset.hpp"
#include "bonsai/detail/bin_walk.hpp"
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

// Same walk with the tree's expected value supplied by the caller. The bias
// is row-independent, so a batch caller computes tree_expected_value(tree)
// once per tree instead of once per (row, tree).
void tree_shap(DenseTree const &tree, features_view X, row_id_t row,
               std::span<double> phi, double expected_value);

// The same walk over a row the caller already binned: the hot child follows
// routes_left on the row's bin instead of comparing the raw value, which is
// the same child under the dataset's own cuts. sb is split_bins(tree, bins).
// Everything downstream (covers, path arithmetic, bias) is identical, so the
// contributions are bit-identical to the raw call.
void tree_shap_binned(DenseTree const &tree, detail::SplitBins const &sb,
                      Dataset const &bins, row_id_t row, std::span<double> phi,
                      double expected_value);

// Conditional expectation of the tree given the feature subset S (features
// present follow x; absent features average children by cover). This is the
// quantity TreeSHAP attributes; exposed for the brute-force Shapley
// reference used in tests.
double tree_expected_value(DenseTree const &tree, features_view X, row_id_t row,
                           std::span<bool const> in_subset);

// The unconditioned expectation (empty subset): the cover-weighted mean of
// leaf values, a per-tree constant that never reads a row.
double tree_expected_value(DenseTree const &tree);

} // namespace bonsai
