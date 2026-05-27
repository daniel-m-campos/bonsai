#pragma once

#include <cstddef>
#include <limits>

#include "bonsai/types.hpp"

namespace bonsai::test
{

// ---- Sentinel ---------------------------------------------------------------

constexpr auto f_nan = std::numeric_limits<float>::quiet_NaN();

// ---- Single-leaf test -------------------------------------------------------

constexpr auto single_leaf_value = 0.25F;
constexpr auto single_leaf_input = 42.0F;

// ---- one_split helper defaults ---------------------------------------------

constexpr auto default_threshold  = 1.0F;
constexpr auto default_left_leaf  = -0.5F;
constexpr auto default_right_leaf = 0.5F;

// ---- Boundary / routing inputs ---------------------------------------------

constexpr auto below_threshold = 0.5F;
constexpr auto above_threshold = 2.0F;
constexpr auto at_threshold    = 1.0F;

// ---- Multi-level tree structure --------------------------------------------

constexpr auto left_internal_node  = node_id_t{1};
constexpr auto right_internal_node = node_id_t{2};
constexpr auto leaf_ll_idx         = node_id_t{3};
constexpr auto leaf_lr_idx         = node_id_t{4};
constexpr auto leaf_rl_idx         = node_id_t{5};
constexpr auto leaf_rr_idx         = node_id_t{6};

constexpr auto root_threshold          = 1.0F;
constexpr auto left_subtree_threshold  = 0.5F;
constexpr auto right_subtree_threshold = 2.0F;

constexpr auto leaf_ll_value = -1.0F;
constexpr auto leaf_lr_value = -0.5F;
constexpr auto leaf_rl_value = 0.5F;
constexpr auto leaf_rr_value = 1.0F;

constexpr auto fid_0 = feature_id_t{0};
constexpr auto fid_1 = feature_id_t{1};

constexpr auto f1_below_half = 0.25F;
constexpr auto f1_one        = 1.0F;
constexpr auto f1_below_two  = 0.5F;
constexpr auto f1_above_two  = 3.0F;

// ---- Batch test inputs ------------------------------------------------------

constexpr auto batch_below = 0.5F;
constexpr auto batch_above = 2.0F;

// ---- params() accessor expectations ----------------------------------------

constexpr auto params_single_depth  = size_t{0};
constexpr auto params_single_leaves = size_t{1};
constexpr auto params_multi_depth   = size_t{2};
constexpr auto params_multi_leaves  = size_t{4};

} // namespace bonsai::test
