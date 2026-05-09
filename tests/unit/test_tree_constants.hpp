#pragma once

#include <cstddef>
#include <limits>

#include "bonsai/types.hpp"

namespace bonsai::test
{

// ---- Sentinel ---------------------------------------------------------------

auto constexpr f_nan = std::numeric_limits<float>::quiet_NaN();

// ---- Single-leaf test -------------------------------------------------------

auto constexpr single_leaf_value = 0.25F;
auto constexpr single_leaf_input = 42.0F;

// ---- one_split helper defaults ---------------------------------------------

auto constexpr default_threshold  = 1.0F;
auto constexpr default_left_leaf  = -0.5F;
auto constexpr default_right_leaf = 0.5F;

// ---- Boundary / routing inputs ---------------------------------------------

auto constexpr below_threshold = 0.5F;
auto constexpr above_threshold = 2.0F;
auto constexpr at_threshold    = 1.0F;

// ---- Multi-level tree structure --------------------------------------------

auto constexpr left_internal_node  = node_id_t{1};
auto constexpr right_internal_node = node_id_t{2};
auto constexpr leaf_ll_idx         = node_id_t{3};
auto constexpr leaf_lr_idx         = node_id_t{4};
auto constexpr leaf_rl_idx         = node_id_t{5};
auto constexpr leaf_rr_idx         = node_id_t{6};

auto constexpr root_threshold          = 1.0F;
auto constexpr left_subtree_threshold  = 0.5F;
auto constexpr right_subtree_threshold = 2.0F;

auto constexpr leaf_ll_value = -1.0F;
auto constexpr leaf_lr_value = -0.5F;
auto constexpr leaf_rl_value = 0.5F;
auto constexpr leaf_rr_value = 1.0F;

auto constexpr fid_0 = feature_id_t{0};
auto constexpr fid_1 = feature_id_t{1};

auto constexpr f1_below_half = 0.25F;
auto constexpr f1_one        = 1.0F;
auto constexpr f1_below_two  = 0.5F;
auto constexpr f1_above_two  = 3.0F;

// ---- Batch test inputs ------------------------------------------------------

auto constexpr batch_below = 0.5F;
auto constexpr batch_above = 2.0F;

// ---- params() accessor expectations ----------------------------------------

auto constexpr params_single_depth  = size_t{0};
auto constexpr params_single_leaves = size_t{1};
auto constexpr params_multi_depth   = size_t{2};
auto constexpr params_multi_leaves  = size_t{4};

} // namespace bonsai::test
