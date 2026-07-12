# Canonical numeric-knob translation from bonsai's vocabulary to each
# reference library, shared by compare.py / bench_gpu.py / bench_scaling.py.
# One source of truth for the keys every harness maps (design review
# 2026-07-12): a one-knob drift here already produced a false experimental
# conclusion once (max_bin 255 vs 256, decision 55's follow-up), so the
# mapping lives in exactly one place. Objectives, metrics, sampling, DART,
# and constraint extras remain script-specific and layer on top.
#
# Known convention split, preserved deliberately (a benchmark-version
# decision, not a refactor): compare.py derives lightgbm num_leaves as
# (1 << depth) - 1 when max_leaves is unset; the bench scripts use
# 1 << depth. Callers pass num_leaves explicitly for that reason.


def xgb_core(*, learning_rate, max_depth, min_data_in_leaf, lambda_l2, max_bin,
             seed) -> dict:
    return {
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "min_child_weight": min_data_in_leaf,
        "reg_lambda": lambda_l2,
        "max_bin": max_bin,
        "tree_method": "hist",
        "seed": seed,
    }


def lgbm_core(*, learning_rate, max_depth, num_leaves, min_data_in_leaf,
              lambda_l2, max_bin, seed) -> dict:
    return {
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "num_leaves": num_leaves,
        "min_data_in_leaf": min_data_in_leaf,
        "lambda_l2": lambda_l2,
        "max_bin": max_bin,
        "seed": seed,
        "verbose": -1,
    }


def catboost_core(*, learning_rate, max_depth, lambda_l2, max_bin, seed,
                  device) -> dict:
    # CatBoost caps GPU border_count at 254 (it clamps/rejects above);
    # making the cap explicit keeps CPU/GPU comparisons honest.
    return {
        "learning_rate": learning_rate,
        "depth": max_depth,
        "l2_leaf_reg": lambda_l2,
        "border_count": min(max_bin, 254) if device == "cuda" else max_bin,
        "random_seed": seed,
    }
