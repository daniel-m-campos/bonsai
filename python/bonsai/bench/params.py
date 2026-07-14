"""One source of truth for benchmark knobs and reference-library mappings.

A one-knob drift here has produced a false experimental conclusion twice
(max_bin 255 vs 256, decision 55's follow-up; min_child_weight 1 vs 20,
decision 68's correction), so every harness and probe imports these instead
of re-deriving them.
"""

from __future__ import annotations

# The campaign regime: quality-division suites (Grinsztajn standings, the
# internal smoke campaign, admission-gate probes). Decisions 56 and 68.
CAMPAIGN = dict(iters=200, lr=0.05, depth=6, bins=255, min_data_in_leaf=20,
                lambda_l2=1.0, seed=42)

# The scale regime: perf-division synthetic sweeps. Decision 46.
SCALING = dict(iters=100, lr=0.1, depth=8, bins=255, min_data_in_leaf=20,
               lambda_l2=1.0, seed=42)

# bonsai dotted keys for the campaign regime (the estimator kwargs carry the
# rest); used by quality suites and probes.
BONSAI_CAMPAIGN_PARAMS = {
    "tree.min_data_in_leaf": 20,
    "tree.lambda_l2": 1.0,
    "bin_mapper.max_bin": 255,
}


def num_leaves_campaign(depth: int) -> int:
    """The campaign/compare convention: (1 << depth) - 1 leaves."""
    return (1 << depth) - 1


def num_leaves_full(depth: int) -> int:
    """The scaling/gpu-bench convention: full 1 << depth leaves."""
    return 1 << depth


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
