"""One source of truth for benchmark knobs and reference-library mappings.

A one-knob drift here has produced a false experimental conclusion twice
(max_bin 255 vs 256, decision 55's follow-up; min_child_weight 1 vs 20,
decision 68's correction), so every harness and probe imports these instead
of re-deriving them.
"""

from __future__ import annotations

from bonsai.bench.variants import Device

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


def bonsai_core(*, learning_rate, max_depth, num_leaves, min_data_in_leaf,
                lambda_l2, max_bin, seed, n_iters, n_threads, grower,
                objective="mse") -> list[tuple[str, str]]:
    """Dotted-key pairs for bonsai.train, mirroring the reference mappings
    below so harnesses stop hand-building (and drifting) the pair list."""
    return [
        ("dispatch.grower_name", grower),
        ("dispatch.objective_name", objective),
        ("booster.n_iters", str(n_iters)),
        ("booster.learning_rate", str(learning_rate)),
        ("booster.random_seed", str(seed)),
        ("tree.max_depth", str(max_depth)),
        ("tree.max_leaves", str(num_leaves)),
        ("tree.min_data_in_leaf", str(min_data_in_leaf)),
        ("tree.lambda_l2", str(lambda_l2)),
        ("bin_mapper.max_bin", str(max_bin)),
        ("parallel.n_threads", str(n_threads)),
    ]


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
    # max_bin arrives in BIN semantics (what bonsai/xgboost/lightgbm count);
    # CatBoost's border_count counts SPLITS, so bins - 1. The fencepost lives
    # here, once, because per-call-site translation drifted three ways across
    # harnesses (2026-07-30 fairness review). GPU caps border_count at 254
    # (= 255 bins, matching the campaign/scale default exactly).
    borders = max_bin - 1
    return {
        "learning_rate": learning_rate,
        "depth": max_depth,
        "l2_leaf_reg": lambda_l2,
        "border_count": min(borders, 254) if device == Device.CUDA else borders,
        "random_seed": seed,
    }
