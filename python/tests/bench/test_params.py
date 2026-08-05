"""Tests for bonsai.bench.params (the shared knob mappings)."""

from __future__ import annotations

import sys

from bonsai import interop
from bonsai.bench import params

# One canonical cell, stated once in bonsai's own keys, for the drift guard.
CANONICAL_NATIVE = [
    ("booster.learning_rate", 0.05),
    ("tree.max_depth", 6),
    ("tree.lambda_l2", 1.0),
    ("bin_mapper.max_bin", 255),
    ("booster.random_seed", 42),
]


def test_reference_param_mappings():
    x = params.xgb_core(learning_rate=0.05, max_depth=6, min_data_in_leaf=20,
                        lambda_l2=1.0, max_bin=255, seed=42)
    assert x["min_child_weight"] == 20 and x["tree_method"] == "hist"
    lgb = params.lgbm_core(learning_rate=0.1, max_depth=8, num_leaves=256,
                           min_data_in_leaf=20, lambda_l2=1.0, max_bin=255,
                           seed=42)
    assert lgb["num_leaves"] == 256 and lgb["verbose"] == -1
    # max_bin arrives in BIN semantics; catboost_core owns the borders
    # fencepost (bins - 1) and the GPU 254-border cap (fairness review
    # 2026-07-30: call-site translations had drifted three ways).
    cb_gpu = params.catboost_core(learning_rate=0.1, max_depth=8, lambda_l2=1.0,
                                  max_bin=1023, seed=42, device="cuda")
    cb_cpu = params.catboost_core(learning_rate=0.1, max_depth=8, lambda_l2=1.0,
                                  max_bin=1023, seed=42, device="cpu")
    assert cb_gpu["border_count"] == 254 and cb_cpu["border_count"] == 1022
    cb_255 = params.catboost_core(learning_rate=0.1, max_depth=8, lambda_l2=1.0,
                                  max_bin=255, seed=42, device="cpu")
    assert cb_255["border_count"] == 254  # 255 bins, matching the others
    assert params.num_leaves_campaign(6) == 63
    assert params.num_leaves_full(6) == 64
    # An uncapped-depth cell names its own budget; 1 << 255 is not one.
    assert params.num_leaves_of({"depth": 6}) == 64
    assert params.num_leaves_of({"depth": 255, "num_leaves": 256}) == 256
    # the shim keeps the documented import path alive
    sys.path.insert(0, "scripts")
    import reference_params as rp
    assert rp.xgb_core is params.xgb_core


def test_reference_builders_agree_with_interop():
    """The builders must name every shared knob exactly as interop does.

    params.py may add a benchmark's own defaults on top (tree_method,
    verbose, the GPU border cap), but it may not rename a knob: one mapping
    table drives both the estimator layer and the harness, and a second
    spelling here is the drift that produced a published correction once.
    """
    builders = (
        (params.xgb_core(learning_rate=0.05, max_depth=6, min_data_in_leaf=20,
                         lambda_l2=1.0, max_bin=255, seed=42),
         interop.to_xgboost(CANONICAL_NATIVE)),
        (params.lgbm_core(learning_rate=0.05, max_depth=6, num_leaves=63,
                          min_data_in_leaf=20, lambda_l2=1.0, max_bin=255,
                          seed=42),
         interop.to_lightgbm(CANONICAL_NATIVE)),
        (params.catboost_core(learning_rate=0.05, max_depth=6, lambda_l2=1.0,
                              max_bin=255, seed=42, device="cpu"),
         interop.to_catboost(CANONICAL_NATIVE)),
    )
    for built, mapped in builders:
        assert mapped, "the canonical cell must translate to something"
        assert {key: built[key] for key in mapped} == mapped


def test_bonsai_core_pairs():
    # The exact dotted keys run_bonsai used to hand-build; a drift here
    # changes every perf row silently.
    pairs = params.bonsai_core(learning_rate=0.1, max_depth=8, num_leaves=256,
                               min_data_in_leaf=20, lambda_l2=1.0, max_bin=255,
                               seed=42, n_iters=100, n_threads=16,
                               grower="depthwise")
    assert dict(pairs) == {
        "dispatch.grower_name": "depthwise",
        "dispatch.objective_name": "mse",
        "booster.n_iters": "100",
        "booster.learning_rate": "0.1",
        "booster.random_seed": "42",
        "tree.max_depth": "8",
        "tree.max_leaves": "256",
        "tree.min_data_in_leaf": "20",
        "tree.lambda_l2": "1.0",
        "bin_mapper.max_bin": "255",
        "parallel.n_threads": "16",
    }
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in pairs)
    logloss = params.bonsai_core(learning_rate=0.1, max_depth=8, num_leaves=256,
                                 min_data_in_leaf=20, lambda_l2=1.0, max_bin=255,
                                 seed=42, n_iters=100, n_threads=16,
                                 grower="levelwise", objective="logloss")
    assert dict(logloss)["dispatch.objective_name"] == "logloss"


def test_early_stop_mappings():
    """Each library's patience must arrive through its documented surface.

    Hand-deriving one of these is the prohibited act (params.py's opening
    note), so the four translations are pinned by name here: a rename in a
    reference library must break this test, not a pod session.
    """
    pairs = dict(params.bonsai_core(learning_rate=0.05, max_depth=8,
                                    num_leaves=256, min_data_in_leaf=20,
                                    lambda_l2=1.0, max_bin=255, seed=42,
                                    n_iters=2000, n_threads=16,
                                    grower="depthwise",
                                    early_stopping_rounds=50))
    assert pairs["booster.early_stopping_rounds"] == "50"
    assert params.xgb_early_stop(50) == {"early_stopping_rounds": 50}
    assert params.lgbm_early_stop(50)["stopping_rounds"] == 50
    assert params.catboost_early_stop(50, has_eval_set=True) == {
        "od_type": "Iter", "od_wait": 50}
    # CatBoost shrinks to its best iteration merely because an eval set was
    # passed, so the fixed-iteration arm must switch that off or it reports
    # a shorter model than the one it was timed training.
    assert params.catboost_early_stop(0, has_eval_set=True) == {
        "use_best_model": False}


def test_early_stop_mappings_are_absent_when_unarmed():
    """Zero patience omits the mechanism instead of writing a default."""
    pairs = params.bonsai_core(learning_rate=0.1, max_depth=8, num_leaves=256,
                               min_data_in_leaf=20, lambda_l2=1.0, max_bin=255,
                               seed=42, n_iters=100, n_threads=16,
                               grower="depthwise")
    assert "booster.early_stopping_rounds" not in dict(pairs)
    assert params.xgb_early_stop(0) == {}
    assert params.lgbm_early_stop(0) == {}
    assert params.catboost_early_stop(0, has_eval_set=False) == {}
    assert params.EARLY_STOP["patience"] == 50
    assert params.EARLY_STOP["iters"] == 2000
