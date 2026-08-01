"""Tests for bonsai.bench.params (the shared knob mappings)."""

from __future__ import annotations

import sys

from bonsai.bench import params


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
    # the shim keeps the documented import path alive
    sys.path.insert(0, "scripts")
    import reference_params as rp
    assert rp.xgb_core is params.xgb_core


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
                                 grower="oblivious", objective="logloss")
    assert dict(logloss)["dispatch.objective_name"] == "logloss"
