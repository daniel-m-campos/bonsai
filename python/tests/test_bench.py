"""bonsai.bench invariants: run with PYTHONPATH=build/python python
python/tests/test_bench.py (wired into make python-test)."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile

import numpy as np
from bonsai.bench import metrics, params, runlog, synth

# Captured from scripts/bench_scaling.py::gen_data BEFORE the move to
# bonsai.bench.synth (2026-07-14): the generator must stay byte-stable or
# every perf-division result loses comparability.
GEN_DATA_GOLDENS = {
    (10_000, 20, 42, 1000, 20): ["33172850a22efea3", "72dd8c4d554f8e15",
                                 "2a69346b81ee31cb", "699d900c1abda049"],
    (5000, 7, 0, 500, 5): ["4052a735cd2e4edf", "bf4a1f3766b784c6",
                           "62e923970a54e7a9", "2e46ac59454ce2d1"],
}


def test_gen_data_bytestable():
    for args, want in GEN_DATA_GOLDENS.items():
        got = [hashlib.sha256(a.tobytes()).hexdigest()[:16]
               for a in synth.gen_data(*args)]
        assert got == want, (args, got, want)


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


def test_variant_registry():
    from bonsai.bench import airline, grinsztajn, scaling, variants

    # Committed rows pin these exact spellings forever.
    assert variants.SCALING == tuple(scaling.VARIANTS) == (
        "bonsai_depthwise", "bonsai_leafwise", "bonsai_oblivious",
        "bonsai_cuda_depthwise", "bonsai_cuda_oblivious", "xgb_hist",
        "xgb_cuda", "lgbm_cpu", "lgbm_cuda", "catboost_cpu", "catboost_gpu")
    assert variants.AIRLINE == tuple(airline.VARIANTS) == (
        "bonsai_depthwise", "bonsai_oblivious", "bonsai_cuda_depthwise",
        "bonsai_cuda_oblivious", "bonsai_ts_depthwise", "bonsai_ts_oblivious",
        "bonsai_ts_cuda_depthwise", "bonsai_ts_cuda_oblivious", "xgb_hist",
        "xgb_cuda", "lgbm_cpu", "catboost_cpu", "catboost_gpu")
    assert grinsztajn.VARIANTS == variants.GRINSZTAJN == (
        "bonsai_dw", "bonsai_lw", "bonsai_obl", "xgb", "lgbm", "catboost")
    for n in (*variants.SCALING, *variants.AIRLINE):
        assert variants.resolve(n).name == n
    # Historical alias spellings resolve to the intended canonical arm.
    for alias, canon in {"bonsai_dw": "bonsai_depthwise",
                         "bonsai_lw": "bonsai_leafwise",
                         "bonsai_obl": "bonsai_oblivious",
                         "xgb": "xgb_hist", "lgbm": "lgbm_cpu",
                         "catboost": "catboost_cpu",
                         "bonsai_cpu": "bonsai_depthwise",
                         "bonsai_leaf_cpu": "bonsai_leafwise",
                         "bonsai_gpu": "bonsai_cuda_depthwise",
                         "bonsai_obl_gpu": "bonsai_cuda_oblivious",
                         "xgb_cpu": "xgb_hist",
                         "xgb_gpu": "xgb_cuda"}.items():
        assert variants.resolve(alias).name == canon
    # No alias shadows a canonical name.
    aliases = {a for v in variants.REGISTRY.values() for a in v.aliases}
    assert not aliases & set(variants.REGISTRY)
    # The (lib, device) views survive the derivation.
    assert scaling.VARIANTS["bonsai_cuda_oblivious"] == ("bonsai", "cuda")
    assert scaling.VARIANTS["lgbm_cuda"] == ("lgbm", "cuda")
    assert airline.VARIANTS["bonsai_ts_cuda_depthwise"] == ("bonsai", "cuda")
    try:
        variants.resolve("nope")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown variant must be rejected")


def test_metrics_against_sklearn():
    rng = np.random.default_rng(0)
    y = rng.random(500)
    pred = y + rng.normal(0, 0.1, 500)
    yb = (rng.random(500) > 0.5).astype(float)
    scores = yb * 0.6 + rng.random(500) * 0.4
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        roc_auc_score,
    )
    assert abs(metrics.r2(y, pred) - r2_score(y, pred)) < 1e-12
    assert abs(metrics.rmse(y, pred) - mean_squared_error(y, pred) ** 0.5) < 1e-12
    assert abs(metrics.mae(y, pred) - mean_absolute_error(y, pred)) < 1e-12
    assert abs(metrics.auc(yb, scores) - roc_auc_score(yb, scores)) < 1e-12
    # the numpy fallback must agree with sklearn, including under ties
    tied = np.round(scores, 1)
    import unittest.mock as mock
    with mock.patch.dict(sys.modules, {"sklearn.metrics": None, "sklearn": None}):
        fallback = metrics.auc(yb, tied)
    assert abs(fallback - roc_auc_score(yb, tied)) < 1e-12


def test_runlog_roundtrip():
    with tempfile.NamedTemporaryFile("r", suffix=".jsonl") as f:
        knobs = {"b": 2, "a": 1}
        row = runlog.emit_row(f.name, division="perf", suite="scaling",
                              knobs=knobs, timing_mode="in_memory",
                              variant="bonsai_dw", value=1.0, metric="r2",
                              status="ok")
        back = json.loads(open(f.name).read().splitlines()[-1])
        assert back == json.loads(json.dumps(row))
        for key in ("schema", "ts", "git_sha", "division", "suite", "cmd",
                    "timing_mode", "host", "knobs", "knobs_hash"):
            assert key in back, key
        # hash is canonical: key order must not matter
        assert runlog.knobs_hash({"a": 1, "b": 2}) == back["knobs_hash"]
    try:
        runlog.emit_row("/tmp/x.jsonl", division="nope", suite="s")
    except AssertionError:
        pass
    else:
        raise AssertionError("bad division must be rejected")


def test_bench_import_is_lazy():
    """Importing bonsai.bench must not drag in the reference libraries."""
    for heavy in ("xgboost", "lightgbm", "catboost", "openml"):
        assert heavy not in sys.modules, f"{heavy} imported eagerly"


if __name__ == "__main__":
    test_bench_import_is_lazy()
    test_gen_data_bytestable()
    test_reference_param_mappings()
    test_bonsai_core_pairs()
    test_variant_registry()
    test_metrics_against_sklearn()
    test_runlog_roundtrip()
    print("all bench tests passed")
