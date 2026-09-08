"""Tests for bonsai.bench.grinsztajn (the quality suite's arm placement)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from bonsai.bench import grinsztajn

X = np.zeros((4, 2), dtype=np.float32)
Y = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)


class _Fitted:
    """A fitted estimator stand-in that records the constructor kwargs."""

    def __init__(self, seen, key, trained_device="cuda:0", **kwargs):
        seen[key] = kwargs
        self.trained_device = trained_device

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))

    def predict_proba(self, X):
        return np.zeros((len(X), 2))

    def get_booster(self):
        cfg = {"learner": {"generic_param": {"device": self.trained_device}}}
        return type("Booster", (), {"save_config": lambda self: json.dumps(cfg)})()


def _stub_references(monkeypatch, seen, trained_device="cuda:0"):
    xgb = pytest.importorskip("xgboost")
    lgb = pytest.importorskip("lightgbm")
    cb = pytest.importorskip("catboost")

    def xgb_cls(**kw):
        return _Fitted(seen, "xgb", trained_device, **kw)

    def cb_cls(**kw):
        return _Fitted(seen, "catboost", **kw)

    def lgb_train(p, ds):
        seen["lgbm"] = p
        return _Fitted(seen, "lgbm_model")

    monkeypatch.setattr(xgb, "XGBRegressor", xgb_cls)
    monkeypatch.setattr(xgb, "XGBClassifier", xgb_cls)
    monkeypatch.setattr(lgb, "train", lgb_train)
    monkeypatch.setattr(cb, "CatBoostRegressor", cb_cls)
    monkeypatch.setattr(cb, "CatBoostClassifier", cb_cls)


@pytest.mark.parametrize("kind", ["r2", "auc"])
def test_cuda_reference_arms_place_their_library_on_the_gpu(monkeypatch, kind):
    """The device sweep ranks like for like only if every reference arm
    trains on the GPU: xgboost by device=, lightgbm by device_type=,
    catboost by task_type= with the GPU border cap applied."""
    seen = {}
    _stub_references(monkeypatch, seen)
    for arm in ("xgb_cuda", "lgbm_cuda", "catboost_gpu"):
        grinsztajn.fit_predict(arm, X, Y, X, kind)
    assert seen["xgb"]["device"] == "cuda"
    assert seen["lgbm"]["device_type"] == "cuda"
    assert (seen["catboost"]["task_type"], seen["catboost"]["devices"]) == ("GPU", "0")
    assert seen["catboost"]["border_count"] == 254


def test_cpu_reference_arms_stay_on_the_host(monkeypatch):
    seen = {}
    _stub_references(monkeypatch, seen)
    for arm in ("xgb", "lgbm", "catboost"):
        grinsztajn.fit_predict(arm, X, Y, X, "r2")
    assert seen["xgb"]["device"] == "cpu"
    assert seen["lgbm"]["device_type"] == "cpu"
    assert seen["catboost"]["task_type"] == "CPU"


def test_a_cuda_xgb_arm_that_fell_back_to_cpu_fails_the_row(monkeypatch):
    """The same post-fit guard the perf runner carries: xgboost 3.3 drops an
    unserved cuda request to CPU without raising, and a quality row born
    from that fallback would rank a CPU fit under a GPU label."""
    seen = {}
    _stub_references(monkeypatch, seen, trained_device="cpu")
    with pytest.raises(RuntimeError, match="fell back to CPU"):
        grinsztajn.fit_predict("xgb_cuda", X, Y, X, "r2")
    grinsztajn.fit_predict("xgb", X, Y, X, "r2")


def _stub_bonsai(monkeypatch, seen):
    import bonsai

    def regressor(**kw):
        return _Fitted(seen, "bonsai", **kw)

    monkeypatch.setattr(bonsai, "BonsaiRegressor", regressor)


def test_the_leaf_capped_regime_holds_the_leaves_and_lifts_the_depth(monkeypatch):
    """lightgbm's CUDA learner caps a tree by its leaf count alone, so the
    head to head hands both learners the campaign's 63 leaves under a depth
    cap a 63-leaf tree cannot reach; the campaign regime keeps depth 6."""
    seen = {}
    _stub_bonsai(monkeypatch, seen)
    _stub_references(monkeypatch, seen)
    for arm in ("bonsai_lw", "lgbm"):
        grinsztajn.fit_predict(arm, X, Y, X, "r2", grinsztajn.LEAF_CAPPED.knobs)
    assert (seen["bonsai"]["max_depth"], seen["bonsai"]["max_leaves"]) == (62, 63)
    assert (seen["lgbm"]["max_depth"], seen["lgbm"]["num_leaves"]) == (62, 63)
    for arm in ("bonsai_lw", "lgbm"):
        grinsztajn.fit_predict(arm, X, Y, X, "r2")
    assert (seen["bonsai"]["max_depth"], seen["bonsai"]["max_leaves"]) == (6, 63)
    assert (seen["lgbm"]["max_depth"], seen["lgbm"]["num_leaves"]) == (6, 63)


def test_the_leaf_capped_regime_runs_only_the_learners_a_leaf_count_caps():
    """A depthwise or symmetric tree at depth 62 is a different experiment,
    not a matched one, so the regime's arms are leafwise bonsai and
    lightgbm on each device, and the CLI reaches them by name."""
    assert grinsztajn.LEAF_CAPPED.arms == {"cpu": ("bonsai_lw", "lgbm"),
                                          "cuda": ("bonsai_cuda_leafwise", "lgbm_cuda")}
    assert grinsztajn.LEAF_CAPPED.knobs == dict(grinsztajn.C, depth=62)
    args = grinsztajn.parse_args(["--device", "cuda", "--regime", "leaf-capped", "o.jsonl"])
    assert (args.device, args.regime) == ("cuda", "leaf-capped")
    assert grinsztajn.parse_args(["o.jsonl"]).regime == "campaign"
