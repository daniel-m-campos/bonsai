"""Tests for scripts/compare.py, the four-library comparison harness."""

from __future__ import annotations

import dataclasses
import pathlib
import sys
import types

import numpy as np
import pytest

pytest.importorskip("tomllib")
pd = pytest.importorskip("pandas")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import compare  # noqa: E402

DEFAULT_HP = compare.HP(
    n_iters=7,
    learning_rate=0.05,
    max_depth=4,
    max_leaves=9,
    min_data_in_leaf=20,
    lambda_l2=1.0,
    lambda_l1=0.0,
    feature_fraction=1.0,
    early_stopping_rounds=0,
    dart_drop_rate=0.0,
    objective="mse",
    n_classes=3,
    huber_delta=1.0,
    quantile_alpha=0.5,
    monotone_constraints=[],
    interaction_constraints=[],
    max_bin=63,
    random_seed=42,
)


def _hp(**over) -> compare.HP:
    return dataclasses.replace(DEFAULT_HP, **over)


class _Recorder:
    """A lightgbm stand-in that records what the arm hands it."""

    def __init__(self, n_classes: int = 1):
        self.datasets: list[dict] = []
        self.train_calls: list[dict] = []
        self.stops: list[int] = []
        self.n_classes = n_classes

    def Dataset(self, data, label=None, reference=None):
        self.datasets.append({"rows": len(data), "reference": reference})
        return ("dataset", len(self.datasets))

    def early_stopping(self, rounds, verbose):
        self.stops.append(rounds)
        return ("stop", rounds)

    def train(self, params, dtrain, num_boost_round, **fit_kwargs):
        self.train_calls.append(
            {"params": params, "dtrain": dtrain, "rounds": num_boost_round, **fit_kwargs}
        )
        n = self.n_classes
        return types.SimpleNamespace(
            predict=lambda df: np.zeros((len(df), n)) if n > 1 else np.zeros(len(df))
        )


def _frames(n: int = 12, cols: int = 3):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.random((n, cols)), columns=[f"f{i}" for i in range(cols)])
    df[compare.LABEL_COL] = rng.integers(0, 2, n).astype(float)
    return df.iloc[: n // 2], df.iloc[n // 2 :]


def _lgbm(monkeypatch, hp: compare.HP, valid_df=None, n_classes: int = 1) -> dict:
    rec = _Recorder(n_classes)
    monkeypatch.setitem(sys.modules, "lightgbm", rec)
    train_df, test_df = _frames()
    result = compare.run_lightgbm(train_df, test_df, hp, valid_df=valid_df)
    assert len(rec.train_calls) == 1
    return {"call": rec.train_calls[0], "rec": rec, "result": result}


def test_lightgbm_params_are_the_mapped_knobs_and_nothing_else(monkeypatch):
    call = _lgbm(monkeypatch, _hp())["call"]
    assert call["params"] == {
        "objective": "regression",
        "metric": "rmse",
        **compare.reference_params.lgbm_core(
            learning_rate=0.05,
            max_depth=4,
            num_leaves=9,
            min_data_in_leaf=20,
            lambda_l2=1.0,
            max_bin=63,
            seed=42,
        ),
        "lambda_l1": 0.0,
        "feature_fraction": 1.0,
    }
    assert call["rounds"] == 7
    assert "valid_sets" not in call and "callbacks" not in call


def test_lightgbm_params_carry_the_optional_knobs_when_set(monkeypatch):
    hp = _hp(
        objective="huber",
        huber_delta=0.9,
        dart_drop_rate=0.1,
        monotone_constraints=[1, -1],
        interaction_constraints=[[0, 1], [2]],
        lambda_l1=0.5,
        feature_fraction=0.8,
    )
    params = _lgbm(monkeypatch, hp)["call"]["params"]
    assert params["objective"] == "huber" and params["alpha"] == 0.9
    assert params["metric"] == "rmse"
    assert params["boosting"] == "dart" and params["drop_rate"] == 0.1
    assert params["monotone_constraints"] == [1, -1, 0]
    assert params["interaction_constraints"] == [[0, 1], [2]]
    assert params["lambda_l1"] == 0.5 and params["feature_fraction"] == 0.8
    assert "num_class" not in params


def test_lightgbm_softmax_fills_the_leaf_budget_from_depth(monkeypatch):
    hp = _hp(objective="softmax", max_leaves=0, max_depth=5, n_classes=3)
    params = _lgbm(monkeypatch, hp, n_classes=3)["call"]["params"]
    assert params["objective"] == "multiclass"
    assert params["num_class"] == 3
    assert params["metric"] == "multi_logloss"
    assert params["num_leaves"] == (1 << 5) - 1


def test_lightgbm_early_stops_only_with_a_valid_frame_and_rounds(monkeypatch):
    train_df, _ = _frames()
    got = _lgbm(monkeypatch, _hp(early_stopping_rounds=3), valid_df=train_df)
    call, rec = got["call"], got["rec"]
    assert call["valid_sets"] == [("dataset", 2)]
    assert call["callbacks"] == [("stop", 3)]
    assert rec.datasets[1]["reference"] == ("dataset", 1)
    without = _lgbm(monkeypatch, _hp(early_stopping_rounds=0), valid_df=train_df)["call"]
    assert "valid_sets" not in without


def test_lightgbm_scores_the_arm_with_the_shared_metric_wrappers(monkeypatch):
    """The Result every arm returns is the six wrappers over (pred, y) plus the
    two timers; a softmax arm scores the argmax label and reports accuracy."""
    got = _lgbm(monkeypatch, _hp())
    _, test_df = _frames()
    y = test_df[compare.LABEL_COL].to_numpy()
    zeros = np.zeros(len(y))
    res = got["result"]
    assert (res.rmse, res.mae, res.r2, res.auc, res.acc) == pytest.approx(
        (
            compare.rmse(zeros, y),
            compare.mae(zeros, y),
            compare.r2(zeros, y),
            compare.maybe_auc(zeros, y),
            compare.maybe_acc(zeros, y, False),
        ),
        nan_ok=True,
    )
    assert res.fit_seconds >= 0 and res.predict_seconds >= 0
    softmax = _lgbm(monkeypatch, _hp(objective="softmax", n_classes=3), n_classes=3)["result"]
    assert softmax.acc == pytest.approx(compare.maybe_acc(zeros, y, True))
