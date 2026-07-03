"""Binding tests: run with  PYTHONPATH=build/python .venv/bin/python -m pytest
python/tests -q  (or plain `python python/tests/test_bindings.py`)."""

from __future__ import annotations

import pathlib
import subprocess
import tempfile

import numpy as np

import bonsai

REPO = pathlib.Path(__file__).resolve().parents[2]
TRAIN_CSV = REPO / "tests/data/california_housing_train.csv"
TEST_CSV = REPO / "tests/data/california_housing_test.csv"
CH_TOML = REPO / "configs/california_housing.toml"
CLI = REPO / "build/src/bonsai"

CH_PARAMS = dict(
    n_iters=200,
    learning_rate=0.05,
    max_depth=6,
    grower="depthwise",
    params={
        "tree.min_data_in_leaf": 20,
        "tree.min_child_hess": 0.001,
        "bin_mapper.max_bin": 255,
    },
)


def load_csv(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    return data[:, 1:], data[:, 0]  # label is column 0


def test_fit_predict_rmse():
    Xtr, ytr = load_csv(TRAIN_CSV)
    Xte, yte = load_csv(TEST_CSV)
    m = bonsai.BonsaiRegressor(**CH_PARAMS).fit(Xtr, ytr)
    rmse = float(np.sqrt(np.mean((m.predict(Xte) - yte) ** 2)))
    assert rmse < 0.50, rmse  # CLI depthwise lands ~0.474


def test_parity_with_cli():
    """Same config through the native module and the CLI must agree."""
    Xtr, ytr = load_csv(TRAIN_CSV)
    Xte, _ = load_csv(TEST_CSV)
    py_pred = bonsai.BonsaiRegressor(**CH_PARAMS).fit(Xtr, ytr).predict(Xte)

    with tempfile.TemporaryDirectory() as td:
        model = pathlib.Path(td) / "m.msgpack"
        preds = pathlib.Path(td) / "p.csv"
        subprocess.run(
            [CLI, "fit", "-c", CH_TOML, "--set", "dispatch.grower_name=depthwise",
             "--model", model],
            check=True, capture_output=True,
        )
        subprocess.run(
            [CLI, "predict", "-c", CH_TOML, "--model", model, "--out", preds],
            check=True, capture_output=True,
        )
        cli_pred = np.loadtxt(preds, skiprows=1, dtype=np.float32)

    np.testing.assert_allclose(py_pred, cli_pred, rtol=0, atol=2e-4)


def test_early_stopping_stops():
    Xtr, ytr = load_csv(TRAIN_CSV)
    m = bonsai.BonsaiRegressor(
        n_iters=400, learning_rate=0.3, early_stopping_rounds=10
    ).fit(Xtr[:-2000], ytr[:-2000], eval_set=(Xtr[-2000:], ytr[-2000:]))
    assert m.n_iters_ < 400


def test_bad_param_raises():
    try:
        bonsai.BonsaiRegressor(params={"tree.nope": 1}).fit(
            np.zeros((4, 2), dtype=np.float32), np.zeros(4, dtype=np.float32)
        )
    except RuntimeError as e:
        assert "nope" in str(e)
    else:
        raise AssertionError("expected a config error")


def test_feature_importance_agreement():
    """bonsai and lightgbm agree on California Housing for BOTH importance
    types — including their disagreement with each other: gain crowns
    MedInc (feature 0), while split-count crowns geography (lat/long,
    features 6/7), the textbook example of why gain is the better default."""
    import lightgbm as lgb

    Xtr, ytr = load_csv(TRAIN_CSV)
    m = bonsai.BonsaiRegressor(**CH_PARAMS).fit(Xtr, ytr)

    gain = m.importance("gain")
    split = m.importance("split")
    assert gain.shape == (Xtr.shape[1],)
    assert int(np.argmax(gain)) == 0, gain
    assert int(np.argmax(split)) in (6, 7), split

    fi = m.feature_importances_
    assert abs(float(fi.sum()) - 1.0) < 1e-6
    assert int(np.argmax(fi)) == 0

    lgbm = lgb.train(
        {"objective": "regression", "verbose": -1, "max_bin": 255},
        lgb.Dataset(Xtr, label=ytr), num_boost_round=200,
    )
    assert int(np.argmax(lgbm.feature_importance("gain"))) == 0
    assert int(np.argmax(lgbm.feature_importance("split"))) in (6, 7)


def test_pred_contribs_efficiency():
    Xtr, ytr = load_csv(TRAIN_CSV)
    m = bonsai.BonsaiRegressor(n_iters=50).fit(Xtr, ytr)
    c = m.pred_contribs(Xtr[:100])
    p = m.predict(Xtr[:100])
    assert c.shape == (100, Xtr.shape[1] + 1)
    np.testing.assert_allclose(c.sum(axis=1), p, atol=1e-3)


if __name__ == "__main__":
    test_fit_predict_rmse()
    test_parity_with_cli()
    test_early_stopping_stops()
    test_bad_param_raises()
    test_feature_importance_agreement()
    test_pred_contribs_efficiency()
    print("all binding tests passed")
