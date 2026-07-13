"""Binding tests: run with  PYTHONPATH=build/python .venv/bin/python -m pytest
python/tests -q  (or plain `python python/tests/test_bindings.py`)."""

from __future__ import annotations

import pathlib
import pickle
import subprocess
import tempfile

import bonsai
import numpy as np

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


def test_sample_weight_ones_is_identity():
    """Uniform weights of 1.0 multiply every gradient/hessian by 1 — the model
    must be identical to fitting with no weights at all."""
    Xtr, ytr = load_csv(TRAIN_CSV)
    Xte, _ = load_csv(TEST_CSV)
    base = bonsai.BonsaiRegressor(**CH_PARAMS).fit(Xtr, ytr).predict(Xte)
    ones = np.ones(len(ytr), dtype=np.float32)
    weighted = (
        bonsai.BonsaiRegressor(**CH_PARAMS).fit(Xtr, ytr, sample_weight=ones).predict(Xte)
    )
    np.testing.assert_array_equal(base, weighted)


def test_sample_weight_shifts_toward_upweighted_rows():
    """With uninformative features the tree can only fit the (weighted) mean,
    so heavily upweighting the high-target rows must raise predictions."""
    rng = np.random.default_rng(0)
    n = 4000
    X = rng.random((n, 5), dtype=np.float32)  # pure noise, no signal
    y = np.concatenate([np.zeros(n // 2), np.full(n // 2, 10.0)]).astype(np.float32)
    params = dict(n_iters=50, learning_rate=0.1, max_depth=4)

    uniform = bonsai.BonsaiRegressor(**params).fit(X, y).predict(X).mean()
    w = np.where(y > 5, 100.0, 1.0).astype(np.float32)  # upweight the tens
    up = bonsai.BonsaiRegressor(**params).fit(X, y, sample_weight=w).predict(X).mean()

    assert uniform < 6.0  # ~weighted-uniform mean, near 5
    assert up > 9.0  # dragged toward the upweighted target of 10
    assert up > uniform + 3.0


def test_sample_weight_length_mismatch_raises():
    X = np.zeros((10, 3), dtype=np.float32)
    y = np.zeros(10, dtype=np.float32)
    try:
        bonsai.BonsaiRegressor(n_iters=5).fit(X, y, sample_weight=np.ones(9, dtype=np.float32))
    except Exception as e:
        assert "sample_weight" in str(e)
    else:
        raise AssertionError("expected a length-mismatch error")


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


def test_toml_config_base_and_precedence():
    """config= is the base (CLI -c); explicit params override it (--set)."""
    Xtr, ytr = load_csv(TRAIN_CSV)
    with tempfile.TemporaryDirectory() as td:
        toml = pathlib.Path(td) / "cfg.toml"
        toml.write_text("[booster]\nn_iters = 7\n")

        pairs = [("dispatch.grower_name", "depthwise")]
        m = bonsai.train(pairs, Xtr[:200], ytr[:200], config=str(toml))
        assert m.n_iters == 7

        pairs.append(("booster.n_iters", "3"))
        m = bonsai.train(pairs, Xtr[:200], ytr[:200], config=str(toml))
        assert m.n_iters == 3

        # BonsaiRegressor always emits its kwargs, so they win over the file.
        r = bonsai.BonsaiRegressor(n_iters=4, config=str(toml)).fit(
            Xtr[:200], ytr[:200]
        )
        assert r.n_iters_ == 4

    try:
        bonsai.train([], Xtr[:200], ytr[:200], config="/nonexistent/cfg.toml")
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected an error for a missing config file")


def test_cuda_available_reports():
    """False on CPU-only builds; on a make python-cuda build with a device,
    a cuda_* grower must actually train."""
    assert isinstance(bonsai.cuda_available(), bool)
    if bonsai.cuda_available():
        Xtr, ytr = load_csv(TRAIN_CSV)
        m = bonsai.BonsaiRegressor(
            n_iters=5, grower="cuda_depthwise"
        ).fit(Xtr[:1000], ytr[:1000])
        assert m.n_iters_ == 5


def test_get_set_params_round_trip():
    est = bonsai.BonsaiRegressor(n_iters=17, learning_rate=0.2, max_depth=4)
    params = est.get_params()
    assert params["n_iters"] == 17
    assert params["learning_rate"] == 0.2
    assert params["max_depth"] == 4

    clone_est = type(est)(**est.get_params())
    assert clone_est.get_params() == params

    est.set_params(n_iters=99)
    assert est.n_iters == 99
    assert est.get_params()["n_iters"] == 99

    try:
        est.set_params(not_a_real_param=1)
    except ValueError as e:
        assert "not_a_real_param" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown param")


def test_score_r2_matches_hand_computation():
    Xtr, ytr = load_csv(TRAIN_CSV)
    Xte, yte = load_csv(TEST_CSV)
    m = bonsai.BonsaiRegressor(n_iters=50).fit(Xtr, ytr)
    pred = m.predict(Xte)

    ss_res = np.sum((yte - pred) ** 2)
    ss_tot = np.sum((yte - yte.mean()) ** 2)
    expected = 1.0 - ss_res / ss_tot

    assert abs(m.score(Xte, yte) - expected) < 1e-9


def test_sklearn_clone():
    try:
        import sklearn.base
    except ImportError:
        return

    est = bonsai.BonsaiRegressor(n_iters=17, learning_rate=0.2, max_depth=4)
    cloned = sklearn.base.clone(est)
    assert cloned is not est
    assert cloned.get_params() == est.get_params()
    assert cloned._model is None


def test_sklearn_cross_val_score():
    try:
        import sklearn.model_selection
    except ImportError:
        return

    Xtr, ytr = load_csv(TRAIN_CSV)
    scores = sklearn.model_selection.cross_val_score(
        bonsai.BonsaiRegressor(n_iters=20), Xtr[:600], ytr[:600], cv=3
    )
    assert len(scores) == 3
    assert all(np.isfinite(scores))


def test_sklearn_grid_search_cv():
    try:
        import sklearn.model_selection
    except ImportError:
        return

    Xtr, ytr = load_csv(TRAIN_CSV)
    gs = sklearn.model_selection.GridSearchCV(
        bonsai.BonsaiRegressor(), {"n_iters": [10, 20]}, cv=2
    )
    gs.fit(Xtr[:400], ytr[:400])
    assert gs.best_params_["n_iters"] in (10, 20)


def test_sklearn_pipeline():
    try:
        import sklearn.pipeline
        import sklearn.preprocessing
    except ImportError:
        return

    Xtr, ytr = load_csv(TRAIN_CSV)
    pipe = sklearn.pipeline.Pipeline([
        ("sc", sklearn.preprocessing.StandardScaler()),
        ("gb", bonsai.BonsaiRegressor(n_iters=20)),
    ])
    pipe.fit(Xtr[:400], ytr[:400])
    pred = pipe.predict(Xtr[:400])
    assert pred.shape == (400,)
    assert np.all(np.isfinite(pred))


def test_pickle_round_trip_fitted():
    Xtr, ytr = load_csv(TRAIN_CSV)
    Xte, _ = load_csv(TEST_CSV)
    m = bonsai.BonsaiRegressor(n_iters=20).fit(Xtr, ytr)
    before = m.predict(Xte)

    restored = pickle.loads(pickle.dumps(m))
    after = restored.predict(Xte)

    assert np.array_equal(before, after)
    assert restored.n_iters_ == m.n_iters_


def test_pickle_round_trip_unfitted():
    m = bonsai.BonsaiRegressor(n_iters=20, max_depth=3)
    restored = pickle.loads(pickle.dumps(m))
    assert restored.get_params() == m.get_params()
    assert restored._model is None


if __name__ == "__main__":
    test_fit_predict_rmse()
    test_parity_with_cli()
    test_early_stopping_stops()
    test_bad_param_raises()
    test_feature_importance_agreement()
    test_pred_contribs_efficiency()
    test_toml_config_base_and_precedence()
    test_cuda_available_reports()
    test_get_set_params_round_trip()
    test_score_r2_matches_hand_computation()
    test_sklearn_clone()
    test_sklearn_cross_val_score()
    test_sklearn_grid_search_cv()
    test_sklearn_pipeline()
    test_pickle_round_trip_fitted()
    test_pickle_round_trip_unfitted()
    print("all binding tests passed")
