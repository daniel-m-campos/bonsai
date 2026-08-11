"""Tests for bonsai.estimators (BonsaiRegressor / BonsaiClassifier)."""

from __future__ import annotations

import pathlib
import pickle
import subprocess
import sys
import tempfile

import bonsai
import numpy as np
import pytest
from conftest import CH_PARAMS, TEST_CSV, TRAIN_CSV, load_csv

# The sklearn contract (clone, cross-validation, pickling, get/set_params) is
# the same promise for both estimators, so those tests run against each.
ESTIMATORS = [
    pytest.param(bonsai.BonsaiRegressor, id="regressor"),
    pytest.param(bonsai.BonsaiClassifier, id="classifier"),
]


def _fit_data(cls):
    """A small train set the estimator type can fit: the housing regression
    CSV, or the separable binary blobs for the classifier."""
    if cls is bonsai.BonsaiClassifier:
        return _separable_binary()
    Xtr, ytr = load_csv(TRAIN_CSV)
    return Xtr[:600], ytr[:600]


def test_fit_predict_rmse():
    Xtr, ytr = load_csv(TRAIN_CSV)
    Xte, yte = load_csv(TEST_CSV)
    m = bonsai.BonsaiRegressor(**CH_PARAMS).fit(Xtr, ytr)
    rmse = float(np.sqrt(np.mean((m.predict(Xte) - yte) ** 2)))
    assert rmse < 0.50, rmse  # CLI depthwise lands ~0.474


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
    with pytest.raises(Exception) as e:
        bonsai.BonsaiRegressor(n_iters=5).fit(X, y, sample_weight=np.ones(9, dtype=np.float32))
        assert "sample_weight" in str(e.value)


def test_early_stopping_stops():
    Xtr, ytr = load_csv(TRAIN_CSV)
    m = bonsai.BonsaiRegressor(
        n_iters=400, learning_rate=0.3, early_stopping_rounds=10
    ).fit(Xtr[:-2000], ytr[:-2000], eval_set=(Xtr[-2000:], ytr[-2000:]))
    assert m.n_iters_ < 400


_EARLY_STOPPING_FIT = """
import numpy as np

import bonsai

rng = np.random.default_rng(0)
X = rng.random((2000, 4), dtype=np.float32)
y = rng.random(2000, dtype=np.float32)
model = bonsai.BonsaiRegressor(
    n_iters=200, learning_rate=0.5, early_stopping_rounds=2
).fit(X[:1500], y[:1500], eval_set=(X[1500:], y[1500:]))
assert model.n_iters_ < 200, model.n_iters_
"""


def test_early_stopping_fit_writes_nothing_to_stdout():
    """The native fit must not write to the process stdout. A subprocess is the
    only honest check: the C runtime's stdout is not python's, so it escapes
    capsys and contextlib.redirect_stdout alike."""
    done = subprocess.run(
        [sys.executable, "-c", _EARLY_STOPPING_FIT], capture_output=True, check=True
    )
    assert done.stdout == b""


@pytest.mark.parametrize(
    "knob,value", [("data.missing_sentinel", -999.0), ("data.missing_nan", False)]
)
def test_retired_missing_knobs_are_unknown_keys(knob, value):
    """NaN is the missing marker and nothing configures it, so both knobs are
    gone; a config that still sets one must fail rather than be ignored (the
    sentinel used to train -999 as an ordinary low value on array input)."""
    rng = np.random.default_rng(1)
    X = rng.random((200, 3), dtype=np.float32)
    y = X[:, 0].copy()
    with pytest.raises(Exception, match=knob.split(".")[1]):
        bonsai.BonsaiRegressor(n_iters=5, params={knob: value}).fit(X, y)


def test_nan_rows_train_as_missing():
    """NaN is the array path's only missing marker, unchanged: with the target
    set to the missingness itself, the learned default branch must separate the
    NaN rows from the rest."""
    rng = np.random.default_rng(3)
    n = 4000
    X = rng.random((n, 1), dtype=np.float32)
    missing = rng.random(n) < 0.5
    X[missing, 0] = np.nan
    y = missing.astype(np.float32)

    pred = bonsai.BonsaiRegressor(n_iters=60, learning_rate=0.3, max_depth=3).fit(
        X, y
    ).predict(X)
    assert pred[missing].mean() > 0.9
    assert pred[~missing].mean() < 0.1


def test_bad_param_raises():
    with pytest.raises(RuntimeError) as e:
        bonsai.BonsaiRegressor(params={"tree.nope": 1}).fit(
            np.zeros((4, 2), dtype=np.float32), np.zeros(4, dtype=np.float32)
        )
        assert "nope" in str(e.value)


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

    with pytest.raises(RuntimeError):
        bonsai.train([], Xtr[:200], ytr[:200], config="/nonexistent/cfg.toml")


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


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_get_set_params_round_trip(cls):
    est = cls(n_iters=17, learning_rate=0.2, max_depth=4)
    params = est.get_params()
    assert params["n_iters"] == 17
    assert params["learning_rate"] == 0.2
    assert params["max_depth"] == 4
    # only the regressor has an objective knob; the classifier derives it
    # from the number of classes at fit time
    assert ("objective" in params) == (cls is bonsai.BonsaiRegressor)

    clone_est = type(est)(**est.get_params())
    assert clone_est.get_params() == params

    est.set_params(n_iters=99)
    assert est.n_iters == 99
    assert est.get_params()["n_iters"] == 99

    with pytest.raises(ValueError, match="not_a_real_param"):
        est.set_params(not_a_real_param=1)


def test_score_r2_matches_hand_computation():
    Xtr, ytr = load_csv(TRAIN_CSV)
    Xte, yte = load_csv(TEST_CSV)
    m = bonsai.BonsaiRegressor(n_iters=50).fit(Xtr, ytr)
    pred = m.predict(Xte)

    ss_res = np.sum((yte - pred) ** 2)
    ss_tot = np.sum((yte - yte.mean()) ** 2)
    expected = 1.0 - ss_res / ss_tot

    assert abs(m.score(Xte, yte) - expected) < 1e-9


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_sklearn_clone(cls):
    sklearn_base = pytest.importorskip("sklearn.base")

    est = cls(n_iters=17, learning_rate=0.2, max_depth=4)
    cloned = sklearn_base.clone(est)
    assert cloned is not est
    assert cloned.get_params() == est.get_params()
    assert cloned._model is None


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_sklearn_cross_val_score(cls):
    model_selection = pytest.importorskip("sklearn.model_selection")

    X, y = _fit_data(cls)
    scores = model_selection.cross_val_score(cls(n_iters=20), X, y, cv=3)
    assert len(scores) == 3
    assert all(np.isfinite(scores))


def test_sklearn_grid_search_cv():
    model_selection = pytest.importorskip("sklearn.model_selection")

    Xtr, ytr = load_csv(TRAIN_CSV)
    gs = model_selection.GridSearchCV(
        bonsai.BonsaiRegressor(), {"n_iters": [10, 20]}, cv=2
    )
    gs.fit(Xtr[:400], ytr[:400])
    assert gs.best_params_["n_iters"] in (10, 20)


def test_sklearn_pipeline():
    pipeline = pytest.importorskip("sklearn.pipeline")
    preprocessing = pytest.importorskip("sklearn.preprocessing")

    Xtr, ytr = load_csv(TRAIN_CSV)
    pipe = pipeline.Pipeline([
        ("sc", preprocessing.StandardScaler()),
        ("gb", bonsai.BonsaiRegressor(n_iters=20)),
    ])
    pipe.fit(Xtr[:400], ytr[:400])
    pred = pipe.predict(Xtr[:400])
    assert pred.shape == (400,)
    assert np.all(np.isfinite(pred))


@pytest.mark.parametrize("cls", ESTIMATORS)
def test_pickle_round_trip_fitted(cls):
    X, y = _fit_data(cls)
    m = cls(n_iters=20).fit(X, y)
    before = m.predict(X)

    restored = pickle.loads(pickle.dumps(m))
    after = restored.predict(X)

    assert np.array_equal(before, after)
    assert restored.n_iters_ == m.n_iters_
    if cls is bonsai.BonsaiClassifier:
        np.testing.assert_array_equal(restored.classes_, m.classes_)


def test_pickle_round_trip_unfitted():
    m = bonsai.BonsaiRegressor(n_iters=20, max_depth=3)
    restored = pickle.loads(pickle.dumps(m))
    assert restored.get_params() == m.get_params()
    assert restored._model is None


def _separable_binary(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    logits = 3.0 * X[:, 0] - 2.0 * X[:, 1]
    y = (logits > 0).astype(np.float32)
    return X, y


def _blobs_multiclass(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [5.0, 5.0], [5.0, -5.0]], dtype=np.float32)
    labels = rng.integers(0, 3, size=n)
    X = (centers[labels] + rng.normal(scale=0.6, size=(n, 2))).astype(np.float32)
    return X, labels.astype(np.float32)


def test_classifier_binary_predict_and_proba():
    X, y = _separable_binary()
    m = bonsai.BonsaiClassifier(n_iters=50).fit(X, y)

    pred = m.predict(X)
    assert set(np.unique(pred)) <= set(m.classes_)

    proba = m.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    acc = m.score(X, y)
    assert acc > 0.9, acc


def test_classifier_non_01_labels():
    X, y01 = _separable_binary()

    # integer labels not starting at 0
    y_int = np.where(y01 == 0, 10, 20).astype(np.int64)
    m_int = bonsai.BonsaiClassifier(n_iters=50).fit(X, y_int)
    np.testing.assert_array_equal(m_int.classes_, [10, 20])
    pred_int = m_int.predict(X)
    assert set(np.unique(pred_int)) <= {10, 20}

    # string labels
    y_str = np.where(y01 == 0, "a", "b")
    m_str = bonsai.BonsaiClassifier(n_iters=50).fit(X, y_str)
    np.testing.assert_array_equal(m_str.classes_, ["a", "b"])
    pred_str = m_str.predict(X)
    assert set(np.unique(pred_str)) <= {"a", "b"}


def test_classifier_multiclass_predict_and_proba():
    X, y = _blobs_multiclass()
    m = bonsai.BonsaiClassifier(n_iters=50).fit(X, y)

    assert m.n_classes_ == 3
    pred = m.predict(X)
    assert set(np.unique(pred)) <= set(m.classes_)
    assert m.score(X, y) > 0.8

    proba = m.predict_proba(X)
    assert proba.shape == (len(X), 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0).all() and (proba <= 1).all()
    # argmax column maps back to the predicted label via classes_
    assert np.array_equal(m.classes_[proba.argmax(axis=1)], pred)


def _dumped_trees(text: str) -> list[list[int]]:
    """[round, class, leaf count] for every tree block of a softmax dump."""
    trees: list[list[int]] = []
    for line in text.splitlines():
        if line.startswith("tree "):
            parts = line.split()
            trees.append([int(parts[1]), int(parts[3].rstrip(":")), 0])
        elif "leaf=" in line:
            trees[-1][2] += 1
    return trees


def test_classifier_multiclass_predict_leaf_one_column_per_tree():
    """Softmax grows one tree per class per round, so the leaf matrix is
    n_iters * n_classes wide. Sizing it by n_iters alone overran the output
    buffer and corrupted the heap."""
    n_rows, n_classes, n_iters = 400, 5, 8
    rng = np.random.default_rng(0)
    y = rng.integers(0, n_classes, size=n_rows)
    X = (y[:, None] * 3.0 + rng.normal(scale=0.5, size=(n_rows, 3))).astype(np.float32)
    m = bonsai.BonsaiClassifier(n_iters=n_iters, max_depth=3).fit(X, y)

    leaves = m.predict_leaf(X)

    assert m.n_classes_ == n_classes
    assert leaves.shape == (n_rows, n_iters * n_classes)

    trees = _dumped_trees(m.dump())
    assert len(trees) == leaves.shape[1]
    for t, (round_id, class_id, n_leaves) in enumerate(trees):
        assert (round_id, class_id) == (t // n_classes, t % n_classes)
        column = leaves[:, t]
        assert column.min() >= 0
        assert len(np.unique(column)) <= n_leaves


def test_classifier_too_few_classes_raises():
    X = np.zeros((10, 3), dtype=np.float32)
    y = np.zeros(10, dtype=np.float32)
    with pytest.raises(ValueError) as e:
        bonsai.BonsaiClassifier(n_iters=5).fit(X, y)
        assert "class" in str(e.value).lower()


def test_predict_proba_rejects_regression_objective():
    Xtr, ytr = load_csv(TRAIN_CSV)
    m = bonsai.train([("booster.n_iters", "5")], Xtr[:500], ytr[:500])
    with pytest.raises(Exception) as e:
        m.predict_proba(Xtr[:10])
        assert "classification" in str(e.value) and "mse" in str(e.value)


def test_classifier_score_zero_weights_raise():
    X, y = _three_class_data(300)
    clf = bonsai.BonsaiClassifier(n_iters=5).fit(X, y)
    with pytest.raises(ValueError) as e:
        clf.score(X, y, sample_weight=np.zeros(len(y), dtype=np.float32))
        assert "zero" in str(e.value)


def _three_class_data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n, 6), dtype=np.float32)
    y = np.digitize(X[:, 0] + 0.3 * rng.random(n), [0.45, 0.85]).astype(np.float32)
    return X, y


def test_multiclass_sample_weight_applied():
    """The softmax booster must scale grad/hess by sample_weight (it silently
    ignored it before): upweighting one class shifts probability mass toward
    it, and all-ones weights stay bit-identical to no weights."""
    X, y = _three_class_data()
    params = dict(n_iters=30, learning_rate=0.1, max_depth=4)

    base = bonsai.BonsaiClassifier(**params).fit(X, y)
    ones = bonsai.BonsaiClassifier(**params).fit(
        X, y, sample_weight=np.ones(len(y), dtype=np.float32)
    )
    np.testing.assert_array_equal(base.predict_proba(X), ones.predict_proba(X))

    w = np.where(y == 0, 20.0, 1.0).astype(np.float32)
    up = bonsai.BonsaiClassifier(**params).fit(X, y, sample_weight=w)
    p_base = base.predict_proba(X)[:, 0].mean()
    p_up = up.predict_proba(X)[:, 0].mean()
    assert p_up > p_base + 0.05, (p_base, p_up)


def test_classifier_from_file_restores_class_metadata():
    """from_file used to crash predict/predict_proba (classes_/n_classes_ were
    only set by fit). It now restores them as encoded ids 0..K-1."""
    X, y = _three_class_data()
    for labels in (y, (y * 3 + 1)):  # encoded ids differ from raw labels
        m = bonsai.BonsaiClassifier(n_iters=10).fit(X, labels)
        with tempfile.TemporaryDirectory() as td:
            path = str(pathlib.Path(td) / "clf.msgpack")
            m.save(path)
            restored = bonsai.BonsaiClassifier.from_file(path)
        assert restored.n_classes_ == 3
        assert np.array_equal(restored.classes_, np.arange(3))
        # same decisions, in encoded-id space
        expected = np.searchsorted(m.classes_, m.predict(X))
        assert np.array_equal(restored.predict(X), expected)
        assert restored.predict_proba(X).shape == (len(X), 3)

    # binary round-trips too
    yb = (y > 0).astype(np.float32)
    m = bonsai.BonsaiClassifier(n_iters=10).fit(X, yb)
    with tempfile.TemporaryDirectory() as td:
        path = str(pathlib.Path(td) / "clf.msgpack")
        m.save(path)
        restored = bonsai.BonsaiClassifier.from_file(path)
    assert restored.n_classes_ == 2
    assert np.array_equal(restored.predict(X), m.predict(X).astype(np.int64))

    # a regression model is not a classifier: refuse rather than mislabel
    Xr, yr = load_csv(TRAIN_CSV)
    with tempfile.TemporaryDirectory() as td:
        path = str(pathlib.Path(td) / "reg.msgpack")
        bonsai.BonsaiRegressor(n_iters=5).fit(Xr[:500], yr[:500]).save(path)
        with pytest.raises(ValueError) as e:
            bonsai.BonsaiClassifier.from_file(path)
            assert "objective" in str(e.value)


def test_classifier_eval_set_unseen_label_raises():
    """Labels the training fold never saw cannot be encoded; silently
    mis-encoding them corrupted the eval metric and early stopping."""
    X, y = _three_class_data()
    Xv, yv = X[:200].copy(), y[:200].copy()
    for bad_label in (5.0, 0.5, -3.0):  # out of range and in-between
        yv_bad = yv.copy()
        yv_bad[:10] = bad_label
        with pytest.raises(ValueError) as e:
            bonsai.BonsaiClassifier(n_iters=5).fit(X, y, eval_set=(Xv, yv_bad))
            assert "eval_set" in str(e.value)
    # a valid eval_set still works
    bonsai.BonsaiClassifier(n_iters=5).fit(X, y, eval_set=(Xv, yv))


def test_model_n_classes_semantics():
    """n_classes is 0 unless the model is softmax; the config-struct default
    of 3 must never leak out of a regression or binary model."""
    X, y = _three_class_data(500)
    m3 = bonsai.BonsaiClassifier(n_iters=3).fit(X, y)._model
    assert m3.objective_name == "softmax" and m3.n_classes == 3
    yb = (y > 0).astype(np.float32)
    mb = bonsai.BonsaiClassifier(n_iters=3).fit(X, yb)._model
    assert mb.objective_name == "logloss" and mb.n_classes == 0
    mr = bonsai.train([("booster.n_iters", "3")], X, y)
    assert mr.objective_name == "mse" and mr.n_classes == 0


def test_classifier_nan_labels_raise():
    X, y = _three_class_data()
    y = y.copy()
    y[0] = np.nan
    with pytest.raises(ValueError) as e:
        bonsai.BonsaiClassifier(n_iters=5).fit(X, y)
        assert "NaN" in str(e.value)
