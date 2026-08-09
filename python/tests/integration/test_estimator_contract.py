"""The estimator layer end to end, in bonsai's own parameter vocabulary.

Covers what a full fit has to honor: the eval history and its early-stopping
attributes, warm starts, save and load, and the device-to-grower mapping.
Cross-library parameter names are no part of this contract; they live in
`bonsai.interop` and are tested there.

Run with  PYTHONPATH=build/python .venv/bin/python -m pytest python/tests -q
"""

from __future__ import annotations

import tempfile

import bonsai
import numpy as np
import pytest
from bonsai.estimators import _grower_for_device

RNG = np.random.default_rng(7)


def _reg_data(n=600, p=8):
    X = RNG.standard_normal((n, p)).astype(np.float32)
    y = X[:, 0] * 2.0 + np.sin(X[:, 1]) + 0.1 * RNG.standard_normal(n)
    return X[: n // 2], y[: n // 2].astype(np.float32), X[n // 2 :], y[n // 2 :].astype(np.float32)


def _cls_data(n=600, p=8, k=2):
    X = RNG.standard_normal((n, p)).astype(np.float32)
    y = (X[:, 0] + 0.3 * RNG.standard_normal(n) > 0).astype(np.int64) if k == 2 \
        else RNG.integers(0, k, n)
    return X[: n // 2], y[: n // 2], X[n // 2 :], y[n // 2 :]


def test_full_regressor_script():
    """Every first-class knob at once, then the whole post-fit surface."""
    Xtr, ytr, Xva, yva = _reg_data()
    est = bonsai.BonsaiRegressor(
        n_iters=80,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        objective="mse",
        random_seed=0,
        n_threads=2,
        early_stopping_rounds=10,
        params={"tree.feature_fraction": 0.8, "tree.min_child_hess": 1.0},
    )
    est.fit(Xtr, ytr, eval_set=(Xva, yva))
    assert est.n_features_in_ == Xtr.shape[1]

    hist = est.evals_result()["valid"]["mse"]
    assert len(hist) >= 1
    assert est.best_iteration == int(np.argmin(hist))
    assert abs(est.best_score - min(hist)) < 1e-12
    assert min(hist) < hist[0]

    pred = est.predict(Xva)
    assert pred.shape == yva.shape
    head = est.predict(Xva, num_iteration=5)
    assert not np.allclose(head, pred)
    leaves = est.predict_leaf(Xva)
    assert leaves.shape[0] == Xva.shape[0]
    assert est.feature_importances_.shape == (Xtr.shape[1],)


def test_eval_history_reports_the_native_objective_name():
    """The metric key is bonsai's objective, in its own units: mse is the
    squared error, not its root."""
    Xtr, ytr, Xva, yva = _reg_data()
    est = bonsai.BonsaiRegressor(n_iters=15).fit(Xtr, ytr, eval_set=(Xva, yva))
    result = est.evals_result()["valid"]
    assert list(result) == ["mse"]
    assert len(result["mse"]) == 15
    # best_iteration is an early-stopping concept, undefined without it
    with pytest.raises(AttributeError):
        _ = est.best_iteration


def test_eval_set_list_is_rejected():
    Xtr, ytr, Xva, yva = _reg_data()
    with pytest.raises(ValueError, match=r"one \(X, y\) tuple"):
        bonsai.BonsaiRegressor(n_iters=5).fit(Xtr, ytr, eval_set=[(Xva, yva)])


def test_save_from_file_roundtrip():
    Xtr, ytr, Xva, _ = _reg_data()
    est = bonsai.BonsaiRegressor(n_iters=20).fit(Xtr, ytr)
    with tempfile.NamedTemporaryFile(suffix=".msgpack") as f:
        est.save(f.name)
        out = bonsai.BonsaiRegressor.from_file(f.name)
        np.testing.assert_allclose(out.predict(Xva), est.predict(Xva))


def test_classifier_derives_its_objective_from_the_labels():
    Xtr, ytr, Xva, yva = _cls_data(k=2)
    est = bonsai.BonsaiClassifier(n_iters=30, random_seed=0)
    est.fit(Xtr, ytr, eval_set=(Xva, yva))
    proba = est.predict_proba(Xva)
    assert proba.shape == (len(Xva), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert list(est.evals_result()["valid"]) == ["logloss"]


def test_multiclass_reports_the_softmax_objective():
    Xtr, ytr, Xva, yva = _cls_data(k=3)
    est = bonsai.BonsaiClassifier(n_iters=25)
    est.fit(Xtr, ytr, eval_set=(Xva, yva))
    assert "softmax" in est.evals_result()["valid"]
    assert est.predict_proba(Xva).shape == (len(Xva), 3)


def test_regressor_quantile_objective():
    Xtr, ytr, Xva, _ = _reg_data()
    est = bonsai.BonsaiRegressor(
        n_iters=30, objective="quantile", quantile_alpha=0.9
    ).fit(Xtr, ytr)
    med = bonsai.BonsaiRegressor(
        n_iters=30, objective="quantile", quantile_alpha=0.5
    ).fit(Xtr, ytr)
    assert est.predict(Xva).mean() > med.predict(Xva).mean()


def test_warm_start_best_iteration_is_absolute():
    """best_iteration is an absolute index into the model, so after an
    init_model continuation it counts the warm-start rounds and
    predict(num_iteration=best_iteration + 1) uses the whole best model,
    not a continuation-relative prefix."""
    Xtr, ytr, Xva, yva = _reg_data()
    rounds_a = 30
    first = bonsai.BonsaiRegressor(n_iters=rounds_a).fit(Xtr, ytr)
    with tempfile.NamedTemporaryFile(suffix=".msgpack") as f:
        first.save(f.name)
        est = bonsai.BonsaiRegressor(n_iters=40, early_stopping_rounds=5)
        est.fit(Xtr, ytr, eval_set=(Xva, yva), init_model=f.name)

    assert est.best_iteration >= rounds_a
    assert est.best_iteration < est.n_iters_
    if est.n_iters_ < rounds_a + 40:  # early stopping fired and truncated
        assert est.n_iters_ == est.best_iteration + 1
    # evals_result reports only the measured continuation rounds; the best
    # absolute round is the warm-start offset plus its argmin
    cont = est.evals_result()["valid"]["mse"]
    assert len(cont) <= 40
    assert est.best_iteration == rounds_a + int(np.argmin(cont))
    assert abs(est.best_score - min(cont)) < 1e-12
    best = est.predict(Xva, num_iteration=est.best_iteration + 1)
    head = est.predict(Xva, num_iteration=4)
    assert not np.allclose(best, head)


def test_dart_with_eval_set_raises():
    """Eval history is unsupported under DART (its per-round rescaling
    invalidates the incremental valid-loss bookkeeping); passing eval_set
    must fail loudly at fit time instead of silently recording nothing."""
    Xtr, ytr, Xva, yva = _reg_data()
    est = bonsai.BonsaiRegressor(
        n_iters=10, params={"booster.dart_drop_rate": 0.1}
    )
    with pytest.raises(ValueError) as e:
        est.fit(Xtr, ytr, eval_set=(Xva, yva))
    assert "dart_drop_rate" in str(e.value)
    assert "eval_set" in str(e.value)
    # without an eval_set DART still fits normally
    est.fit(Xtr, ytr)
    assert est.n_iters_ == 10


def test_device_and_grower_mapping():
    assert _grower_for_device("leafwise", "cuda") == "cuda_leafwise"
    assert _grower_for_device("depthwise", "cuda") == "cuda_depthwise"
    assert _grower_for_device("levelwise", "cuda") == "cuda_levelwise"
    assert _grower_for_device("cuda_levelwise", "cpu") == "levelwise"
    assert _grower_for_device("depthwise", "cpu") == "depthwise"
    with pytest.raises(ValueError):
        _grower_for_device("leafwise", "tpu")
    if not bonsai.cuda_available():
        with np.testing.assert_raises(Exception):
            Xtr, ytr, _, _ = _reg_data(n=100)
            bonsai.BonsaiRegressor(n_iters=2, device="cuda").fit(Xtr, ytr)


def test_sklearn_clone_roundtrips_every_kwarg():
    from sklearn.base import clone

    est = bonsai.BonsaiClassifier(subsample=0.7, max_bin=63, device="cpu")
    twin = clone(est)
    assert twin.get_params() == est.get_params()

    reg = bonsai.BonsaiRegressor(objective="quantile", quantile_alpha=0.9)
    assert clone(reg).get_params() == reg.get_params()
