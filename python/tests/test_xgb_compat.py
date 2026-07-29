"""The xgboost drop-in contract: canonical XGBRegressor/XGBClassifier call
shapes must run against Bonsai estimators with only the class name swapped.

Run with  PYTHONPATH=build/python .venv/bin/python -m pytest python/tests -q
(or plain `python python/tests/test_xgb_compat.py`).
"""

from __future__ import annotations

import tempfile

import bonsai
import numpy as np

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


def test_canonical_xgb_regressor_script():
    """The most-copied xgboost pattern, class name swapped and nothing else."""
    Xtr, ytr, Xva, yva = _reg_data()
    est = bonsai.BonsaiRegressor(
        n_estimators=80,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1.0,
        gamma=0.0,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=0,
        n_jobs=2,
        early_stopping_rounds=10,
    )
    est.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    assert est.n_features_in_ == Xtr.shape[1]

    r = est.evals_result()
    hist = r["validation_0"]["rmse"]
    assert len(hist) >= 1
    assert est.best_iteration == int(np.argmin(hist))
    assert abs(est.best_score - min(hist)) < 1e-12
    # rmse presentation: the eval curve is the root of bonsai's mse values
    assert min(hist) < hist[0]

    pred = est.predict(Xva)
    assert pred.shape == yva.shape
    # iteration_range is xgboost's prediction-truncation spelling
    head = est.predict(Xva, iteration_range=(0, 5))
    assert not np.allclose(head, pred)
    leaves = est.apply(Xva)
    assert leaves.shape[0] == Xva.shape[0]
    assert est.feature_importances_.shape == (Xtr.shape[1],)


def test_save_model_load_model_roundtrip():
    Xtr, ytr, Xva, _ = _reg_data()
    est = bonsai.BonsaiRegressor(n_estimators=20).fit(Xtr, ytr)
    with tempfile.NamedTemporaryFile(suffix=".msgpack") as f:
        est.save_model(f.name)
        out = bonsai.BonsaiRegressor()
        assert out.load_model(f.name) is out
        np.testing.assert_allclose(out.predict(Xva), est.predict(Xva))


def test_classifier_accepts_xgb_objective_strings():
    Xtr, ytr, Xva, yva = _cls_data(k=2)
    est = bonsai.BonsaiClassifier(
        n_estimators=30, objective="binary:logistic", random_state=0
    )
    est.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=True)
    proba = est.predict_proba(Xva)
    assert proba.shape == (len(Xva), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    with np.testing.assert_raises(ValueError):
        bonsai.BonsaiClassifier(objective="reg:squarederror").fit(Xtr, ytr)


def test_multiclass_objective_string_and_mlogloss_key():
    Xtr, ytr, Xva, yva = _cls_data(k=3)
    est = bonsai.BonsaiClassifier(n_estimators=25, objective="multi:softprob")
    est.fit(Xtr, ytr, eval_set=[(Xva, yva)])
    r = est.evals_result()
    assert "mlogloss" in r["validation_0"]
    assert est.predict_proba(Xva).shape == (len(Xva), 3)


def test_regressor_quantile_objective_string():
    Xtr, ytr, Xva, _ = _reg_data()
    est = bonsai.BonsaiRegressor(
        n_estimators=30, objective="reg:quantileerror", quantile_alpha=0.9
    )
    est.fit(Xtr, ytr)
    # a 0.9-quantile fit predicts above the median fit on average
    med = bonsai.BonsaiRegressor(
        n_estimators=30, objective="reg:quantileerror", quantile_alpha=0.5
    ).fit(Xtr, ytr)
    assert est.predict(Xva).mean() > med.predict(Xva).mean()


def test_eval_history_without_early_stopping():
    """xgboost records evals_result whenever eval_set is passed; early
    stopping is not required."""
    Xtr, ytr, Xva, yva = _reg_data()
    est = bonsai.BonsaiRegressor(n_estimators=15)
    est.fit(Xtr, ytr, eval_set=(Xva, yva))
    hist = est.evals_result()["validation_0"]["rmse"]
    assert len(hist) == 15
    # best_iteration is an early-stopping concept, undefined here (as in xgb)
    with np.testing.assert_raises(AttributeError):
        _ = est.best_iteration


def test_device_and_grower_mapping():
    assert bonsai._grower_for_device("leafwise", "cuda") == "cuda_depthwise"
    assert bonsai._grower_for_device("oblivious", "cuda") == "cuda_oblivious"
    assert bonsai._grower_for_device("cuda_oblivious", "cpu") == "oblivious"
    assert bonsai._grower_for_device("depthwise", "cpu") == "depthwise"
    with np.testing.assert_raises(ValueError):
        bonsai._grower_for_device("leafwise", "tpu")
    if not bonsai.cuda_available():
        with np.testing.assert_raises(Exception):
            Xtr, ytr, _, _ = _reg_data(n=100)
            bonsai.BonsaiRegressor(n_estimators=2, device="cuda").fit(Xtr, ytr)


def test_sklearn_clone_roundtrips_new_aliases():
    from sklearn.base import clone

    est = bonsai.BonsaiClassifier(objective="binary:logistic", subsample=0.7,
                                  min_child_weight=2.0, device="cpu")
    twin = clone(est)
    assert twin.get_params() == est.get_params()

    reg = bonsai.BonsaiRegressor(gamma=0.5, quantile_alpha=0.9)
    assert clone(reg).get_params() == reg.get_params()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name} ok")
