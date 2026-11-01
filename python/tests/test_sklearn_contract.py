"""The scikit-learn contract of BonsaiRegressor and BonsaiClassifier: the
shapes, refusals and fitted state sklearn's own estimators promise."""

from __future__ import annotations

import pickle
import sys

import bonsai
import numpy as np
import pytest


def _binary(n: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A small separable binary problem with 0/1 labels."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    return X, (X[:, 0] - X[:, 1] > 0).astype(np.int64)


def _classifier(**kwargs) -> bonsai.BonsaiClassifier:
    return bonsai.BonsaiClassifier(n_iters=10, n_threads=2, **kwargs)


def test_classifier_score_reads_a_label_column_as_a_vector():
    X, y = _binary()
    clf = _classifier().fit(X, y)

    assert clf.score(X, y.reshape(-1, 1)) == clf.score(X, y)


def test_classifier_eval_set_reads_a_label_column_as_a_vector():
    X, y = _binary()
    column = y.reshape(-1, 1)

    clf = _classifier().fit(X, column, eval_set=(X[:100], column[:100]))

    assert len(clf.evals_result()["valid"]["logloss"]) == clf.n_iters_


def test_classifier_refuses_labels_wider_than_one_column():
    X, y = _binary()
    with pytest.raises(ValueError, match="y must be 1-dimensional or a single column"):
        _classifier().fit(X, np.column_stack([y, y]))


def _regression(n: int = 200, n_targets: int = 1, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A small linear problem with ``n_targets`` columns of y."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4)).astype(np.float32)
    Y = np.column_stack([X[:, k] + 0.1 * rng.normal(size=n) for k in range(n_targets)])
    return X, Y[:, 0] if n_targets == 1 else Y


def test_score_refuses_y_whose_row_count_differs_from_X():
    Xr, yr = _regression()
    Xc, yc = _binary(200)
    reg = bonsai.BonsaiRegressor(n_iters=5, n_threads=2).fit(Xr, yr)
    clf = _classifier().fit(Xc, yc)

    for est, X, y in ((reg, Xr, yr), (clf, Xc, yc)):
        with pytest.raises(ValueError, match=r"len\(y\) must equal the row count of X"):
            est.score(X, y[:1])
        with pytest.raises(ValueError, match=r"len\(y\) must equal the row count of X"):
            est.score(X[:1], y)


def test_score_refuses_sample_weight_whose_row_count_differs_from_X():
    X, y = _binary(200)
    clf = _classifier().fit(X, y)

    with pytest.raises(ValueError, match=r"len\(sample_weight\) must equal the row count"):
        clf.score(X, y, sample_weight=np.ones(1))


def test_regressor_score_refuses_fewer_targets_than_the_model_predicts():
    X, Y = _regression(n_targets=3)
    reg = bonsai.BonsaiRegressor(n_iters=5, n_threads=2).fit(X, Y)

    with pytest.raises(ValueError, match="y has 2 targets but the model predicts 3"):
        reg.score(X, Y[:, :2])


def test_regressor_score_scores_the_targets_in_float64():
    X = np.zeros((40, 2), dtype=np.float32)
    y = np.where(np.arange(40) % 2 == 0, 2.0**24, 2.0**24 + 1)
    reg = bonsai.BonsaiRegressor(n_iters=2, n_threads=2).fit(X, y)
    y_pred = reg.predict(X).astype(np.float64)
    expected = 1.0 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)

    assert reg.score(X, y) == pytest.approx(expected)
    assert reg.score(X, y) < 0.0


def test_classifier_staged_predict_decodes_each_stage_like_predict():
    X, y = _binary()
    names = np.array(["cat", "dog"])[y]
    three = np.array([10, 20, 30])[np.digitize(X[:, 0], [-0.5, 0.5])]

    for labels in (names, three):
        clf = _classifier().fit(X, labels)
        staged = clf.staged_predict(X)

        assert staged.shape == (clf.n_iters_, len(X))
        np.testing.assert_array_equal(staged[-1], clf.predict(X))
        np.testing.assert_array_equal(staged[2], clf.predict(X, num_iteration=3))


def test_classifier_predict_proba_of_a_two_class_softmax_is_its_probabilities(tmp_path):
    X, y = _binary()
    softmax = {"dispatch.objective_name": "softmax", "objective.n_classes": 2}
    model = bonsai.train({**softmax, "booster.n_iters": 10, "parallel.n_threads": 2}, X, y)
    path = str(tmp_path / "softmax.msgpack")
    model.save(path)

    loaded = bonsai.BonsaiClassifier.from_file(path)
    forced = _classifier(params=softmax).fit(X, y)

    np.testing.assert_allclose(loaded.predict_proba(X[:20]), model.predict_proba(X[:20]))
    assert 0.0 < forced.predict_proba(X).min() < 1.0
    np.testing.assert_array_equal(loaded.predict(X), np.argmax(model.predict_proba(X), axis=1))


def test_fit_that_raises_leaves_the_previous_fit_in_place():
    X, y = _binary()
    labels = np.array(["cat", "dog", "eel"])[np.digitize(X[:, 0], [-0.5, 0.5])]
    clf = _classifier().fit(X, labels)
    before = clf.predict_proba(X[:5])

    with pytest.raises(ValueError, match="not in the training classes"):
        clf.fit(X, y, eval_set=(X[:50], np.full(50, 7)))

    np.testing.assert_array_equal(clf.classes_, ["cat", "dog", "eel"])
    assert clf.n_classes_ == 3
    np.testing.assert_array_equal(clf.predict_proba(X[:5]), before)


def test_regressor_fit_that_raises_keeps_its_target_count():
    X, Y = _regression(n_targets=3)
    reg = bonsai.BonsaiRegressor(n_iters=5, n_threads=2).fit(X, Y)

    with pytest.raises(ValueError):
        reg.fit(X, Y[:, 0], sample_weight=np.ones(3))

    assert reg.n_outputs_ == 3
    assert reg.predict(X).shape == (len(X), 3)


def _early_stopping_fit(**kwargs) -> bonsai.BonsaiRegressor:
    """A regressor fit with a held-out eval set that stops well before 200 rounds."""
    X, y = _regression(n=400)
    y = y + np.random.default_rng(1).normal(size=len(y))
    est = bonsai.BonsaiRegressor(n_iters=200, learning_rate=0.3, n_threads=2, **kwargs)
    return est.fit(X[100:], y[100:], eval_set=(X[:100], y[:100]))


def test_best_iteration_follows_early_stopping_set_through_params():
    by_kwarg = _early_stopping_fit(early_stopping_rounds=5)
    by_params = _early_stopping_fit(params={"booster.early_stopping_rounds": 5})

    assert by_params.n_iters_ == by_kwarg.n_iters_ < 200
    assert by_params.best_iteration == by_kwarg.best_iteration
    assert by_params.best_score == by_kwarg.best_score


def test_best_iteration_is_undefined_when_params_turn_early_stopping_off():
    est = _early_stopping_fit(early_stopping_rounds=5, params={"booster.early_stopping_rounds": 0})

    assert est.n_iters_ == 200
    for attribute in ("best_iteration", "best_score"):
        with pytest.raises(AttributeError, match="needs fit"):
            getattr(est, attribute)


def test_pickle_keeps_the_eval_history():
    est = _early_stopping_fit(early_stopping_rounds=5)

    loaded = pickle.loads(pickle.dumps(est))

    assert loaded.evals_result() == est.evals_result()
    assert loaded.best_iteration == est.best_iteration
    assert loaded.best_score == est.best_score


def test_pickle_written_without_an_eval_history_loads_without_one():
    est = _early_stopping_fit(early_stopping_rounds=5)
    state = est.__getstate__()
    del state["_fit_eval_history"], state["_fit_early_stopping_rounds"]

    loaded = bonsai.BonsaiRegressor.__new__(bonsai.BonsaiRegressor)
    loaded.__setstate__(state)

    assert loaded.evals_result() == {}
    np.testing.assert_array_equal(loaded.predict(np.zeros((2, 4))), est.predict(np.zeros((2, 4))))


def _overrides_fit_passes(monkeypatch, est) -> dict:
    """The overrides ``fit`` hands the native ``train``, captured before any training."""
    import bonsai.estimators

    captured: dict = {}

    def capture(overrides, *args, **kwargs):
        captured.update(overrides)
        raise RuntimeError("captured")

    monkeypatch.setattr(bonsai.estimators, "train", capture)
    X, y = _regression(n=50)
    with pytest.raises(RuntimeError, match="captured"):
        est.fit(X, y)
    return captured


def test_device_cuda_n_places_the_fit_on_device_n(monkeypatch):
    overrides = _overrides_fit_passes(monkeypatch, bonsai.BonsaiRegressor(device="cuda:1"))

    assert overrides["parallel.device_id"] == 1
    assert overrides["dispatch.grower_name"] == "cuda_leafwise"


def test_device_refuses_a_malformed_cuda_index():
    X, y = _regression(n=50)
    for device in ("cuda:abc", "cuda:", "tpu:0"):
        with pytest.raises(ValueError, match="device must be 'cpu', 'cuda' or 'cuda:N'"):
            bonsai.BonsaiRegressor(device=device).fit(X, y)


def test_tags_allow_nan_features():
    from sklearn.utils import get_tags

    for est in (bonsai.BonsaiRegressor(), bonsai.BonsaiClassifier()):
        assert get_tags(est).input_tags.allow_nan


def test_tags_mark_only_the_regressor_multi_output():
    from sklearn.utils import get_tags

    assert get_tags(bonsai.BonsaiRegressor()).target_tags.multi_output
    assert not get_tags(bonsai.BonsaiClassifier()).target_tags.multi_output


def test_sequential_feature_selector_passes_nan_through():
    from sklearn.feature_selection import SequentialFeatureSelector

    X, y = _regression()
    X[::10, 1] = np.nan
    selector = SequentialFeatureSelector(
        bonsai.BonsaiRegressor(n_iters=5, n_threads=2), n_features_to_select=2, cv=2
    )

    assert selector.fit(X, y).transform(X).shape == (len(X), 2)


def test_unfitted_estimator_raises_not_fitted_error():
    from sklearn.exceptions import NotFittedError

    X = np.zeros((3, 2), dtype=np.float32)
    for est in (bonsai.BonsaiRegressor(), bonsai.BonsaiClassifier()):
        for method in ("predict", "pred_contribs", "predict_leaf", "staged_predict"):
            with pytest.raises(NotFittedError, match="call fit"):
                getattr(est, method)(X)
        with pytest.raises(RuntimeError):
            est.predict(X)
    with pytest.raises(NotFittedError):
        bonsai.BonsaiClassifier().predict_proba(X)


def test_not_fitted_error_pickles():
    with pytest.raises(Exception) as raised:
        bonsai.BonsaiRegressor().predict(np.zeros((1, 2)))

    again = pickle.loads(pickle.dumps(raised.value))

    assert type(again) is type(raised.value)


def test_unfitted_estimator_passes_sklearn_check():
    from sklearn.utils.estimator_checks import check_estimators_unfitted

    for est in (bonsai.BonsaiRegressor(n_iters=2), bonsai.BonsaiClassifier(n_iters=2)):
        check_estimators_unfitted(type(est).__name__, est)


def test_not_fitted_error_without_sklearn_keeps_its_bases(monkeypatch):
    import bonsai.estimators

    monkeypatch.setitem(sys.modules, "sklearn.exceptions", None)
    error = bonsai.estimators._not_fitted_error.__wrapped__()

    assert all(issubclass(error, base) for base in (ValueError, AttributeError, RuntimeError))
