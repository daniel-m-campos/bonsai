"""Tests for feature names: where they come from and what carries them."""

from __future__ import annotations

import bonsai
import numpy as np
import pytest

NAMES = ["age", "income", "tenure", "score", "visits", "spend"]


def _reg_data(n=3000, f=6, seed=0):
    """(X, y) float32 arrays with signal in the first two columns."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, f), dtype=np.float32)
    y = (X[:, 0] * 2 + X[:, 1] + rng.normal(0, 0.1, n)).astype(np.float32)
    return X, y


def test_feature_names_come_from_the_keyword():
    X, y = _reg_data()
    ds = bonsai.Dataset(X, y, feature_names=NAMES)
    assert list(ds.feature_names) == NAMES
    text = bonsai.train({"booster.n_iters": "5", "tree.max_depth": "4"}, ds).dump()
    assert "income <=" in text
    assert "f1 <=" not in text


def test_feature_names_default_to_f0_fn():
    X, y = _reg_data()
    ds = bonsai.Dataset(X, y)
    assert list(ds.feature_names) == ["f0", "f1", "f2", "f3", "f4", "f5"]
    assert "f0" in bonsai.train({"booster.n_iters": "5"}, ds).dump()


def test_feature_names_reject_a_wrong_count_or_a_repeat():
    X, y = _reg_data()
    with pytest.raises(ValueError, match="one name per column"):
        bonsai.Dataset(X, y, feature_names=["a", "b", "c"])
    with pytest.raises(ValueError, match="must be unique"):
        bonsai.Dataset(X, y, feature_names=["a", "b", "c", "d", "e", "a"])


def test_feature_names_survive_a_reference_dataset():
    """A reference supplies the cuts, and the names travel with them."""
    X, y = _reg_data()
    train_ds = bonsai.Dataset(X[:2000], y[:2000], feature_names=NAMES)
    valid_ds = bonsai.Dataset(X[2000:], y[2000:], reference=train_ds)
    assert list(valid_ds.feature_names) == NAMES


def test_feature_names_with_a_reference_raise():
    """The reference already names the columns, so a second set here would be
    validated and then dropped; say so instead."""
    X, y = _reg_data()
    train_ds = bonsai.Dataset(X[:2000], y[:2000], feature_names=NAMES)
    with pytest.raises(ValueError, match="feature_names cannot be given with reference="):
        bonsai.Dataset(X[2000:], y[2000:], reference=train_ds, feature_names=NAMES)


def test_model_carries_its_feature_names(tmp_path):
    """The names are the model's own, and they survive save/load."""
    X, y = _reg_data()
    model = bonsai.train({"booster.n_iters": "5"}, X, y, feature_names=NAMES)
    assert list(model.feature_names) == NAMES
    path = str(tmp_path / "named.msgpack")
    model.save(path)
    assert list(bonsai.load(path).feature_names) == NAMES
    assert list(bonsai.train({"booster.n_iters": "5"}, X, y).feature_names) == [
        f"f{i}" for i in range(6)
    ]


# Monotone constraints keyed by feature name =======================================================

MONO_PARAMS = {"dispatch.grower_name": "depthwise", "booster.n_iters": "25", "tree.max_depth": "5"}


def _mono(params, **extra):
    """MONO_PARAMS plus a tree.monotone_constraints value."""
    return {**MONO_PARAMS, "tree.monotone_constraints": params, **extra}


def test_named_monotone_matches_the_positional_list_on_both_train_paths(tmp_path):
    """The dict form is a spelling of the list, so it must produce the same
    model byte for byte, not merely the same predictions."""
    X, y = _reg_data()
    positional = [1, 0, -1, 0, 0, 0]
    named = {"age": 1, "tenure": -1}

    ds = bonsai.Dataset(X, y, feature_names=NAMES)
    by_list = bonsai.train(_mono(positional), ds)
    by_name = bonsai.train(_mono(named), ds)
    np.testing.assert_array_equal(np.asarray(by_list.predict(ds)), np.asarray(by_name.predict(ds)))
    assert by_list.config_toml == by_name.config_toml
    for model, stem in ((by_list, "list"), (by_name, "named")):
        model.save(str(tmp_path / f"{stem}.msgpack"))
    assert (tmp_path / "list.msgpack").read_bytes() == (tmp_path / "named.msgpack").read_bytes()

    # the fused array path resolves against the names it synthesizes
    array_list = bonsai.train(_mono(positional), X, y)
    array_named = bonsai.train(_mono({"f0": 1, "f2": -1}), X, y)
    np.testing.assert_array_equal(
        np.asarray(array_list.predict(X)), np.asarray(array_named.predict(X))
    )


def test_named_monotone_resolves_against_synthesized_names():
    """f0..fN are real names as far as the model is concerned."""
    X, y = _reg_data()
    ds = bonsai.Dataset(X, y)
    by_list = bonsai.train(_mono([0, -1, 0, 0, 0, 1]), ds)
    by_name = bonsai.train(_mono({"f1": -1, "f5": 1}), ds)
    assert by_list.config_toml == by_name.config_toml
    np.testing.assert_array_equal(np.asarray(by_list.predict(ds)), np.asarray(by_name.predict(ds)))


def test_named_monotone_leaves_unlisted_features_free():
    """An unlisted feature is 0, not absent: a partial dict equals the explicit
    all-zeros-but-one list."""
    X, y = _reg_data()
    ds = bonsai.Dataset(X, y, feature_names=NAMES)
    explicit = bonsai.train(_mono([0, 0, 0, 0, 0, 0]), ds)
    empty = bonsai.train(_mono({}), ds)
    assert explicit.config_toml == empty.config_toml
    np.testing.assert_array_equal(np.asarray(explicit.predict(ds)), np.asarray(empty.predict(ds)))


def test_named_monotone_rejects_unknown_names_and_bad_values():
    X, y = _reg_data()
    ds = bonsai.Dataset(X, y, feature_names=NAMES)
    with pytest.raises(ValueError, match="does not have: 'salary'"):
        bonsai.train(_mono({"age": 1, "salary": -1}), ds)
    # every offender is listed, however many there are
    with pytest.raises(ValueError, match=r"'nope0'.*'nope6'.*carries 6 feature names"):
        bonsai.train(_mono({f"nope{i}": 1 for i in range(7)}), ds)
    with pytest.raises(ValueError, match="must be the int -1, 0, or 1"):
        bonsai.train(_mono({"age": 2}), ds)
    with pytest.raises(ValueError, match="must be the int -1, 0, or 1"):
        bonsai.train(_mono({"age": "1"}), ds)
    # the array path resolves against its own names, so a Dataset name is
    # unknown there when X carries none
    with pytest.raises(ValueError, match="does not have: 'age'"):
        bonsai.train(_mono({"age": 1}), X, y)


def test_named_monotone_flows_through_the_estimator_params():
    """The estimators hand params= to this same binding, so the dict form
    arrives with no estimator-side code. A bare array carries no names, so the
    dict resolves against the synthesized ones; real names come from a named
    frame or from ``fit(feature_names=...)``."""
    X, y = _reg_data()

    def fit(constraints):
        return bonsai.BonsaiRegressor(
            n_iters=25, max_depth=5, grower="depthwise",
            params={"tree.monotone_constraints": constraints},
        ).fit(X, y)

    np.testing.assert_array_equal(
        fit({"f0": 1, "f2": -1}).predict(X), fit([1, 0, -1, 0, 0, 0]).predict(X)
    )
    with pytest.raises(ValueError, match="does not have: 'age'"):
        fit({"age": 1})


# feature_names on the fused train =================================================================

def test_train_takes_feature_names_for_an_array():
    X, y = _reg_data()
    model = bonsai.train(
        {"booster.n_iters": "5", "tree.max_depth": "4"}, X, y, feature_names=NAMES
    )
    text = model.dump()
    assert "income <=" in text
    assert "f1 <=" not in text


def test_train_named_monotone_resolves_against_the_given_names():
    X, y = _reg_data()
    by_name = bonsai.train(_mono({"age": 1, "tenure": -1}), X, y, feature_names=NAMES)
    by_list = bonsai.train(_mono([1, 0, -1, 0, 0, 0]), X, y, feature_names=NAMES)
    assert by_name.config_toml == by_list.config_toml
    np.testing.assert_array_equal(
        np.asarray(by_name.predict(X)), np.asarray(by_list.predict(X))
    )


def test_train_feature_names_with_init_model_raises(tmp_path):
    """A warm start carries the loaded model's names, so supplied ones would be
    a lie rather than an override."""
    X, y = _reg_data()
    path = str(tmp_path / "warm.msgpack")
    bonsai.train({"booster.n_iters": "5"}, X, y, feature_names=NAMES).save(path)
    with pytest.raises(ValueError, match="feature_names cannot be given with init_model"):
        bonsai.train({"booster.n_iters": "5"}, X, y, init_model=path, feature_names=NAMES)


# The estimator layer ==============================================================================

class _Frame:
    """The little of a DataFrame the estimator layer reads: column names, and
    an array to coerce from. pandas is not a test dependency, and no DLPack
    forwarding is needed because the estimators convert through numpy."""

    def __init__(self, values, columns):
        self._values = values
        self.columns = list(columns)

    def __array__(self, dtype=None, copy=None):
        return self._values if dtype is None else self._values.astype(dtype)


def test_estimator_reads_the_frame_columns():
    X, y = _reg_data()
    est = bonsai.BonsaiRegressor(n_iters=10, max_depth=4).fit(_Frame(X, NAMES), y)
    np.testing.assert_array_equal(est.feature_names_in_, np.asarray(NAMES, dtype=object))
    assert est.feature_names_in_.dtype == object
    text = est.dump()
    assert "income <=" in text
    assert "f1 <=" not in text


def test_estimator_on_a_bare_array_has_no_names():
    X, y = _reg_data()
    est = bonsai.BonsaiRegressor(n_iters=10, max_depth=4).fit(X, y)
    assert not hasattr(est, "feature_names_in_")
    assert "f0" in est.dump()


def test_estimator_feature_names_kwarg_overrides_the_columns():
    X, y = _reg_data()
    other = ["c0", "c1", "c2", "c3", "c4", "c5"]
    est = bonsai.BonsaiRegressor(n_iters=10, max_depth=4).fit(
        _Frame(X, NAMES), y, feature_names=other
    )
    np.testing.assert_array_equal(est.feature_names_in_, np.asarray(other, dtype=object))
    assert "c1 <=" in est.dump()


def test_estimator_refit_on_an_array_clears_stale_names():
    X, y = _reg_data()
    est = bonsai.BonsaiRegressor(n_iters=10, max_depth=4).fit(_Frame(X, NAMES), y)
    assert hasattr(est, "feature_names_in_")
    est.fit(X, y)
    assert not hasattr(est, "feature_names_in_")
    assert "f0" in est.dump()


def test_estimator_warm_start_on_a_frame(tmp_path):
    """Names discovered on a frame are not an override, so a warm start reads
    them without tripping the native guard; an explicit set still raises."""
    X, y = _reg_data()
    frame = _Frame(X, NAMES)
    path = str(tmp_path / "warm.msgpack")
    bonsai.BonsaiRegressor(n_iters=10, max_depth=4).fit(frame, y).save(path)

    warm = bonsai.BonsaiRegressor(n_iters=10, max_depth=4).fit(
        frame, y, init_model=path
    )
    assert warm.n_iters_ == 20
    assert "income <=" in warm.dump()
    # the names are the loaded model's, which here agree with the frame's
    np.testing.assert_array_equal(warm.feature_names_in_, np.asarray(NAMES, dtype=object))

    with pytest.raises(ValueError, match="feature_names cannot be given with init_model"):
        bonsai.BonsaiRegressor(n_iters=10, max_depth=4).fit(
            frame, y, init_model=path, feature_names=NAMES
        )


def test_estimator_integer_columns_are_not_names():
    """sklearn's rule: default integer columns are no names at all, not the
    names "0" and "1"."""
    X, y = _reg_data()
    est = bonsai.BonsaiRegressor(n_iters=10, max_depth=4).fit(_Frame(X, range(6)), y)
    assert not hasattr(est, "feature_names_in_")
    assert "f0" in est.dump()


def test_from_file_reports_the_saved_names(tmp_path):
    """dump() shows the saved model's names, so the fitted-input attributes
    have to agree with it."""
    X, y = _reg_data()
    named = str(tmp_path / "named.msgpack")
    bare = str(tmp_path / "bare.msgpack")
    bonsai.BonsaiRegressor(n_iters=10, max_depth=4).fit(_Frame(X, NAMES), y).save(named)
    bonsai.BonsaiRegressor(n_iters=10, max_depth=4).fit(X, y).save(bare)

    est = bonsai.BonsaiRegressor.from_file(named)
    assert est.n_features_in_ == 6
    np.testing.assert_array_equal(est.feature_names_in_, np.asarray(NAMES, dtype=object))

    # synthesized f0..fN are how the artifact spells "unnamed"
    unnamed = bonsai.BonsaiRegressor.from_file(bare)
    assert unnamed.n_features_in_ == 6
    assert not hasattr(unnamed, "feature_names_in_")


def test_classifier_reads_the_frame_columns():
    X, _ = _reg_data()
    labels = (X[:, 0] > 0.5).astype(np.float32)
    est = bonsai.BonsaiClassifier(n_iters=10, max_depth=4).fit(_Frame(X, NAMES), labels)
    np.testing.assert_array_equal(est.feature_names_in_, np.asarray(NAMES, dtype=object))
    assert "age <=" in est.dump()


def test_named_monotone_works_through_an_estimator_fit_on_a_frame():
    X, y = _reg_data()
    est = bonsai.BonsaiRegressor(
        n_iters=25, max_depth=5, grower="depthwise",
        params={"tree.monotone_constraints": {"age": 1, "tenure": -1}},
    ).fit(_Frame(X, NAMES), y)
    by_list = bonsai.BonsaiRegressor(
        n_iters=25, max_depth=5, grower="depthwise",
        params={"tree.monotone_constraints": [1, 0, -1, 0, 0, 0]},
    ).fit(_Frame(X, NAMES), y)
    frame = _Frame(X, NAMES)
    np.testing.assert_array_equal(est.predict(frame), by_list.predict(frame))


# Names at predict time ============================================================================

def test_predict_warns_when_the_names_go_missing():
    X, y = _reg_data()
    est = bonsai.BonsaiRegressor(n_iters=10, max_depth=4).fit(_Frame(X, NAMES), y)
    with pytest.warns(UserWarning, match="X does not have valid feature names"):
        est.predict(X)
    with pytest.warns(UserWarning, match="BonsaiRegressor was fitted with feature names"):
        est.pred_contribs(X)


def test_predict_warns_when_the_names_appear():
    X, y = _reg_data()
    est = bonsai.BonsaiRegressor(n_iters=10, max_depth=4).fit(X, y)
    with pytest.warns(UserWarning, match="X has feature names, but BonsaiRegressor"):
        est.predict(_Frame(X, NAMES))
    with pytest.warns(UserWarning, match="fitted without feature names"):
        est.staged_predict(_Frame(X, NAMES))


def test_predict_rejects_names_that_disagree():
    X, y = _reg_data()
    est = bonsai.BonsaiRegressor(n_iters=10, max_depth=4).fit(_Frame(X, NAMES), y)
    other = ["age", "income", "tenure", "score", "visits", "salary"]
    with pytest.raises(ValueError, match="should match those that were passed during fit"):
        est.predict(_Frame(X, other))
    with pytest.raises(ValueError, match=r"Unseen at fit time: \['salary'\]"):
        est.predict_leaf(_Frame(X, other))
    with pytest.raises(ValueError, match=r"now missing: \['spend'\]"):
        est.predict(_Frame(X, other))
    with pytest.raises(ValueError, match="different order"):
        est.predict(_Frame(X, list(reversed(NAMES))))


def test_predict_names_the_disagreement_it_found():
    """One sentence per kind of disagreement: only the unseen names, only the
    missing names, or both in that order, and the order remark only when the
    sets agree."""
    X, y = _reg_data()
    est = bonsai.BonsaiRegressor(n_iters=10, max_depth=4).fit(_Frame(X, NAMES), y)
    prefix = "The feature names should match those that were passed during fit."
    with pytest.raises(ValueError) as unseen_only:
        est.predict(_Frame(np.hstack([X, X[:, :1]]), [*NAMES, "salary"]))
    assert str(unseen_only.value) == prefix + " Unseen at fit time: ['salary']."
    with pytest.raises(ValueError) as missing_only:
        est.predict(_Frame(X[:, :5], NAMES[:5]))
    assert str(missing_only.value) == prefix + " Seen at fit time, now missing: ['spend']."
    with pytest.raises(ValueError) as both:
        est.predict(_Frame(X, [*NAMES[:5], "salary"]))
    assert str(both.value) == (
        prefix + " Unseen at fit time: ['salary']. Seen at fit time, now missing: ['spend']."
    )
    with pytest.raises(ValueError) as reordered:
        est.predict(_Frame(X, list(reversed(NAMES))))
    assert str(reordered.value) == prefix + " The same names arrived in a different order."


def test_predict_on_a_dataset_reads_its_names(recwarn):
    """A Dataset names every column, so it must not read as unnamed; its
    synthesized f0..fN must not read as named either."""
    X, y = _reg_data()
    named = bonsai.BonsaiRegressor(n_iters=10, max_depth=4).fit(_Frame(X, NAMES), y)
    named.predict(bonsai.Dataset(X, y, feature_names=NAMES))

    bare = bonsai.BonsaiRegressor(n_iters=10, max_depth=4).fit(X, y)
    bare.predict(bonsai.Dataset(X, y))
    assert [str(w.message) for w in recwarn] == []

    with pytest.raises(ValueError, match="should match those"):
        named.predict(bonsai.Dataset(X, y, feature_names=[f"c{i}" for i in range(6)]))
    with pytest.warns(UserWarning, match="X does not have valid feature names"):
        named.predict(bonsai.Dataset(X, y))


def test_classifier_predict_proba_checks_the_names():
    X, _ = _reg_data()
    labels = (X[:, 0] > 0.5).astype(np.float32)
    est = bonsai.BonsaiClassifier(n_iters=10, max_depth=4).fit(_Frame(X, NAMES), labels)
    with pytest.warns(UserWarning, match="BonsaiClassifier was fitted with feature names"):
        est.predict_proba(X)
    with pytest.warns(UserWarning, match="BonsaiClassifier was fitted with feature names"):
        est.predict(X)


# Column count ====================================================================================


def _fit_on(n_features, n=400, seed=0):
    """A model over `n_features` columns, and the matrix it was fit on."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 20, size=(n, n_features)).astype(np.float32)
    y = (X[:, 0] * 0.3 - X[:, 1] * 0.2).astype(np.float32)
    return bonsai.train({"booster.n_iters": "5", "tree.max_depth": "3"}, X, y), X


@pytest.mark.parametrize("width", [1, 3, 9])
@pytest.mark.parametrize(
    "call",
    [
        lambda m, X: m.predict(X),
        lambda m, X: m.staged_predict(X),
        lambda m, X: m.predict_leaf(X),
        lambda m, X: m.pred_contribs(X),
    ],
)
def test_a_reader_refuses_a_matrix_of_the_wrong_width(call, width):
    """A tree splits on feature ids, so a narrower matrix than the model was
    fit on reads past the end of every row: an out-of-bounds read that used to
    return a plausible number."""
    model, X = _fit_on(6)
    rng = np.random.default_rng(1)
    wrong = rng.integers(0, 20, size=(len(X), width)).astype(np.float32)
    with pytest.raises(ValueError, match="6 features"):
        call(model, wrong)


def test_the_right_width_still_reads():
    model, X = _fit_on(6)
    assert model.predict(X).shape == (len(X),)
