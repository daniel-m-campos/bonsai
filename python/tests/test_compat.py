"""Tests for bonsai._compat (the xgboost-style alias layer)."""

from __future__ import annotations

import pickle

import bonsai
import numpy as np
from conftest import TEST_CSV, TRAIN_CSV, load_csv


def test_alias_mapping_regressor():
    assert ("booster.n_iters", "7") in bonsai.BonsaiRegressor(n_estimators=7)._build_pairs()
    assert ("tree.max_leaves", "8") in bonsai.BonsaiRegressor(num_leaves=8)._build_pairs()
    assert ("booster.random_seed", "3") in bonsai.BonsaiRegressor(random_state=3)._build_pairs()
    assert ("parallel.n_threads", "2") in bonsai.BonsaiRegressor(n_jobs=2)._build_pairs()
    assert ("tree.lambda_l2", "2.0") in bonsai.BonsaiRegressor(reg_lambda=2.0)._build_pairs()
    assert ("tree.lambda_l1", "1.5") in bonsai.BonsaiRegressor(reg_alpha=1.5)._build_pairs()
    assert ("bin_mapper.max_bin", "63") in bonsai.BonsaiRegressor(max_bin=63)._build_pairs()
    assert ("tree.min_data_in_leaf", "5") in bonsai.BonsaiRegressor(
        min_child_samples=5
    )._build_pairs()
    assert ("tree.feature_fraction", "0.8") in bonsai.BonsaiRegressor(
        colsample_bytree=0.8
    )._build_pairs()


def _classifier_pairs(**kwargs):
    """_build_pairs() needs n_classes_, which fit() derives from y; set it
    by hand to check the pure config-mapping logic without a full fit."""
    m = bonsai.BonsaiClassifier(**kwargs)
    m.n_classes_ = 2
    return m._build_pairs()


def test_alias_mapping_classifier():
    assert ("tree.max_leaves", "8") in _classifier_pairs(num_leaves=8)
    assert ("booster.n_iters", "9") in _classifier_pairs(n_estimators=9)
    assert ("tree.lambda_l2", "3.0") in _classifier_pairs(reg_lambda=3.0)


def test_alias_unset_leaves_canonical_untouched():
    """Aliases default to None; when unset the canonical first-class value
    (and its default) is what ends up in the pairs."""
    pairs = dict(bonsai.BonsaiRegressor(n_iters=42)._build_pairs())
    assert pairs["booster.n_iters"] == "42"


def test_alias_precedence_params_wins():
    pairs = dict(
        bonsai.BonsaiRegressor(
            reg_lambda=1.0, params={"tree.lambda_l2": 5.0}
        )._build_pairs()
    )
    assert pairs["tree.lambda_l2"] == "5.0"


def test_alias_precedence_alias_wins_over_canonical():
    pairs = dict(bonsai.BonsaiRegressor(n_iters=10, n_estimators=99)._build_pairs())
    assert pairs["booster.n_iters"] == "99"


def test_alias_end_to_end_fit():
    Xtr, ytr = load_csv(TRAIN_CSV)
    m = bonsai.BonsaiRegressor(n_estimators=7).fit(Xtr, ytr)
    assert m.n_iters_ == 7


def test_alias_sklearn_clone_round_trip():
    try:
        import sklearn.base
    except ImportError:
        return

    est = bonsai.BonsaiRegressor(n_estimators=50, reg_lambda=1.0)
    params = est.get_params()
    assert params["n_estimators"] == 50
    assert params["reg_lambda"] == 1.0

    cloned = sklearn.base.clone(est)
    assert cloned is not est
    assert cloned.get_params() == params
    assert cloned.n_estimators == 50
    assert cloned.reg_lambda == 1.0


def test_alias_pickle_fitted_predicts_identically():
    Xtr, ytr = load_csv(TRAIN_CSV)
    Xte, _ = load_csv(TEST_CSV)
    m = bonsai.BonsaiRegressor(n_estimators=20, reg_lambda=0.5).fit(Xtr, ytr)
    before = m.predict(Xte)

    restored = pickle.loads(pickle.dumps(m))
    after = restored.predict(Xte)

    assert np.array_equal(before, after)
    assert restored.n_iters_ == m.n_iters_
    assert restored.get_params()["reg_lambda"] == 0.5


def test_alias_table_matches_init_signatures():
    """The alias set is maintained in three places (base __init__, the
    _ALIAS_TO_KEY table, BonsaiRegressor's re-declared __init__); this pins
    them together so adding an alias in one place can't silently drift."""
    import inspect

    aliases = set(bonsai.BonsaiClassifier._ALIAS_TO_KEY)
    for cls in (bonsai.BonsaiRegressor, bonsai.BonsaiClassifier):
        params = set(inspect.signature(cls.__init__).parameters) - {"self"}
        missing = aliases - params
        assert not missing, f"{cls.__name__}.__init__ missing aliases: {missing}"
        est = cls()
        assert aliases <= set(est.get_params()), "get_params must cover aliases"
