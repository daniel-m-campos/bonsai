"""Tests for bonsai.interop (the cross-library parameter mapping)."""

from __future__ import annotations

import bonsai
import pytest
from bonsai import interop

# from_xgboost =====================================================================================

def test_from_xgboost_translates_the_canonical_call():
    pairs = dict(interop.from_xgboost({
        "n_estimators": 80,
        "learning_rate": 0.1,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1.0,
        "gamma": 0.0,
        "reg_lambda": 1.0,
        "objective": "reg:squarederror",
        "random_state": 0,
        "n_jobs": 2,
        "early_stopping_rounds": 10,
    }))
    assert pairs == {
        "booster.n_iters": "80",
        "booster.learning_rate": "0.1",
        "tree.max_depth": "4",
        "sampler.subsample": "0.8",
        "dispatch.sampler_name": "bernoulli",
        "tree.feature_fraction": "0.8",
        "tree.min_child_hess": "1.0",
        "tree.min_gain_to_split": "0.0",
        "tree.lambda_l2": "1.0",
        "dispatch.objective_name": "mse",
        "booster.random_seed": "0",
        "parallel.n_threads": "2",
        "booster.early_stopping_rounds": "10",
    }


def test_from_xgboost_pairs_train_a_model(toy_train):
    Xtr, ytr = toy_train
    pairs = interop.from_xgboost({"n_estimators": 7, "max_depth": 3})
    assert bonsai.train(pairs, Xtr, ytr).n_iters == 7


def test_from_xgboost_strict_names_every_unmappable_key():
    with pytest.raises(ValueError) as excinfo:
        interop.from_xgboost({"colsample_bylevel": 0.5, "max_cat_to_onehot": 4})
    message = str(excinfo.value)
    assert "colsample_bylevel" in message and "max_cat_to_onehot" in message


def test_from_xgboost_lenient_drops_unmappable_keys():
    pairs = interop.from_xgboost({"colsample_bylevel": 0.5, "max_depth": 4},
                                 strict=False)
    assert pairs == [("tree.max_depth", "4")]


def test_from_xgboost_drops_recognized_non_knobs():
    assert interop.from_xgboost({"tree_method": "hist", "device": "cuda"}) == []


def test_from_xgboost_rejects_an_unknown_objective():
    with pytest.raises(ValueError, match="objective"):
        interop.from_xgboost({"objective": "rank:pairwise"})


def test_from_xgboost_accepts_the_booster_api_aliases():
    assert dict(interop.from_xgboost({"eta": 0.1, "lambda": 2.0})) == {
        "booster.learning_rate": "0.1",
        "tree.lambda_l2": "2.0",
    }


# from_lightgbm ====================================================================================

def test_from_lightgbm_maps_uncapped_depth_to_the_leaf_budget():
    pairs = dict(interop.from_lightgbm({"max_depth": -1, "num_leaves": 63}))
    assert pairs["tree.max_depth"] == str(interop.UNCAPPED_DEPTH)
    assert pairs["tree.max_leaves"] == "63"


def test_from_lightgbm_keeps_a_positive_depth_cap():
    assert dict(interop.from_lightgbm({"max_depth": 8}))["tree.max_depth"] == "8"


def test_from_lightgbm_accepts_the_documented_aliases():
    assert dict(interop.from_lightgbm({
        "num_iterations": 100, "min_child_samples": 20, "bagging_fraction": 0.5,
    })) == {
        "booster.n_iters": "100",
        "tree.min_data_in_leaf": "20",
        "sampler.subsample": "0.5",
        "dispatch.sampler_name": "bernoulli",
    }


def test_from_lightgbm_translates_objectives():
    assert dict(interop.from_lightgbm({"objective": "regression_l1"})) == {
        "dispatch.objective_name": "mae"
    }
    assert dict(interop.from_lightgbm({"objective": "multiclass"})) == {
        "dispatch.objective_name": "softmax"
    }


def test_from_lightgbm_drops_the_overloaded_alpha():
    assert interop.from_lightgbm({"alpha": 0.9}) == []


# from_catboost ====================================================================================

def test_from_catboost_adds_the_border_fencepost():
    assert dict(interop.from_catboost({"border_count": 254})) == {
        "bin_mapper.max_bin": "255"
    }


def test_from_catboost_translates_the_grow_policy():
    assert dict(interop.from_catboost({"grow_policy": "SymmetricTree"})) == {
        "dispatch.grower_name": "levelwise"
    }


def test_from_catboost_rejects_a_parametrized_loss():
    with pytest.raises(ValueError, match="loss_function"):
        interop.from_catboost({"loss_function": "Huber:delta=1.0"})


def test_from_catboost_drops_the_best_model_flag():
    assert interop.from_catboost({"use_best_model": False, "task_type": "GPU"}) == []


# to_xgboost =======================================================================================

def test_to_xgboost_keeps_value_types():
    out = interop.to_xgboost([("tree.lambda_l2", 2.0), ("booster.n_iters", 200)])
    assert out == {"reg_lambda": 2.0, "n_estimators": 200}


def test_to_xgboost_accepts_params():
    """A Params renders to the same foreign dict as the equivalent pairs."""
    from bonsai.params import Params
    pairs = [("tree.lambda_l2", 2.0), ("booster.n_iters", 200)]
    assert interop.to_xgboost(Params.from_dict(dict(pairs))) == interop.to_xgboost(pairs)


def test_to_xgboost_picks_one_spelling_per_key():
    assert interop.to_xgboost([("booster.random_seed", 42)]) == {"seed": 42}


def test_to_xgboost_drops_the_implied_sampler():
    assert interop.to_xgboost([
        ("sampler.subsample", 0.8), ("dispatch.sampler_name", "bernoulli"),
    ]) == {"subsample": 0.8}


def test_to_xgboost_strict_names_every_unmappable_key():
    with pytest.raises(ValueError, match=r"tree\.min_data_in_leaf"):
        interop.to_xgboost([("tree.min_data_in_leaf", 20)])


def test_to_xgboost_rejects_the_levelwise_grower():
    with pytest.raises(ValueError, match="grow_policy"):
        interop.to_xgboost([("dispatch.grower_name", "levelwise")])


# to_lightgbm ======================================================================================

def test_to_lightgbm_uses_the_canonical_names():
    assert interop.to_lightgbm([
        ("tree.max_leaves", 63), ("tree.lambda_l2", 1.0),
        ("tree.min_data_in_leaf", 20), ("bin_mapper.max_bin", 255),
    ]) == {
        "num_leaves": 63, "lambda_l2": 1.0, "min_data_in_leaf": 20,
        "max_bin": 255,
    }


def test_to_lightgbm_drops_the_grower_name():
    assert interop.to_lightgbm([("dispatch.grower_name", "leafwise")]) == {}


# to_catboost ======================================================================================

def test_to_catboost_subtracts_the_border_fencepost():
    assert interop.to_catboost([("bin_mapper.max_bin", 255)]) == {
        "border_count": 254
    }


def test_to_catboost_maps_a_cuda_grower_to_its_growth_policy():
    assert interop.to_catboost([("dispatch.grower_name", "cuda_levelwise")]) == {
        "grow_policy": "SymmetricTree"
    }


# round trips ======================================================================================

def test_round_trip_preserves_the_shared_knobs():
    original = {
        "learning_rate": 0.05, "max_depth": 6, "reg_lambda": 1.0,
        "max_bin": 255, "seed": 42,
    }
    pairs = interop.from_xgboost(original)
    assert interop.to_xgboost(pairs) == {k: str(v) for k, v in original.items()}


def test_round_trip_survives_the_catboost_fencepost():
    pairs = interop.from_catboost({"border_count": 254})
    assert interop.to_catboost(pairs) == {"border_count": 254}


def test_every_native_key_is_reachable_from_the_reverse_index():
    """Each library's reverse index must be unambiguous: one outbound
    spelling per bonsai key, or a translated config would depend on table
    order.
    """
    for library in (interop._XGBOOST, interop._LIGHTGBM, interop._CATBOOST):
        canonical = [knob.native for knob in library.knobs if not knob.alias]
        assert len(canonical) == len(set(canonical)), library.name
