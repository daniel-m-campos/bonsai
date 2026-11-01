"""Multi-target regression: a 2-D y through BonsaiRegressor and the Dataset API."""

from __future__ import annotations

import os
import pickle
import re
import subprocess
import sys
import tempfile

import bonsai
import numpy as np
import pytest

K = 3
N_ITERS = 12

requires_cuda = pytest.mark.skipif(
    not bonsai.cuda_available(), reason="no CUDA build or no visible device"
)


def _regressor(
    n_iters: int = N_ITERS,
    early_stopping_rounds: int = 0,
    grower: str = "depthwise",
    device: str | None = None,
) -> bonsai.BonsaiRegressor:
    """A small regressor every test fits, depthwise on the host unless told otherwise."""
    return bonsai.BonsaiRegressor(
        n_iters=n_iters,
        learning_rate=0.3,
        max_depth=3,
        grower=grower,
        device=device,
        n_threads=2,
        early_stopping_rounds=early_stopping_rounds,
    )


def _data(n=600, p=6, seed=1):
    """(X, Y) float32 arrays with three correlated targets in the first columns."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, p), dtype=np.float32)
    Y = np.stack(
        [
            X[:, 0] * 2 + X[:, 1],
            np.sin(6 * X[:, 1]) - X[:, 2],
            np.where(X[:, 0] > 0.5, 1.0, -1.0) * X[:, 2],
        ],
        axis=1,
    )
    return X, (Y + rng.normal(0, 0.05, Y.shape)).astype(np.float32)


def test_regressor_fits_a_2d_y_and_predicts_one_column_per_target():
    X, Y = _data()
    m = _regressor().fit(X, Y)
    assert m.n_outputs_ == K
    assert m._fitted().n_targets == K
    P = m.predict(X)
    assert P.shape == (len(X), K)
    for k in range(K):
        single = _regressor().fit(X, Y[:, k])
        np.testing.assert_array_equal(P[:, k], single.predict(X))


def test_regressor_scores_the_mean_r2_over_targets():
    X, Y = _data()
    m = _regressor().fit(X, Y)
    P = m.predict(X)
    per = [
        1 - np.sum((Y[:, k] - P[:, k]) ** 2) / np.sum((Y[:, k] - Y[:, k].mean()) ** 2)
        for k in range(K)
    ]
    assert m.score(X, Y) == pytest.approx(float(np.mean(per)), rel=1e-5)
    assert m.score(X, Y) > 0.9


def test_regressor_staged_and_contribs_carry_the_target_axis():
    X, Y = _data()
    m = _regressor().fit(X, Y)
    staged = m.staged_predict(X)
    assert staged.shape == (N_ITERS, len(X), K)
    np.testing.assert_array_equal(staged[-1], m.predict(X))
    contribs = m.pred_contribs(X)
    assert contribs.shape == (len(X), K, X.shape[1] + 1)
    np.testing.assert_allclose(contribs.sum(axis=2), m.predict(X), rtol=1e-4, atol=1e-4)


def test_regressor_single_column_2d_y_is_one_target():
    X, Y = _data()
    m = _regressor().fit(X, Y[:, :1])
    assert m.n_outputs_ == 1
    assert m.predict(X).shape == (len(X),)
    with pytest.raises(ValueError, match="1- or 2-dimensional"):
        _regressor().fit(X, Y[:, :, None])


def test_regressor_round_trips_through_pickle_and_file():
    X, Y = _data()
    m = _regressor().fit(X, Y)
    want = m.predict(X)
    clone = pickle.loads(pickle.dumps(m))
    assert clone.n_outputs_ == K
    np.testing.assert_array_equal(clone.predict(X), want)
    with tempfile.TemporaryDirectory() as td:
        path = f"{td}/m.msgpack"
        m.save(path)
        loaded = bonsai.BonsaiRegressor.from_file(path)
    assert loaded.n_outputs_ == K
    np.testing.assert_array_equal(loaded.predict(X), want)


def test_regressor_early_stops_on_a_2d_eval_set():
    X, Y = _data()
    Xv, Yv = _data(n=200, seed=2)
    m = _regressor(n_iters=60, early_stopping_rounds=3).fit(X, Y, eval_set=(Xv, Yv))
    (metrics,) = m.evals_result().values()
    (losses,) = metrics.values()
    assert m.n_iters_ < 60
    assert len(losses) == m.n_iters_ + 3
    assert m.predict(Xv).shape == (len(Xv), K)
    with pytest.raises(ValueError, match="eval_set y has"):
        _regressor().fit(X, Y, eval_set=(Xv, Yv[:, :2]))


def test_dataset_api_fits_and_predicts_a_target_block():
    X, Y = _data()
    ds = bonsai.Dataset(X, Y)
    pairs = {"booster.n_iters": 12, "booster.learning_rate": 0.3, "tree.max_depth": 3}
    model = bonsai.train(pairs, ds)
    assert model.n_targets == K
    P = np.asarray(model.predict(ds))
    assert P.shape == (len(X), K)
    np.testing.assert_array_equal(P, np.asarray(model.predict(X)))
    np.testing.assert_array_equal(P, np.asarray(bonsai.train(pairs, X, Y).predict(X)))
    valid = bonsai.Dataset(X[:100], Y[:100], reference=ds)
    stopped = bonsai.train({**pairs, "booster.early_stopping_rounds": 2}, ds, eval_set=valid)
    assert np.asarray(stopped.predict(valid)).shape == (100, K)


def test_dataset_api_refuses_a_width_that_disagrees_with_the_config():
    X, Y = _data()
    with pytest.raises(ValueError, match=r"objective\.n_targets is 2 and y carries 3"):
        bonsai.train({"objective.n_targets": 2}, bonsai.Dataset(X, Y))
    with pytest.raises(ValueError, match=r"objective\.n_targets is 3 and y carries 1"):
        bonsai.train({"objective.n_targets": 3}, bonsai.Dataset(X, Y[:, 0]))
    binary = (Y > 0).astype(np.float32)
    with pytest.raises((ValueError, RuntimeError), match="regression objectives"):
        bonsai.train({"dispatch.objective_name": "logloss"}, bonsai.Dataset(X, binary))


def test_an_eval_dataset_of_another_target_count_is_refused():
    X, Y = _data()
    pairs = {"booster.n_iters": 3}
    train3 = bonsai.Dataset(X, Y)
    for labels in (Y[:, 0], np.ascontiguousarray(Y[:, :2])):
        valid = bonsai.Dataset(X, labels, reference=train3)
        with pytest.raises(ValueError, match="eval_set carries"):
            bonsai.train(pairs, train3, eval_set=valid)
    train1 = bonsai.Dataset(X, Y[:, 0])
    with pytest.raises(ValueError, match="1 target per row and this fit trains 3"):
        bonsai.train(
            pairs, bonsai.Dataset(X, Y), eval_set=bonsai.Dataset(X, Y[:, 0], reference=train1)
        )
    with pytest.raises(ValueError, match="3 targets per row and this fit trains 1"):
        _regressor(n_iters=3).fit(X, Y[:, 0], eval_set=bonsai.Dataset(X, Y, reference=train1))


def test_a_row_view_eval_set_scores_its_own_target_rows():
    X, Y = _data()
    full = bonsai.Dataset(X, Y)
    pairs = {"booster.n_iters": 4, "parallel.n_threads": 2}
    for valid_rows in (np.arange(400, 600), np.arange(400, 600)[::-1]):
        model = bonsai.train(
            pairs, full.subset(rows=slice(0, 400)), eval_set=full.subset(rows=valid_rows)
        )
        by_hand = [
            float(((model.predict(X[400:], num_iteration=r + 1) - Y[400:]) ** 2).mean())
            for r in range(4)
        ]
        np.testing.assert_allclose(model.eval_history, by_hand, rtol=1e-4)


def test_regressor_refuses_an_eval_y_whose_rows_disagree_with_eval_x():
    X, Y = _data()
    Xv, Yv = _data(n=60, seed=2)
    Yv2 = np.ascontiguousarray(Yv[:, :2])
    with pytest.raises(ValueError, match="eval_set y has 2 rows and eval_set X has 60"):
        _regressor(n_iters=3).fit(X, Y[:, :2], eval_set=(Xv, Yv2.T))
    with pytest.raises(ValueError, match="eval_set y has 30 rows and eval_set X has 60"):
        _regressor(n_iters=3).fit(X, Y[:, :2], eval_set=(Xv, Yv2.reshape(30, 4)))


@requires_cuda
@pytest.mark.parametrize("grower", ["depthwise", "levelwise", "leafwise"])
def test_cuda_grower_fits_a_2d_y_and_matches_k_device_fits(grower, monkeypatch):
    """A K-wide device fit equals K single device fits bit for bit on both
    device paths: resident (the default) and through the host-objective hatch."""
    X, Y = _data()
    m = _regressor(grower=grower, device="cuda").fit(X, Y)
    assert m.n_outputs_ == K
    P = m.predict(X)
    assert P.shape == (len(X), K)
    assert m.score(X, Y) > 0.9
    for k in range(K):
        resident = _regressor(grower=grower, device="cuda").fit(X, Y[:, k])
        np.testing.assert_array_equal(P[:, k], resident.predict(X))
    monkeypatch.setenv("BONSAI_HOST_OBJECTIVE", "1")
    hatched = _regressor(grower=grower, device="cuda").fit(X, Y).predict(X)
    assert hatched.shape == P.shape
    for k in range(K):
        single = _regressor(grower=grower, device="cuda").fit(X, Y[:, k])
        np.testing.assert_array_equal(hatched[:, k], single.predict(X))
    contribs = m.pred_contribs(X)
    assert contribs.shape == (len(X), K, X.shape[1] + 1)
    np.testing.assert_allclose(contribs.sum(axis=2), P, rtol=1e-4, atol=1e-4)


@requires_cuda
def test_cuda_dataset_api_predicts_a_k_wide_model_on_the_host():
    X, Y = _data()
    ds = bonsai.Dataset(X, Y)
    pairs = {
        "dispatch.grower_name": "cuda_depthwise",
        "booster.n_iters": 12,
        "booster.learning_rate": 0.3,
        "tree.max_depth": 3,
    }
    model = bonsai.train(pairs, ds)
    assert model.n_targets == K
    P = np.asarray(model.predict(ds))
    assert P.shape == (len(X), K)
    np.testing.assert_array_equal(P, np.asarray(model.predict(X)))
    assert np.asarray(model.pred_contribs(ds)).shape == (len(X), K, X.shape[1] + 1)


_DEVICE_K_WIDE_WORKER = """
import sys
import numpy as np
import bonsai
sys.path.insert(0, sys.argv[1])
from test_multi_target import _data, K

X, Y = _data(n=4000)
train = bonsai.Dataset(X, Y, device="cuda")
pairs = {
    "dispatch.grower_name": sys.argv[2],
    "booster.n_iters": 12,
    "booster.learning_rate": 0.3,
    "tree.max_depth": 3,
}
model = bonsai.train(pairs, train)
P = np.asarray(model.predict(train))
assert P.shape == (len(X), K), P.shape
np.testing.assert_allclose(P, np.asarray(model.predict(X)), rtol=1e-5, atol=1e-5)
C = np.asarray(model.pred_contribs(train))
assert C.shape == (len(X), K, X.shape[1] + 1), C.shape
np.testing.assert_allclose(C, np.asarray(model.pred_contribs(X)), rtol=1e-4, atol=1e-4)
staged = np.asarray(model.predict(train, num_iteration=5))
host5 = np.asarray(model.predict(X, num_iteration=5))
np.testing.assert_allclose(staged, host5, rtol=1e-5, atol=1e-5)
print("DONE", flush=True)
"""


@requires_cuda
@pytest.mark.parametrize("grower", ["cuda_depthwise", "cuda_levelwise"])
def test_a_k_wide_model_predicts_and_explains_on_the_device(grower):
    """A device-resident Dataset has no host matrix to fall back to: the
    K-wide model walks it as K width-1 plans, one per target, matching the
    host walk, and the profile lines report K device predicts and K device
    SHAP walks over the plane."""
    env = {
        **os.environ,
        "BONSAI_CUDA_PROFILE": "1",
        "PYTHONPATH": os.pathsep.join(sys.path),
    }
    here = os.path.dirname(os.path.abspath(__file__))
    proc = subprocess.run(
        [sys.executable, "-c", _DEVICE_K_WIDE_WORKER, here, grower],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert "DONE" in proc.stdout, proc.stderr[-2000:]
    assert len(re.findall(r"cuda-predict: ", proc.stderr)) == 2 * K, proc.stderr[-2000:]
    assert len(re.findall(r"cuda-shap: ", proc.stderr)) == K, proc.stderr[-2000:]


_DEVICE_K_WIDE_EVAL_WORKER = """
import sys
import numpy as np
import bonsai
sys.path.insert(0, sys.argv[1])
from test_multi_target import _data, K

X, Y = _data(n=4000)
Xv, Yv = _data(n=1000, seed=7)
m = bonsai.BonsaiRegressor(
    n_iters=40,
    learning_rate=0.3,
    max_depth=3,
    grower=sys.argv[2],
    device="cuda",
    n_threads=2,
    early_stopping_rounds=3,
).fit(X, Y, eval_set=(Xv, Yv))
(metrics,) = m.evals_result().values()
(losses,) = metrics.values()
assert m.n_outputs_ == K
staged = np.asarray(m.staged_predict(Xv))
pooled = ((staged - Yv[None]) ** 2).mean(axis=(1, 2))
assert len(losses) >= len(pooled), (len(losses), len(pooled))
np.testing.assert_allclose(np.asarray(losses[: len(pooled)]), pooled, rtol=1e-4, atol=1e-6)
print("DONE", m.n_iters_, flush=True)
"""


@requires_cuda
@pytest.mark.parametrize("grower", ["cuda_depthwise", "cuda_levelwise", "cuda_leafwise"])
def test_a_k_wide_eval_set_is_scored_on_the_device(grower):
    """Early stopping on a CUDA grower scores a K-wide eval set on the device:
    the plane arms once, and the loss series is the pooled squared error the
    host computes from the staged predictions."""
    env = {
        **os.environ,
        "BONSAI_CUDA_PROFILE": "1",
        "PYTHONPATH": os.pathsep.join(sys.path),
    }
    here = os.path.dirname(os.path.abspath(__file__))
    proc = subprocess.run(
        [sys.executable, "-c", _DEVICE_K_WIDE_EVAL_WORKER, here, grower],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert "DONE" in proc.stdout, proc.stderr[-2000:]
    assert len(re.findall(r"cuda eval plane armed", proc.stderr)) == 1, proc.stderr[-2000:]
