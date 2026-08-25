"""Tests for ``Dataset.subset(rows=...)``, the row view.

A view shares its parent's binned plane and selects rows out of it, so the
tests here are mostly equalities against the materialized copy the view is
supposed to replace. The copy is always built with ``reference=`` the parent:
a view carries the parent's cuts by construction, and a copy that refit its
own would differ for a reason that has nothing to do with the view.
"""

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

_PAIRS = {"booster.n_iters": "25", "tree.max_depth": "5"}
_READ_PAIRS = {"booster.n_iters": "15", "tree.max_depth": "4"}

requires_cuda = pytest.mark.skipif(
    not bonsai.cuda_available(), reason="no CUDA build or no visible device"
)


def _blocky_data(n=4000, f=10, seed=0):
    """(X, y) drawn from a small alphabet per column.

    Every column holds far fewer distinct values than ``max_bin``, so the cut
    points are the distinct values themselves and any large enough row subset
    fits the same cuts. That keeps ``reference=`` honest rather than merely
    convenient.
    """
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 20, size=(n, f)).astype(np.float32)
    y = (X[:, 0] * 0.3 + X[:, 1] * -0.2 + rng.normal(0, 0.5, n)).astype(np.float32)
    return X, y


def _model_bytes(params, dataset):
    """The saved artifact of a fit, as bytes."""
    with tempfile.NamedTemporaryFile(suffix=".msgpack", delete=False) as f:
        path = f.name
    bonsai.train(params, dataset).save(path)
    with open(path, "rb") as f:
        return f.read()


def test_full_range_view_is_byte_identical_to_a_plain_fit():
    """arange(n) selects every row in order, which is the identity: the fit
    must take exactly the path it takes without a view and produce the same
    artifact byte for byte."""
    X, y = _blocky_data()
    ds = bonsai.Dataset(X, y)
    view = ds.subset(rows=np.arange(len(X)))
    assert view.n_rows == len(X)
    assert view.shape == (len(X), X.shape[1])
    assert len(view) == len(X)
    assert "1 range" in repr(view)
    assert _model_bytes(_PAIRS, view) == _model_bytes(_PAIRS, ds)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_view_equals_a_materialized_copy(seed):
    """The point of the whole feature: selecting rows through a view must give
    the model that copying them out would, for each of the three descriptor
    shapes."""
    X, y = _blocky_data(seed=seed)
    n = len(X)
    rng = np.random.default_rng(seed + 100)
    shapes = {
        "range": np.arange(1000, 3000),
        "segments": np.concatenate([np.arange(0, 500), np.arange(1500, 2500),
                                    np.arange(3000, 3600)]),
        "gather": np.sort(rng.choice(n, size=2000, replace=False)),
    }
    for name, idx in shapes.items():
        ds = bonsai.Dataset(X, y)
        view = ds.subset(rows=idx)
        copy = bonsai.Dataset(X[idx], y[idx], reference=ds)
        assert view.n_rows == len(idx), name
        assert _model_bytes(_PAIRS, view) == _model_bytes(_PAIRS, copy), name


def test_rle_picks_the_right_representation():
    """The descriptor is chosen by run-length encoding what the caller passed,
    and repr says which one it landed on."""
    X, y = _blocky_data()
    ds = bonsai.Dataset(X, y)
    assert "1 range" in repr(ds.subset(rows=np.arange(len(X))))
    assert "1 range" in repr(ds.subset(rows=np.arange(100, 900)))
    assert "1 range" in repr(ds.subset(rows=slice(100, 900)))
    blocks = np.concatenate([np.arange(0, 100), np.arange(500, 700), np.arange(3000, 3100)])
    assert "3 segments" in repr(ds.subset(rows=blocks))
    rng = np.random.default_rng(0)
    scattered = np.sort(rng.choice(len(X), size=800, replace=False))
    assert "gathered" in repr(ds.subset(rows=scattered))
    # density is the selection's occupancy of its own bounding span
    assert "density 1.00" in repr(ds.subset(rows=np.arange(100, 900)))
    assert "shares parent plane" in repr(ds.subset(rows=np.arange(100, 900)))


def test_views_compose_and_base_names_the_root():
    """Selecting from a view selects from the parent's rows, so composing two
    selections is one selection through the first; and .base names the plane
    the rows live in, which is the root, not the intermediate."""
    X, y = _blocky_data()
    ds = bonsai.Dataset(X, y)
    a = np.arange(500, 3500)
    b = np.sort(np.random.default_rng(7).choice(len(a), size=1200, replace=False))
    chained = ds.subset(rows=a).subset(rows=b)
    direct = ds.subset(rows=a[b])
    assert chained.n_rows == len(b)
    assert _model_bytes(_PAIRS, chained) == _model_bytes(_PAIRS, direct)
    assert ds.base is None
    assert ds.subset(rows=a).base is ds
    assert chained.base is ds


def test_view_accepts_a_slice_and_a_boolean_mask():
    X, y = _blocky_data()
    ds = bonsai.Dataset(X, y)
    mask = np.zeros(len(X), dtype=bool)
    mask[1000:3000] = True
    by_mask = ds.subset(rows=mask)
    by_slice = ds.subset(rows=slice(1000, 3000))
    by_array = ds.subset(rows=np.arange(1000, 3000))
    assert by_mask.n_rows == by_slice.n_rows == by_array.n_rows == 2000
    assert _model_bytes(_PAIRS, by_mask) == _model_bytes(_PAIRS, by_array)
    assert _model_bytes(_PAIRS, by_slice) == _model_bytes(_PAIRS, by_array)


def test_view_outlives_its_parent():
    """The view shares the parent's plane through a shared pointer, so
    dropping the Python parent must not take the bins with it."""
    X, y = _blocky_data()
    ds = bonsai.Dataset(X, y)
    reference = _model_bytes(_PAIRS, bonsai.Dataset(X[500:3500], y[500:3500], reference=ds))
    v = ds.subset(rows=np.arange(500, 3500))
    del ds
    assert _model_bytes(_PAIRS, v) == reference


def test_chained_view_outlives_both_ancestors():
    X, y = _blocky_data()
    ds = bonsai.Dataset(X, y)
    idx = np.arange(500, 3500)
    inner = np.arange(0, 1000)
    reference = _model_bytes(
        _PAIRS, bonsai.Dataset(X[idx[inner]], y[idx[inner]], reference=ds))
    v = ds.subset(rows=idx)
    v2 = v.subset(rows=inner)
    del v
    del ds
    assert _model_bytes(_PAIRS, v2) == reference


def test_a_view_is_not_picklable():
    X, y = _blocky_data(n=500)
    v = bonsai.Dataset(X, y).subset(rows=np.arange(100, 400))
    with pytest.raises(Exception, match="not picklable"):
        pickle.dumps(v)


def test_duplicate_rows_are_weighted_not_silently_dropped():
    """A with-replacement draw repeats row ids. Each occurrence must count,
    which is what a materialized copy of the same draw does."""
    X, y = _blocky_data()
    rng = np.random.default_rng(3)
    idx = np.sort(rng.integers(0, len(X), size=len(X)))
    ds = bonsai.Dataset(X, y)
    view = ds.subset(rows=idx)
    copy = bonsai.Dataset(X[idx], y[idx], reference=ds)
    assert view.n_rows == len(idx)
    assert _model_bytes(_PAIRS, view) == _model_bytes(_PAIRS, copy)


def test_unsorted_rows_keep_the_callers_order():
    """The fit sums a node's rows in list order, so the caller's order is part
    of the answer: it is kept, not silently sorted, and the view still equals
    the copy of the same permutation."""
    X, y = _blocky_data()
    idx = np.random.default_rng(5).permutation(len(X))[:2000]
    ds = bonsai.Dataset(X, y)
    view = ds.subset(rows=idx)
    copy = bonsai.Dataset(X[idx], y[idx], reference=ds)
    assert _model_bytes(_PAIRS, view) == _model_bytes(_PAIRS, copy)


def test_subset_rejects_abuse():
    X, y = _blocky_data(n=500)
    ds = bonsai.Dataset(X, y)
    with pytest.raises(Exception, match="empty"):
        ds.subset(rows=np.array([], dtype=np.int64))
    with pytest.raises(Exception, match="empty"):
        ds.subset(rows=slice(100, 100))
    with pytest.raises(Exception, match="out of range"):
        ds.subset(rows=np.array([0, 1, 500]))
    with pytest.raises(Exception, match="negative"):
        ds.subset(rows=np.array([0, -1, 2]))
    with pytest.raises(Exception, match="mask"):
        ds.subset(rows=np.zeros(499, dtype=bool))
    with pytest.raises(Exception, match="rows="):
        ds.subset()
    with pytest.raises(TypeError):
        ds.subset(rows="everything")


def test_subset_columns_is_not_this_rung():
    X, y = _blocky_data(n=500)
    ds = bonsai.Dataset(X, y)
    with pytest.raises(NotImplementedError, match="rung 8"):
        ds.subset(columns=["f0", "f1"])


_BERNOULLI = {"dispatch.sampler_name": "bernoulli", "sampler.subsample": "0.5"}


def test_subsample_draws_within_the_view():
    """A sampler on a view draws out of the view's rows, not out of the
    parent's [0, n): the view is the candidate universe the draw is handed, so
    the same draw over the materialized copy gives the same model."""
    X, y = _blocky_data()
    idx = np.arange(3000, 4000)
    ds = bonsai.Dataset(X, y)
    pairs = dict(_PAIRS, **_BERNOULLI)
    view = ds.subset(rows=idx)
    copy = bonsai.Dataset(X[idx], y[idx], reference=ds)
    assert _model_bytes(pairs, view) == _model_bytes(pairs, copy)


def test_subsample_never_draws_a_row_outside_the_view():
    """Every id drawn is one of the view's. The rows outside carry a label the
    view's never take, so one of them reaching a histogram moves the fit off
    the constant the view alone implies: the view's labels are all equal, so a
    fit that saw only them has zero gradients and predicts that constant."""
    n = 4000
    rng = np.random.default_rng(0)
    X = rng.integers(0, 20, size=(n, 10)).astype(np.float32)
    y = np.full(n, -5.0, dtype=np.float32)
    idx = np.arange(1000, 3000)
    y[idx] = 5.0
    view = bonsai.Dataset(X, y).subset(rows=idx)
    model = bonsai.train(dict(_PAIRS, **_BERNOULLI), view)
    assert np.allclose(model.predict(X[idx]), 5.0, atol=1e-5)


def test_subsample_of_one_over_a_view_is_the_plain_view_fit():
    """subsample=1.0 copies the candidate list, so the draw is the view itself
    and the trees are the no-sampler fit's. The saved artifact records the
    sampler that was configured, so the equality is on what the trees answer
    rather than on the artifact's bytes."""
    X, y = _blocky_data()
    view = bonsai.Dataset(X, y).subset(rows=np.arange(1500, 3500))
    plain = bonsai.train(_PAIRS, view).predict(X)
    full = dict(_PAIRS, **{"dispatch.sampler_name": "bernoulli",
                           "sampler.subsample": "1.0"})
    assert np.array_equal(plain, bonsai.train(full, view).predict(X))


# --- the read side: a reader over a view answers one row per view row


def _row_shapes(n, seed=0):
    """The three descriptor forms as row-index arrays into ``n`` rows."""
    rng = np.random.default_rng(seed)
    return {
        "range": np.arange(1000, 3000),
        "segments": np.concatenate(
            [np.arange(0, 500), np.arange(1500, 2500), np.arange(3000, 3600)]
        ),
        "gather": np.sort(rng.choice(n, size=2000, replace=False)),
    }


def _trained(X, y, params=None):
    """A model and the Dataset it was fit on, sharing one set of cuts."""
    ds = bonsai.Dataset(X, y)
    return bonsai.train(params or _READ_PAIRS, ds), ds


@pytest.mark.parametrize("shape", ["range", "segments", "gather"])
def test_predict_over_a_view_equals_the_array_call(shape):
    """The cuts are the parent's, so a view routes in bin space, which is
    bit-identical to the raw walk over the same rows copied out."""
    X, y = _blocky_data()
    model, ds = _trained(X, y)
    idx = _row_shapes(len(X))[shape]
    got = model.predict(ds.subset(rows=idx))
    assert got.shape == (len(idx),)
    assert np.array_equal(got, model.predict(X[idx]))


@pytest.mark.parametrize("shape", ["range", "segments", "gather"])
def test_predict_leaf_and_staged_predict_over_a_view(shape):
    X, y = _blocky_data()
    model, ds = _trained(X, y)
    idx = _row_shapes(len(X))[shape]
    view = ds.subset(rows=idx)
    leaves = model.predict_leaf(view)
    assert leaves.shape == (len(idx), model.predict_leaf(X[:2]).shape[1])
    assert np.array_equal(leaves, model.predict_leaf(X[idx]))
    staged = model.staged_predict(view)
    assert staged.shape == (model.n_iters, len(idx))
    assert np.array_equal(staged, model.staged_predict(X[idx]))


@pytest.mark.parametrize("shape", ["range", "segments", "gather"])
def test_pred_contribs_over_a_view(shape):
    """Contributions keep their (n, n_features + 1) shape with the bias last,
    counted over the view's rows."""
    X, y = _blocky_data()
    model, ds = _trained(X, y)
    idx = _row_shapes(len(X))[shape]
    phi = model.pred_contribs(ds.subset(rows=idx))
    assert phi.shape == (len(idx), X.shape[1] + 1)
    np.testing.assert_allclose(phi, model.pred_contribs(X[idx]), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        phi.sum(axis=1), model.predict(X[idx]), rtol=1e-5, atol=1e-5
    )


def test_predict_proba_over_a_view():
    X, _ = _blocky_data()
    labels = (X[:, 0] > 10).astype(np.float32)
    ds = bonsai.Dataset(X, labels)
    model = bonsai.train(dict(_READ_PAIRS, **{"dispatch.objective_name": "logloss"}), ds)
    idx = _row_shapes(len(X))["gather"]
    got = model.predict_proba(ds.subset(rows=idx))
    assert got.shape == (len(idx),)
    assert np.array_equal(got, model.predict_proba(X[idx]))


def test_view_readers_follow_the_views_order():
    """A deliberately unsorted selection: the answer is in the caller's order,
    not the sorted one, which is the difference a gather makes visible."""
    X, y = _blocky_data()
    model, ds = _trained(X, y)
    idx = np.random.default_rng(11).permutation(len(X))[:1500]
    assert not np.array_equal(idx, np.sort(idx))
    view = ds.subset(rows=idx)
    assert np.array_equal(model.predict(view), model.predict(X[idx]))
    assert np.array_equal(model.predict_leaf(view), model.predict_leaf(X[idx]))
    assert not np.array_equal(model.predict(view), model.predict(X[np.sort(idx)]))


def test_a_view_of_a_view_reads_correctly():
    X, y = _blocky_data()
    model, ds = _trained(X, y)
    a = np.arange(500, 3500)
    b = np.random.default_rng(13).permutation(len(a))[:900]
    chained = ds.subset(rows=a).subset(rows=b)
    assert len(chained) == len(b)
    assert np.array_equal(model.predict(chained), model.predict(X[a][b]))


def test_eval_set_narrows_to_the_view():
    """The per-round loss must score the view's rows and nothing else, which a
    hand-computed MSE over the same rows pins down."""
    X, y = _blocky_data()
    ds = bonsai.Dataset(X, y)
    train_rows = np.arange(0, 3000)
    valid_rows = np.arange(3000, 4000)
    model = bonsai.train(
        _READ_PAIRS, ds.subset(rows=train_rows), eval_set=ds.subset(rows=valid_rows)
    )
    history = np.asarray(model.eval_history, dtype=np.float64)
    assert len(history) == int(_READ_PAIRS["booster.n_iters"])
    by_hand = np.mean(
        (model.predict(X[valid_rows]).astype(np.float64) - y[valid_rows]) ** 2
    )
    assert history[-1] == pytest.approx(by_hand, rel=1e-4)
    copies = bonsai.train(
        _READ_PAIRS,
        bonsai.Dataset(X[train_rows], y[train_rows], reference=ds),
        eval_set=bonsai.Dataset(X[valid_rows], y[valid_rows], reference=ds),
    )
    assert np.array_equal(history, np.asarray(copies.eval_history, dtype=np.float64))


def test_eval_set_view_drives_early_stopping():
    X, y = _blocky_data()
    ds = bonsai.Dataset(X, y)
    pairs = dict(_READ_PAIRS, **{"booster.n_iters": "200",
                                 "booster.early_stopping_rounds": "3"})
    model = bonsai.train(
        pairs,
        ds.subset(rows=np.arange(0, 3000)),
        eval_set=ds.subset(rows=np.arange(3000, 4000)),
    )
    assert model.n_iters <= 200
    assert len(model.eval_history) >= 1


def test_the_five_fold_walk_forward_loop_runs():
    """The loop the whole feature exists for: five folds trained and scored
    off one plane, no copy anywhere."""
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    sklearn_selection = pytest.importorskip("sklearn.model_selection")
    X, y = _blocky_data(n=6000)
    ds = bonsai.Dataset(X, y)
    pairs = dict(_READ_PAIRS, **{"booster.n_iters": "80"})
    scores = []
    for train_rows, valid_rows in sklearn_selection.TimeSeriesSplit(5).split(X):
        model = bonsai.train(
            pairs, ds.subset(rows=train_rows), eval_set=ds.subset(rows=valid_rows)
        )
        scores.append(
            sklearn_metrics.r2_score(
                y[valid_rows], model.predict(ds.subset(rows=valid_rows))
            )
        )
    assert len(scores) == 5
    assert min(scores) > 0.5


def test_foreign_cuts_still_refuse_a_view():
    """A view carries its parent's cuts, and cuts that are not the model's name
    other splits: the refusal stands rather than answering with the wrong
    leaves."""
    X, y = _blocky_data()
    ds = bonsai.Dataset(X, y)
    elsewhere = bonsai.Dataset((X * 3.0 + 1.0).astype(np.float32), y)
    model = bonsai.train(_READ_PAIRS, elsewhere)
    with pytest.raises(Exception, match="cut points"):
        model.predict(ds.subset(rows=np.arange(100, 900)))
    with pytest.raises(Exception, match="cut points"):
        model.pred_contribs(ds.subset(rows=np.arange(100, 900)))


def test_a_wide_binned_view_is_refused_as_an_eval_set():
    """Above 255 bins there is no row-major mirror, so the binned route the
    view needs does not exist; the raw walk would read the parent's rows."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(2000, 4)).astype(np.float32)
    y = (X[:, 0] + rng.normal(0, 0.1, 2000)).astype(np.float32)
    ds = bonsai.Dataset(X, y, max_bin=400)
    with pytest.raises(Exception, match="row view"):
        bonsai.train(
            _READ_PAIRS,
            ds.subset(rows=np.arange(0, 1500)),
            eval_set=ds.subset(rows=np.arange(1500, 2000)),
        )


def test_a_view_eval_set_is_refused_on_a_warm_start():
    """A warm start seeds from raw rows, which a view does not narrow."""
    X, y = _blocky_data()
    ds = bonsai.Dataset(X, y)
    with tempfile.NamedTemporaryFile(suffix=".msgpack", delete=False) as handle:
        path = handle.name
    bonsai.train(_READ_PAIRS, ds).save(path)
    with pytest.raises(Exception, match="row view"):
        bonsai.train(
            _READ_PAIRS,
            ds.subset(rows=np.arange(0, 3000)),
            eval_set=ds.subset(rows=np.arange(3000, 4000)),
            init_model=path,
        )


# --- what a view does not do: carry the rows it left out through the tree


_FINALIZE_WORKER = """
import sys
import numpy as np
import bonsai

parent, kept = int(sys.argv[1]), int(sys.argv[2])
rng = np.random.default_rng(0)
X = rng.random((parent, 8), dtype=np.float32)
y = (X[:, 0] * 2 + X[:, 1] + rng.normal(0, 0.1, parent)).astype(np.float32)
view = bonsai.Dataset(X, y, n_threads=4).subset(rows=np.arange(kept))
bonsai.train({"booster.n_iters": "20", "tree.max_depth": "6",
              "parallel.n_threads": "4"}, view)
print("DONE", flush=True)
"""


def _finalize_seconds(parent_rows, kept_rows):
    """The grow profile's finalize bucket for one view fit, in seconds.

    finalize holds the walk that carries rows outside the fit's row list
    through the finished tree. The profiler prints its line when the process
    exits, so the fit runs in a child.
    """
    env = {
        **os.environ,
        "BONSAI_GROW_PROFILE": "1",
        "PYTHONPATH": os.pathsep.join(sys.path),
    }
    proc = subprocess.run(
        [sys.executable, "-c", _FINALIZE_WORKER, str(parent_rows), str(kept_rows)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert "DONE" in proc.stdout, proc.stderr[-2000:]
    blocks = re.findall(r"grow-profile: (.+)", proc.stderr)
    fields = dict(kv.split("=") for kv in blocks[-1].split())
    return float(fields["finalize"].rstrip("s"))


def test_a_view_does_not_carry_the_rows_it_left_out():
    """The finalize bucket must not scale with what the view left behind. Both
    fits below train the same 50k rows; the second holds them inside a plane
    20x larger, and one that walked every row outside the view would pay for
    the 950k it never trained on: 0.16s against 0.01s, measured."""
    tight = _finalize_seconds(100_000, 50_000)
    roomy = _finalize_seconds(1_000_000, 50_000)
    assert roomy <= tight + 0.05


def test_leaf_renewal_over_a_view_reads_only_the_views_rows():
    """MAE renews each leaf from the residuals of the rows it covers, and the
    rows a view left out are covered by nothing its fit produced. They keep a
    stale leaf id and a stale training score, so the renewal has to be over the
    view's rows for a view fit to equal the copy it stands in for."""
    X, y = _blocky_data()
    idx = np.arange(1000, 3000)
    ds = bonsai.Dataset(X, y)
    pairs = dict(_PAIRS, **{"dispatch.objective_name": "mae"})
    view = ds.subset(rows=idx)
    copy = bonsai.Dataset(X[idx], y[idx], reference=ds)
    assert _model_bytes(pairs, view) == _model_bytes(pairs, copy)


@requires_cuda
def test_a_view_of_a_device_resident_parent_reads_through_the_bin_route():
    """The case the bin route exists for: the plane is on the device and there
    is no host matrix to fall back to, so a view must route its own rows."""
    X, y = _blocky_data()
    ds = bonsai.Dataset(X, y, device="cuda")
    model = bonsai.train(dict(_READ_PAIRS, **{"dispatch.grower_name": "cuda"}), ds)
    idx = _row_shapes(len(X))["gather"]
    view = ds.subset(rows=idx)
    assert len(view) == len(idx)
    np.testing.assert_allclose(
        model.predict(view), model.predict(X[idx]), rtol=1e-5, atol=1e-5
    )
    assert model.predict_leaf(view).shape[0] == len(idx)
    assert model.pred_contribs(view).shape == (len(idx), X.shape[1] + 1)
