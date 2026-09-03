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


def test_view_accepts_a_stepped_slice_and_an_unsigned_array():
    X, y = _blocky_data()
    ds = bonsai.Dataset(X, y)
    by_slice = ds.subset(rows=slice(1000, 3000, 7))
    by_unsigned = ds.subset(rows=np.arange(1000, 3000, 7, dtype=np.uint32))
    assert by_slice.n_rows == by_unsigned.n_rows == len(range(1000, 3000, 7))
    assert _model_bytes(_PAIRS, by_slice) == _model_bytes(_PAIRS, by_unsigned)


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
    with pytest.raises(Exception, match="one dimension"):
        ds.subset(rows=np.arange(4).reshape(2, 2))


# Column selection ================================================================================


def test_columns_by_name_keeps_them_in_the_order_asked_for():
    X, y = _blocky_data(n=500, f=6)
    ds = bonsai.Dataset(X, y)
    sub = ds.subset(columns=["f4", "f1"])
    assert sub.n_features == 2
    assert sub.feature_names == ["f4", "f1"]
    assert sub.n_rows == len(X)
    # A rewrite owns its plane, so it is nobody's view.
    assert sub.base is None


@pytest.mark.parametrize(
    "columns",
    [
        ["f4", "f1"],
        np.array([4, 1]),
        [4, 1],
    ],
)
def test_columns_accepts_names_and_indices(columns):
    X, y = _blocky_data(n=500, f=6)
    ds = bonsai.Dataset(X, y)
    sub = ds.subset(columns=columns)
    assert sub.feature_names == ["f4", "f1"]


def test_columns_accepts_a_slice_and_a_boolean_mask():
    X, y = _blocky_data(n=500, f=6)
    ds = bonsai.Dataset(X, y)
    assert ds.subset(columns=slice(1, 4)).feature_names == ["f1", "f2", "f3"]
    mask = np.array([False, True, False, False, True, False])
    assert ds.subset(columns=mask).feature_names == ["f1", "f4"]


def test_a_column_selection_fits_like_a_matrix_of_those_columns():
    """The claim the whole rewrite rests on: gathering bins out of a plane
    gives the same model as binning a matrix that only ever held those
    columns, so a selection is a real dataset and not an approximation."""
    X, y = _blocky_data(n=2000, f=8)
    keep = [5, 0, 3]
    ds = bonsai.Dataset(X, y)
    sub = ds.subset(columns=keep)
    direct = bonsai.Dataset(
        np.ascontiguousarray(X[:, keep]), y, feature_names=[f"f{i}" for i in keep]
    )
    assert _model_bytes(_PAIRS, sub) == _model_bytes(_PAIRS, direct)


def test_keeping_every_column_in_order_is_the_same_fit():
    X, y = _blocky_data(n=1000, f=6)
    ds = bonsai.Dataset(X, y)
    same = ds.subset(columns=list(range(6)))
    assert _model_bytes(_PAIRS, same) == _model_bytes(_PAIRS, ds)


def test_rows_and_columns_commute():
    """One order materializes the fold and the other views it; the model must
    not be able to tell which route it came by."""
    X, y = _blocky_data(n=2000, f=8)
    rows = np.arange(400, 1500)
    cols = ["f6", "f2", "f0"]
    ds = bonsai.Dataset(X, y)
    rows_first = ds.subset(rows=rows).subset(columns=cols)
    cols_first = ds.subset(columns=cols).subset(rows=rows)
    assert rows_first.n_rows == cols_first.n_rows == len(rows)
    # Only one of them views anything, which is the point of the check.
    assert rows_first.base is None
    assert cols_first.base is not None
    assert _model_bytes(_PAIRS, rows_first) == _model_bytes(_PAIRS, cols_first)


def test_rows_and_columns_in_one_call():
    X, y = _blocky_data(n=2000, f=8)
    rows = np.arange(400, 1500)
    cols = ["f6", "f2", "f0"]
    ds = bonsai.Dataset(X, y)
    both = ds.subset(rows=rows, columns=cols)
    assert both.n_rows == len(rows)
    assert both.feature_names == cols
    staged = ds.subset(rows=rows).subset(columns=cols)
    assert _model_bytes(_PAIRS, both) == _model_bytes(_PAIRS, staged)


def test_a_column_selection_outlives_its_parent():
    X, y = _blocky_data(n=1000, f=6)
    ds = bonsai.Dataset(X, y)
    sub = ds.subset(columns=["f0", "f3"])
    expected = _model_bytes(_PAIRS, sub)
    del ds
    assert _model_bytes(_PAIRS, sub) == expected


def test_a_column_selection_serves_its_own_bins():
    """cols_ and row_major_ are shared across Dataset copies by design. A
    selection built by copying the parent would keep serving the parent's
    columns, and a model trained on two features would split on eight."""
    X, y = _blocky_data(n=1500, f=8)
    ds = bonsai.Dataset(X, y)
    sub = ds.subset(columns=["f7", "f2"])
    model = bonsai.train(_PAIRS, sub)
    # Predicting the selection through the model trained on it must agree with
    # predicting the same two columns passed as a bare matrix. A selection
    # still serving eight columns would route on the wrong ones and diverge.
    direct = np.ascontiguousarray(X[:, [7, 2]])
    np.testing.assert_allclose(model.predict(sub), model.predict(direct), rtol=1e-6)


def test_a_column_selection_is_not_picklable():
    X, y = _blocky_data(n=300, f=4)
    sub = bonsai.Dataset(X, y).subset(columns=[0, 1])
    with pytest.raises(Exception, match="not picklable"):
        pickle.dumps(sub)


def test_subset_columns_rejects_abuse():
    X, y = _blocky_data(n=500, f=6)
    ds = bonsai.Dataset(X, y)
    with pytest.raises(Exception, match="no features"):
        ds.subset(columns=[])
    with pytest.raises(IndexError):
        ds.subset(columns=[0, 6])
    with pytest.raises(IndexError):
        ds.subset(columns=[-1])  # feature ids do not wrap, same as row ids
    with pytest.raises(KeyError):
        ds.subset(columns=["nope"])
    with pytest.raises(Exception, match="mask"):
        ds.subset(columns=np.array([True, False]))
    with pytest.raises(TypeError):
        ds.subset(columns=3.5)


def test_a_model_trained_on_a_selection_refuses_the_parent():
    """The parent's cuts describe eight columns and the model's describe two;
    routing the parent through it would read the wrong feature at every node."""
    X, y = _blocky_data(n=1000, f=8)
    ds = bonsai.Dataset(X, y)
    model = bonsai.train(_READ_PAIRS, ds.subset(columns=["f0", "f5"]))
    with pytest.raises(ValueError, match="columns must match"):
        model.predict(ds)


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


_DEVICE_SCORE_WORKER = """
import sys
import numpy as np
import bonsai
sys.path.insert(0, sys.argv[1])
from test_dataset_views import _blocky_data, _row_shapes, _READ_PAIRS

X, y = _blocky_data()
ds = bonsai.Dataset(X, y, device="cuda")
model = bonsai.train(dict(_READ_PAIRS, **{"dispatch.grower_name": "cuda_depthwise"}), ds)
idx = _row_shapes(len(X))[sys.argv[2]]
view = ds.subset(rows=idx)
assert len(view) == len(idx)
np.testing.assert_allclose(model.predict(view), model.predict(X[idx]), rtol=1e-5, atol=1e-5)
np.testing.assert_allclose(
    model.pred_contribs(view), model.pred_contribs(X[idx]), rtol=1e-4, atol=1e-4
)
assert model.predict_leaf(view).shape[0] == len(idx)
print("DONE", flush=True)
"""


def _device_scoring_lines(shape):
    """stderr of one child that scores a view of a device-resident parent.

    The device profile flag is read once per process, so the child is where
    the engagement lines can be asserted.
    """
    env = {
        **os.environ,
        "BONSAI_CUDA_PROFILE": "1",
        "PYTHONPATH": os.pathsep.join(sys.path),
    }
    here = os.path.dirname(os.path.abspath(__file__))
    proc = subprocess.run(
        [sys.executable, "-c", _DEVICE_SCORE_WORKER, here, shape],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert "DONE" in proc.stdout, proc.stderr[-2000:]
    return proc.stderr


@requires_cuda
@pytest.mark.parametrize("shape", ["range", "segments", "gather"])
def test_a_view_of_a_device_resident_parent_predicts_on_the_device(shape):
    """The plane is on the device and there is no host matrix to fall back
    to. A view hands its row ids to the device walks, which read the parent
    plane in place: the predict and SHAP lines report the view's rows against
    the parent's plane, and the parent is never pulled to the host."""
    n_rows = len(_blocky_data()[0])
    idx = _row_shapes(n_rows)[shape]
    lines = _device_scoring_lines(shape)
    assert re.search(rf"cuda-predict: .*rows={len(idx)} plane={n_rows} ", lines), lines
    assert re.search(rf"cuda-shap: .*rows={len(idx)} plane={n_rows} ", lines), lines


_DEVICE_EVAL_WORKER = """
import sys
import numpy as np
import bonsai
sys.path.insert(0, sys.argv[1])
from test_dataset_views import _blocky_data, _row_shapes, _READ_PAIRS

X, y = _blocky_data()
Xh, yh = _blocky_data(seed=1)
train = bonsai.Dataset(X, y, device="cuda")
hold = bonsai.Dataset(Xh, yh, reference=train)
idx = _row_shapes(len(Xh))[sys.argv[2]]
pairs = dict(
    _READ_PAIRS,
    **{
        "dispatch.grower_name": "cuda_depthwise",
        "booster.n_iters": "60",
        "booster.early_stopping_rounds": "5",
    },
)
viewed = bonsai.train(pairs, train, eval_set=hold.subset(rows=idx))
copied = bonsai.train(
    pairs, train, eval_set=bonsai.Dataset(Xh[idx], yh[idx], reference=train)
)
assert viewed.n_iters == copied.n_iters, (viewed.n_iters, copied.n_iters)
np.testing.assert_allclose(viewed.eval_history, copied.eval_history, rtol=1e-5, atol=1e-5)
print("DONE", flush=True)
"""


def _device_eval_lines(shape):
    """stderr of one child that fits against an eval_set view of a
    device-resident holdout, under the device profile flag."""
    env = {
        **os.environ,
        "BONSAI_CUDA_PROFILE": "1",
        "PYTHONPATH": os.pathsep.join(sys.path),
    }
    here = os.path.dirname(os.path.abspath(__file__))
    proc = subprocess.run(
        [sys.executable, "-c", _DEVICE_EVAL_WORKER, here, shape],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert "DONE" in proc.stdout, proc.stderr[-2000:]
    return proc.stderr


@requires_cuda
@pytest.mark.parametrize("shape", ["range", "segments", "gather"])
def test_an_eval_set_view_of_a_device_resident_parent_scores_on_the_device(shape):
    """One holdout on the device, split by view into the eval set: each round
    walks the parent plane in place through the view's ids, so the plane is
    adopted rather than retiled through the host, and the history (and the
    early stop it drives) matches a copied eval set within device rounding."""
    n_rows = len(_blocky_data()[0])
    idx = _row_shapes(n_rows)[shape]
    lines = _device_eval_lines(shape)
    armed = rf"cuda eval plane armed: rows={len(idx)} plane={n_rows} adopted=1 "
    assert re.search(armed, lines), lines


# Reorder =========================================================================================


def test_reorder_lays_the_rows_out_in_the_given_order():
    X, y = _blocky_data(n=1000, f=6)
    ds = bonsai.Dataset(X, y)
    order = np.random.default_rng(3).permutation(len(X))
    laid = ds.reorder(rows=order)
    assert laid.n_rows == len(X)
    assert laid.n_features == 6
    # A rewrite owns its plane, so it views nothing.
    assert laid.base is None
    # Row i of the result is row order[i] of the parent, which is exactly what
    # a caller who reordered the matrix themselves would have.
    direct = bonsai.Dataset(np.ascontiguousarray(X[order]), y[order])
    assert _model_bytes(_PAIRS, laid) == _model_bytes(_PAIRS, direct)


def test_reorder_by_the_identity_is_the_same_fit():
    X, y = _blocky_data(n=800, f=5)
    ds = bonsai.Dataset(X, y)
    assert _model_bytes(_PAIRS, ds.reorder(rows=np.arange(len(X)))) == _model_bytes(
        _PAIRS, ds
    )


def test_a_permutation_moves_the_model_only_by_rounding():
    """A histogram bin sums its rows in list order and float addition is not
    associative, so a permutation is not byte-identical. It must still be the
    same model to float32 rounding, which is the real claim."""
    X, y = _blocky_data(n=3000, f=8)
    ds = bonsai.Dataset(X, y)
    order = np.random.default_rng(5).permutation(len(X))
    plain = bonsai.train(_PAIRS, ds).predict(X)
    laid = bonsai.train(_PAIRS, ds.reorder(rows=order)).predict(X)
    np.testing.assert_allclose(plain, laid, rtol=0, atol=1e-4)


def test_reorder_makes_a_scattered_fold_contiguous():
    """The point of reordering: a fold that was scattered through the parent
    becomes a range, which the fill reads as a subspan rather than a gather."""
    X, y = _blocky_data(n=2000, f=6)
    groups = np.arange(len(X)) % 5
    order = np.argsort(groups, kind="stable")
    ds = bonsai.Dataset(X, y).reorder(rows=order)
    # Group 0 now occupies the first fifth, in one piece.
    fold = ds.subset(rows=slice(0, 400))
    assert "range" in repr(fold)
    assert fold.n_rows == 400
    # And it is the same 400 rows the scattered selection would have named.
    scattered = bonsai.Dataset(X, y).subset(rows=np.flatnonzero(groups == 0))
    assert _model_bytes(_PAIRS, fold) == _model_bytes(_PAIRS, scattered)


def test_reorder_composes_with_a_column_selection():
    X, y = _blocky_data(n=1000, f=6)
    ds = bonsai.Dataset(X, y)
    order = np.random.default_rng(7).permutation(len(X))
    laid = ds.reorder(rows=order).subset(columns=["f3", "f0"])
    assert laid.feature_names == ["f3", "f0"]
    assert laid.n_rows == len(X)
    direct = bonsai.Dataset(
        np.ascontiguousarray(X[np.ix_(order, [3, 0])]), y[order],
        feature_names=["f3", "f0"],
    )
    assert _model_bytes(_PAIRS, laid) == _model_bytes(_PAIRS, direct)


def test_reorder_demands_a_permutation():
    X, y = _blocky_data(n=500, f=4)
    ds = bonsai.Dataset(X, y)
    with pytest.raises(ValueError, match="permutation"):
        ds.reorder(rows=np.arange(100))  # too few
    with pytest.raises(ValueError, match="permutation"):
        ds.reorder(rows=np.zeros(500, dtype=np.int64))  # a row twice
    with pytest.raises(IndexError):
        ds.reorder(rows=np.full(500, 500, dtype=np.int64))
    with pytest.raises(Exception, match="rows="):
        ds.reorder()


def test_reorder_of_a_view_lays_out_only_the_views_rows():
    X, y = _blocky_data(n=1000, f=5)
    ds = bonsai.Dataset(X, y)
    view = ds.subset(rows=np.arange(200, 700))
    order = np.random.default_rng(9).permutation(500)
    laid = view.reorder(rows=order)
    assert laid.n_rows == 500
    assert laid.base is None
    picked = np.arange(200, 700)[order]
    direct = bonsai.Dataset(np.ascontiguousarray(X[picked]), y[picked])
    assert _model_bytes(_PAIRS, laid) == _model_bytes(_PAIRS, direct)


def test_reorder_outlives_its_parent():
    X, y = _blocky_data(n=600, f=4)
    ds = bonsai.Dataset(X, y)
    laid = ds.reorder(rows=np.arange(len(X))[::-1])
    expected = _model_bytes(_PAIRS, laid)
    del ds
    assert _model_bytes(_PAIRS, laid) == expected


@pytest.mark.parametrize(("name", "idx"), [
    ("contiguous", np.arange(0, 1500)),
    ("strided", np.arange(0, 4000, 3)),
])
def test_a_view_leaves_out_of_view_scores_alone(name, idx):
    """The contract stated on RecycledOutputs, pinned.

    The score update walks the plane, not the view, because a repeated row id
    must not be advanced twice. That makes the rows a view omits readable, and
    they are only harmless because nothing ever writes their slot: they hold
    zero from the first grow's resize and stay there. If any writer starts
    touching them, a view's model stops matching the materialized copy of the
    same rows, which is what this asserts across enough rounds for a drift to
    compound. Strided as well as contiguous, so a stride bug cannot hide
    behind the omitted rows being one suffix.
    """
    X, y = _blocky_data()
    ds = bonsai.Dataset(X, y)
    view = ds.subset(rows=idx)
    copy = bonsai.Dataset(X[idx], y[idx], reference=ds)
    pairs = dict(_PAIRS, **{"booster.n_iters": "60"})
    assert _model_bytes(pairs, view) == _model_bytes(pairs, copy), name
