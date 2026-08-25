"""Tests for the cross-validation suite's fold construction and dispatch."""

from __future__ import annotations

import numpy as np
import pytest
from bonsai.bench import cv, runlog, runners


def _cell(**over):
    # gen_data lays informative features out in blocks of five and weights
    # whole blocks, so fewer than five leaves y constant and every r2 NaN.
    base = {"axis": "folds", "rows": 1000, "cols": 10, "bins": 255,
            "bins_effective": 255, "depth": 4, "iters": 5, "lr": 0.1,
            "informative": 5, "n_test": 1, "seed": 42, "folds": 4}
    base.update(over)
    return base


# Fold construction ===============================================================================


def test_walk_forward_folds_never_look_ahead():
    """The point of a temporal split: every test row comes after every
    training row, or the fit has seen its own future."""
    for tr, te in cv.folds_of(_cell(scheme="walk_forward")):
        assert tr.max() < te.min()


def test_walk_forward_training_sets_grow():
    folds = cv.folds_of(_cell(scheme="walk_forward"))
    sizes = [len(tr) for tr, _ in folds]
    assert sizes == sorted(sizes)
    assert len(set(sizes)) == len(sizes)


def test_walk_forward_folds_are_contiguous():
    """A contiguous fold is what a row descriptor encodes as a range, which
    is the whole reason this scheme is cheap."""
    for tr, te in cv.folds_of(_cell(scheme="walk_forward")):
        assert np.array_equal(tr, np.arange(tr[0], tr[-1] + 1))
        assert np.array_equal(te, np.arange(te[0], te[-1] + 1))


def test_shuffled_folds_partition_the_rows():
    """Every row is tested exactly once and never trains on its own fold."""
    cell = _cell(scheme="shuffled")
    folds = cv.folds_of(cell)
    tested = np.concatenate([te for _, te in folds])
    assert sorted(tested) == list(range(cell["rows"]))
    for tr, te in folds:
        assert not set(tr) & set(te)
        assert len(tr) + len(te) == cell["rows"]


def test_shuffled_folds_are_scattered():
    """The counterpart claim: these are not ranges, which is what makes them
    the expensive case every library pays for."""
    tr, _ = cv.folds_of(_cell(scheme="shuffled"))[0]
    assert not np.array_equal(tr, np.arange(tr[0], tr[-1] + 1))


def test_the_scheme_defaults_to_walk_forward():
    assert np.array_equal(cv.folds_of(_cell())[0][0], cv.folds_of(
        _cell(scheme="walk_forward"))[0][0])


def test_shuffled_folds_are_reproducible():
    a = cv.folds_of(_cell(scheme="shuffled"))
    b = cv.folds_of(_cell(scheme="shuffled"))
    assert all(np.array_equal(x[0], y[0]) for x, y in zip(a, b))


# Dispatch ========================================================================================


def _spec(**over):
    return {runlog.Row.VARIANT: "bonsai_depthwise", runlog.Row.THREADS: 2,
            runlog.Row.REPEAT: 0, runlog.Row.CELL: _cell(**over)}


def test_a_cell_naming_folds_reaches_the_cv_suite():
    """The hook: a folds key routes to the loop question, and the single-fit
    rates that assume one fit are absent rather than wrong."""
    out = runners.worker(_spec())
    assert out[cv.FOLDS] == 4
    assert cv.LOOP_S in out
    assert "fit_rows_per_s" not in out


def test_a_cell_without_folds_still_takes_the_single_fit_path():
    cell = _cell()
    del cell["folds"]
    spec = {runlog.Row.VARIANT: "bonsai_depthwise", runlog.Row.THREADS: 2,
            runlog.Row.REPEAT: 0, runlog.Row.CELL: cell}
    out = runners.worker(spec)
    assert cv.LOOP_S not in out
    assert "fit_rows_per_s" in out


def test_the_loop_and_ingest_account_for_the_whole_run():
    out = runners.worker(_spec())
    assert out[cv.LOOP_S] + out[runlog.Row.INGEST_S] == pytest.approx(
        out[runlog.Row.FIT_S], abs=0.002)


def test_fit_and_score_account_for_the_loop():
    """The split has to conserve, or the decomposition is decoration: the
    fold mechanism's cost and the predict path's cost must add back up to
    the loop a tuning run waits on."""
    out = runners.worker(_spec())
    assert out[cv.FIT_LOOP_S] + out[cv.SCORE_LOOP_S] == pytest.approx(
        out[cv.LOOP_S], abs=0.02)
    # Scoring a thousand rows lands under this cell's rounding, so the
    # floor is non-negative rather than positive.
    assert out[cv.FIT_LOOP_S] > 0 and out[cv.SCORE_LOOP_S] >= 0


@pytest.mark.parametrize("strategy", [cv.Strategy.VIEW, cv.Strategy.COPY])
def test_both_strategies_score_the_same_way(strategy, monkeypatch):
    """A view scored through a Dataset view takes the host bin walk a
    device-resident view still falls back to. If the arms scored
    differently the comparison would be about predict, not about folds."""
    import bonsai
    seen = []
    real = bonsai.Model.predict

    def spy(self, X, *args, **kwargs):
        seen.append(type(X).__name__)
        return real(self, X, *args, **kwargs)

    monkeypatch.setattr(bonsai.Model, "predict", spy)
    runners.worker(_spec(strategy=strategy))
    assert set(seen) == {"ndarray"}


@pytest.mark.parametrize("scheme", ["walk_forward", "shuffled"])
def test_a_view_and_a_copy_fit_the_same_folds(scheme):
    """The control that makes the comparison mean anything: the view arm and
    the materialized arm must score identically, or a speed difference could
    just be one of them doing less work."""
    view = runners.worker(_spec(scheme=scheme, strategy=cv.Strategy.VIEW))
    copy = runners.worker(_spec(scheme=scheme, strategy=cv.Strategy.COPY))
    assert view[cv.R2_FOLDS] == copy[cv.R2_FOLDS]


def test_the_strategy_defaults_to_the_view():
    assert (runners.worker(_spec())[cv.R2_FOLDS]
            == runners.worker(_spec(strategy=cv.Strategy.VIEW))[cv.R2_FOLDS])


@pytest.mark.parametrize(("variant", "want"), [("lgbm_cpu", "cpu"),
                                               ("lgbm_cuda", "cuda")])
def test_the_lightgbm_arm_fits_where_its_variant_says(variant, want, monkeypatch):
    """A device constant hard-coded here would put a CPU number in a
    lgbm_cuda row, and nothing downstream reports the device a fit actually
    used, so the row would simply be wrong and look fine."""
    lgb = pytest.importorskip("lightgbm")
    seen = {}

    def fake_train(params, *args, **kwargs):
        seen.update(params)
        raise RuntimeError("params captured")

    monkeypatch.setattr(lgb, "train", fake_train)
    spec = _spec()
    spec[runlog.Row.VARIANT] = variant
    with pytest.raises(RuntimeError, match="params captured"):
        runners.worker(spec)
    assert seen["device_type"] == want


@pytest.mark.parametrize("variant", ["lgbm_cpu", "catboost_cpu", "xgb_hist"])
def test_only_bonsai_is_offered_the_copy_strategy(variant):
    """A copy cell asks "what would this cost materialized", which only
    bonsai can answer, because only bonsai had the choice. The others record
    a refusal rather than re-running their single path under a second label
    and inflating the row count with duplicates."""
    spec = _spec(strategy=cv.Strategy.COPY)
    spec[runlog.Row.VARIANT] = variant
    with pytest.raises(RuntimeError, match=r"unsupported: .* offers no 'copy'"):
        runners.worker(spec)


def test_a_quantile_dmatrix_records_its_refusal():
    """xgboost cannot slice a QuantileDMatrix. That is the finding, so it is
    a row with a reason and not an omission."""
    xgb = pytest.importorskip("xgboost")
    del xgb
    spec = _spec(strategy="quantile")
    spec[runlog.Row.VARIANT] = "xgb_hist"
    with pytest.raises(RuntimeError, match=r"unsupported: QuantileDMatrix\.slice"):
        runners.worker(spec)
