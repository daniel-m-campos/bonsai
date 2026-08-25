"""Tests for bonsai.bench.synth (the byte-stability contract)."""

from __future__ import annotations

import hashlib

import pytest
from bonsai.bench import synth

# Recipe 2 (decision 112): drawn in synth.N_BLOCKS fixed row blocks off
# spawned streams. These replace the recipe-1 goldens, which were captured
# from the pre-move scripts/bench_scaling.py::gen_data; rows measured under
# either recipe are not byte-comparable with the other, which is what
# DATA_RECIPE records. The generator must stay byte-stable within a recipe
# or every perf-division result loses comparability.
GEN_DATA_GOLDENS = {
    (10_000, 20, 42, 1000, 20): ["9bdbdb370123e008", "5384529fb415ba2e",
                                 "d5cd6d231953154b", "f7729ed56e856d80"],
    (5000, 7, 0, 500, 5): ["738b8194e1340bd1", "d81c920d666b38d2",
                           "9d518cfe85657fa7", "ba65f8f194723fa6"],
}

# Bumped in the same commit as the goldens, always. A recipe change that
# leaves this alone publishes incomparable rows under one label.
GOLDEN_RECIPE = 2


def _digest(args) -> list[str]:
    return [hashlib.sha256(a.tobytes()).hexdigest()[:16]
            for a in synth.gen_data(*args)]


def test_gen_data_bytestable():
    for args, want in GEN_DATA_GOLDENS.items():
        got = _digest(args)
        assert got == want, (args, got, want)


def test_the_goldens_name_the_recipe_they_were_taken_under():
    """New bytes need a new number, or a mixed results directory reads as
    comparable when it is not."""
    assert synth.DATA_RECIPE == GOLDEN_RECIPE


@pytest.mark.parametrize("workers", [1, 2, 3, 7, 64])
def test_the_data_does_not_depend_on_how_many_threads_draw_it(monkeypatch,
                                                              workers):
    """The block count is the contract; the worker count is scheduling.

    If these ever diverge, a 27-core pod and a 128-core pod generate
    different data for the same cell and no cross-host row is comparable,
    which is strictly worse than generating serially.
    """
    monkeypatch.setattr(synth, "_workers", lambda: workers)
    args = (20_000, 16, 7, 2000, 10)
    assert _digest(args) == _digest(args)
    monkeypatch.setattr(synth, "_workers", lambda: 1)
    assert _digest(args) == _digest(args)


def test_every_block_is_drawn_even_when_rows_are_fewer_than_blocks():
    """n < N_BLOCKS leaves empty blocks; they must be a no-op, not a crash."""
    rows, n_test = 10, 3
    assert rows + n_test < synth.N_BLOCKS
    X, y, Xte, yte = synth.gen_data(rows, 4, 1, n_test, 20)
    assert X.shape == (rows, 4) and Xte.shape == (n_test, 4)
    assert y.shape == (rows,) and yte.shape == (n_test,)
    assert (X >= 0).all() and (X < 1).all()


def test_worker_count_respects_the_container_ceiling(monkeypatch):
    """A pod advertising 256 CPUs may meter 27 (issue #355); oversubscribing
    a bandwidth quota buys throttling, not throughput."""
    monkeypatch.setattr(synth.runlog, "usable_cpus", lambda: 256)
    monkeypatch.setattr(synth.runlog, "cpu_quota", lambda: 27.2)
    assert synth._workers() == 27
    monkeypatch.setattr(synth.runlog, "cpu_quota", lambda: None)
    assert synth._workers() == synth.N_BLOCKS
