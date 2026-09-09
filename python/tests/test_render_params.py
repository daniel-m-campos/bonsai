"""scripts/render_params.py's extraction gate: the committed JSON against a
fresh `bonsai params` dump."""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import render_params  # noqa: E402

COMMITTED = {"tree": {"max_depth": 6, "min_data_in_leaf": 20},
             "boost": {"learning_rate": 0.05}}


def test_extraction_drift_names_every_moved_knob():
    """A knob added, renamed, or re-defaulted in the structs is reported by
    its dotted name and, for a default, by both values, so the author is
    told which declaration moved."""
    fresh = {"tree": {"max_depth": 8, "num_leaves": 63},
             "boost": {"learning_rate": 0.05}}
    assert render_params.extraction_drift(COMMITTED, fresh) == [
        "added tree.num_leaves", "removed tree.min_data_in_leaf",
        "tree.max_depth default 6 -> 8"]


def test_extraction_drift_is_empty_when_the_dump_matches():
    assert render_params.extraction_drift(COMMITTED, json.loads(json.dumps(COMMITTED))) == []


def test_extract_check_refuses_a_stale_file_without_writing(monkeypatch, tmp_path,
                                                            capsys):
    """The CI gate: a drifted dump exits 1 naming the knob and leaves the
    committed file untouched, the same shape as every renderer's --check."""
    pytest.importorskip("tomllib")
    src = tmp_path / "parameters.src.json"
    src.write_text(json.dumps(COMMITTED, indent=2, sort_keys=True) + "\n")
    monkeypatch.setattr(render_params, "SRC", src)
    monkeypatch.setattr(render_params, "REPO", tmp_path)
    monkeypatch.setattr(sys, "argv", ["render_params.py", "--extract", "--check"])
    monkeypatch.setattr(sys, "stdin", _Stdin(
        "[tree]\nmax_depth = 6\nmin_data_in_leaf = 20\n[boost]\nlearning_rate = 0.5\n"))

    assert render_params.extract() == 1
    assert "boost.learning_rate default 0.05 -> 0.5" in capsys.readouterr().err
    assert json.loads(src.read_text()) == COMMITTED


class _Stdin:
    def __init__(self, text: str):
        self.text = text

    def read(self) -> str:
        return self.text
