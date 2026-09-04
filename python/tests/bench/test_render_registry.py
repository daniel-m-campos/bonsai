"""Tests for the standings registry checks in scripts/render_results.py."""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import render_results  # noqa: E402

SHA = "0123456789abcdef"


def _registry(tmp_path, monkeypatch, entries, files):
    """Point the renderer at a throwaway repo holding ``entries`` as the
    standings registry and ``files`` (name -> rows) under results/."""
    results = tmp_path / "benchmarks" / "results"
    results.mkdir(parents=True)
    (tmp_path / "benchmarks" / "standings.json").write_text(
        json.dumps({"_": {"note": "ignored"}, **entries})
    )
    for name, rows in files.items():
        (results / name).write_text(
            "".join(json.dumps(r) + "\n" for r in rows) + "\n"
        )
    monkeypatch.setattr(render_results, "REPO", tmp_path)
    monkeypatch.setattr(render_results, "RESULTS", results)


def test_an_unmeasured_axis_has_nothing_to_check(tmp_path, monkeypatch):
    _registry(tmp_path, monkeypatch, {"rows": {"file": None, "sha": SHA}}, {})
    assert render_results.check_registry(set()) == []


def test_a_registered_file_must_exist_and_be_rendered(tmp_path, monkeypatch):
    entries = {
        "rows": {"file": "rows.jsonl", "sha": SHA},
        "cols": {"file": "cols.jsonl", "sha": SHA},
    }
    _registry(tmp_path, monkeypatch, entries, {"cols.jsonl": []})
    assert render_results.check_registry(set()) == [
        "standings rows: rows.jsonl does not exist",
        "standings cols: cols.jsonl is not rendered",
    ]
    assert render_results.check_registry({"cols.jsonl"}) == [
        "standings rows: rows.jsonl does not exist",
    ]
    (tmp_path / "benchmarks" / "results" / "rows.jsonl").write_text("\n")
    assert render_results.check_registry({"cols.jsonl", "rows.jsonl"}) == []


def test_a_companion_is_held_to_the_same_rule(tmp_path, monkeypatch):
    entries = {"rows": {"file": "rows.jsonl", "companion": "rows_evidence.md"}}
    _registry(tmp_path, monkeypatch, entries, {"rows.jsonl": []})
    assert render_results.check_registry({"rows.jsonl"}) == [
        "standings rows: companion rows_evidence.md does not exist",
    ]
    (tmp_path / "benchmarks" / "results" / "rows_evidence.md").write_text("x\n")
    assert render_results.check_registry({"rows.jsonl"}) == [
        "standings rows: companion rows_evidence.md is not rendered",
    ]
    assert render_results.check_registry({"rows.jsonl", "rows_evidence.md"}) == []


def test_rows_carry_the_registered_sha_or_a_prefix_of_it(tmp_path, monkeypatch):
    entries = {"rows": {"file": "rows.jsonl", "sha": SHA}}
    rows = [{"git_sha": SHA}, {"git_sha": SHA[:7]}, {"git_sha": SHA + "ff"}]
    _registry(tmp_path, monkeypatch, entries, {"rows.jsonl": rows})
    assert render_results.check_registry({"rows.jsonl"}) == []


def test_a_wrong_sha_is_named_with_the_registered_one(tmp_path, monkeypatch):
    entries = {"rows": {"file": "rows.jsonl", "sha": SHA}}
    rows = [{"git_sha": SHA}, {"git_sha": "feedface"}, {"git_sha": "cafebabe"}]
    _registry(tmp_path, monkeypatch, entries, {"rows.jsonl": rows})
    assert render_results.check_registry({"rows.jsonl"}) == [
        f"standings rows: rows carry ['cafebabe', 'feedface'], registry says {SHA}",
    ]


@pytest.mark.parametrize("partial", [False, True])
def test_a_row_without_a_sha_needs_sha_partial(tmp_path, monkeypatch, partial):
    entries = {"rows": {"file": "rows.jsonl", "sha": SHA, "sha_partial": partial}}
    rows = [{"git_sha": SHA}, {"variant": "no provenance"}]
    _registry(tmp_path, monkeypatch, entries, {"rows.jsonl": rows})
    expected = [] if partial else [
        "standings rows: rows without git_sha in a full-provenance standings file",
    ]
    assert render_results.check_registry({"rows.jsonl"}) == expected


def test_an_entry_without_a_sha_skips_the_row_check(tmp_path, monkeypatch):
    entries = {"rows": {"file": "rows.jsonl"}}
    rows = [{"git_sha": "feedface"}, {"variant": "no provenance"}]
    _registry(tmp_path, monkeypatch, entries, {"rows.jsonl": rows})
    assert render_results.check_registry({"rows.jsonl"}) == []
