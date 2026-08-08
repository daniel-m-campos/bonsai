"""Tests for the carry-forward stamp in scripts/update_standings.py."""

from __future__ import annotations

import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import check_standings  # noqa: E402
import update_standings  # noqa: E402

MEASURED_SHA = "148f21b"
PROOF = "9f0a1c2d3e4f5061"
REASON = "host-side fill only; src/cuda byte-identical"


def _fake_tree(root: pathlib.Path) -> None:
    """A miniature repo: one host source, one device source, one harness."""
    (root / "src" / "cuda").mkdir(parents=True)
    (root / "include" / "bonsai" / "cuda").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "src" / "tree.cpp").write_text("host\n")
    (root / "src" / "cuda" / "kernel.cu").write_text("device\n")
    (root / "include" / "bonsai" / "tree.hpp").write_text("host header\n")
    (root / "include" / "bonsai" / "cuda" / "plane.hpp").write_text("device h\n")
    (root / "scripts" / "model_hash.py").write_text("harness\n")


def _registry(root: pathlib.Path, entry: dict) -> pathlib.Path:
    """Write a one-axis registry and point update_standings at it."""
    path = root / "standings.json"
    path.write_text(json.dumps({"gpu-tall": entry}, indent=2) + "\n")
    return path


def _stale_gpu_axis(monkeypatch, tmp_path) -> pathlib.Path:
    """A gpu axis stamped before a host-side edit moved the gpu digest."""
    _fake_tree(tmp_path)
    monkeypatch.setattr(check_standings, "REPO", tmp_path)
    monkeypatch.setattr(check_standings, "MODEL_HASH_SCRIPT",
                        tmp_path / "scripts" / "model_hash.py")
    monkeypatch.setattr(update_standings, "head_sha", lambda: "f078f51")
    entry = {"file": "gpu-tall-2026-08.jsonl", "sha": MEASURED_SHA,
             "host": "pod-blackwell", "date": "2026-08-05", "plane": "gpu",
             "refreshed_for": "1.7.0", "as_of_decision": 104,
             "hash_set": check_standings.plane_digest(check_standings.PLANE_GPU),
             "refs": {"xgboost": "3.2.0"}}
    registry = _registry(tmp_path, entry)
    monkeypatch.setattr(update_standings, "REGISTRY", registry)
    (tmp_path / "src" / "tree.cpp").write_text("host, restructured\n")
    return registry


def _run(monkeypatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["update_standings.py", *argv])
    return update_standings.main()


def test_carry_forward_refuses_without_a_recorded_proof(monkeypatch, tmp_path):
    """No reason, no evidence, or evidence that is not an equality: refuse and
    leave the registry byte-identical, so a carry-forward is never a shrug."""
    registry = _stale_gpu_axis(monkeypatch, tmp_path)
    before = registry.read_text()

    assert _run(monkeypatch, "--axis", "gpu-tall", "--restamp-verified") == 1
    assert _run(monkeypatch, "--axis", "gpu-tall", "--restamp-verified",
                "--reason", REASON) == 1
    assert _run(monkeypatch, "--axis", "gpu-tall", "--restamp-verified",
                "--reason", REASON, "--evidence-before", PROOF,
                "--evidence-after", "0000000000000000") == 1
    assert registry.read_text() == before
    assert check_standings.stale_axes(json.loads(before)) == ["gpu-tall"]


def test_carry_forward_round_trip_clears_the_gates(monkeypatch, tmp_path,
                                                   capsys):
    """The stamp moves, the provenance does not, and every report that clears
    the axis says the stamp was carried rather than measured."""
    registry = _stale_gpu_axis(monkeypatch, tmp_path)
    assert _run(monkeypatch, "--axis", "gpu-tall", "--restamp-verified",
                "--reason", REASON, "--evidence-before", PROOF,
                "--evidence-after", PROOF) == 0

    reg = json.loads(registry.read_text())
    entry = reg["gpu-tall"]
    assert entry["sha"] == MEASURED_SHA
    assert entry["file"] == "gpu-tall-2026-08.jsonl"
    assert entry["date"] == "2026-08-05"
    assert entry["refreshed_for"] == "1.7.0"
    assert entry["hash_set"] == check_standings.plane_digest(
        check_standings.PLANE_GPU)
    assert entry["carried_forward"] == {
        "measured_at": MEASURED_SHA, "stamped_at": "f078f51",
        "reason": REASON,
        "evidence": {"kind": "model-hash", "before": PROOF, "after": PROOF}}

    assert check_standings.stale_axes(reg) == []
    assert check_standings.check_release(reg, "1.8.0") == []
    assert "carried-forward stamp" in capsys.readouterr().out
    notes = check_standings.carried_forward_notes(reg)
    assert len(notes) == 1 and REASON in notes[0]

    # Spent on the next move: a carry-forward is not a standing exemption.
    (tmp_path / "src" / "cuda" / "kernel.cu").write_text("device, faster\n")
    assert check_standings.stale_axes(reg) == ["gpu-tall"]
    assert check_standings.carried_forward_notes(reg) == []


def test_carry_forward_refuses_an_axis_that_is_already_current(monkeypatch,
                                                               tmp_path):
    """Nothing to carry forward is a refusal, not a no-op stamp: an entry only
    grows the block when an equivalence argument was actually needed."""
    registry = _stale_gpu_axis(monkeypatch, tmp_path)
    reg = json.loads(registry.read_text())
    reg["gpu-tall"]["hash_set"] = check_standings.plane_digest(
        check_standings.PLANE_GPU)
    registry.write_text(json.dumps(reg, indent=2) + "\n")

    assert _run(monkeypatch, "--axis", "gpu-tall", "--restamp-verified",
                "--reason", REASON, "--evidence-before", PROOF,
                "--evidence-after", PROOF) == 1
    assert "carried_forward" not in json.loads(registry.read_text())["gpu-tall"]


def test_supersession_drops_a_carried_forward_block(monkeypatch, tmp_path):
    """A fresh measurement is provenance of its own; the equivalence argument
    that carried the old one must not outlive it."""
    registry = _stale_gpu_axis(monkeypatch, tmp_path)
    assert _run(monkeypatch, "--axis", "gpu-tall", "--restamp-verified",
                "--reason", REASON, "--evidence-before", PROOF,
                "--evidence-after", PROOF) == 0

    results = tmp_path / "results"
    results.mkdir()
    (results / "gpu-tall-2026-09.jsonl").write_text(json.dumps(
        {"git_sha": "abc1234", "host": {"name": "pod-blackwell"},
         "ts": "2026-09-01T00:00:00"}) + "\n")
    monkeypatch.setattr(update_standings, "RESULTS", results)
    monkeypatch.setattr(update_standings, "refresh_ref_ledger", lambda refs: None)
    assert _run(monkeypatch, "--axis", "gpu-tall", "--file",
                "gpu-tall-2026-09.jsonl", "--version", "1.8.0") == 0

    entry = json.loads(registry.read_text())["gpu-tall"]
    assert "carried_forward" not in entry
    assert entry["sha"] == "abc1234" and entry["refreshed_for"] == "1.8.0"
