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

# The L40S session behind the tolerance kind: four reps of one cuda_depthwise
# fit at one commit wrote four different model hashes, so the argument is made
# against the run-to-run spread instead (500k x 100, 20 iters, depth 8).
DEVICE_REASON = "no src/cuda change; every gpu spec pins its thread count"
FLOOR_BEFORE = 2.4e-06
FLOOR_AFTER = 2.9e-06
CROSS_COMMIT = 3.3e-06


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


def _tolerance_argv(floor_before: str | None = str(FLOOR_BEFORE),
                    floor_after: str | None = str(FLOOR_AFTER),
                    cross_commit: str = str(CROSS_COMMIT)) -> list[str]:
    """A complete tolerance-kind command line, with the numbers swappable."""
    argv = ["--axis", "gpu-tall", "--restamp-verified",
            "--reason", DEVICE_REASON, "--evidence-kind", "tolerance",
            "--metric", "max abs prediction delta",
            "--cross-commit", cross_commit, "--cell", "500k x 100",
            "--config", "20 iters, depth 8, threads 16",
            "--evidence-host", "L40S"]
    if floor_before is not None:
        argv += ["--floor-before", floor_before]
    if floor_after is not None:
        argv += ["--floor-after", floor_after]
    return argv


def test_carry_forward_refuses_without_a_recorded_proof(monkeypatch, tmp_path):
    """The byte-identity kind with no reason, no hashes, or hashes that are
    not equal: refuse and leave the registry byte-identical, so a
    carry-forward is never a shrug."""
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


def test_tolerance_carries_a_delta_inside_the_noise_floor(monkeypatch,
                                                          tmp_path, capsys):
    """The device plane has no byte identity to offer, so the claim is that
    the cross-commit delta is the size of the plane's own run-to-run spread.
    The block records every number that claim rests on, the provenance stays
    where the last real run left it, and the reports still say
    carried-forward."""
    registry = _stale_gpu_axis(monkeypatch, tmp_path)
    assert _run(monkeypatch, *_tolerance_argv()) == 0

    entry = json.loads(registry.read_text())["gpu-tall"]
    assert entry["carried_forward"]["evidence"] == {
        "kind": "tolerance", "metric": "max abs prediction delta",
        "floor_before": FLOOR_BEFORE, "floor_after": FLOOR_AFTER,
        "cross_commit": CROSS_COMMIT,
        "factor": update_standings.TOLERANCE_FACTOR, "cell": "500k x 100",
        "config": "20 iters, depth 8, threads 16", "host": "L40S"}
    assert entry["sha"] == MEASURED_SHA
    assert entry["carried_forward"]["measured_at"] == MEASURED_SHA
    assert entry["file"] == "gpu-tall-2026-08.jsonl"
    assert entry["date"] == "2026-08-05"

    reg = json.loads(registry.read_text())
    assert check_standings.check_release(reg, "1.8.0") == []
    assert "carried-forward stamp" in capsys.readouterr().out
    notes = check_standings.carried_forward_notes(reg)
    assert len(notes) == 1 and "tolerance evidence" in notes[0]


def test_tolerance_refuses_a_delta_outside_the_noise_floor(monkeypatch,
                                                           tmp_path):
    """A delta many floors wide is a measured difference: the factor is where
    "the same up to noise" stops and "changed" begins."""
    registry = _stale_gpu_axis(monkeypatch, tmp_path)
    before = registry.read_text()
    outside = update_standings.TOLERANCE_FACTOR * FLOOR_AFTER * 10

    assert _run(monkeypatch, *_tolerance_argv(cross_commit=str(outside))) == 1
    assert registry.read_text() == before


def test_tolerance_refuses_a_missing_noise_floor(monkeypatch, tmp_path):
    """With no floor the delta has no scale to be read against, so it is a
    number rather than evidence; both commits have to have been sampled."""
    registry = _stale_gpu_axis(monkeypatch, tmp_path)
    before = registry.read_text()

    assert _run(monkeypatch, *_tolerance_argv(floor_before=None)) == 1
    assert _run(monkeypatch, *_tolerance_argv(floor_after=None)) == 1
    assert registry.read_text() == before


def test_tolerance_refuses_a_zero_noise_floor(monkeypatch, tmp_path):
    """A zero floor says the plane reproduces its bytes run to run, which is a
    stronger claim than a tolerance and belongs to the model-hash kind."""
    registry = _stale_gpu_axis(monkeypatch, tmp_path)
    before = registry.read_text()

    assert _run(monkeypatch, *_tolerance_argv(floor_before="0",
                                              floor_after="0",
                                              cross_commit="0")) == 1
    assert registry.read_text() == before
