"""Tests for the two stamps in scripts/update_standings.py: the carry-forward
and the supersession."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import check_standings  # noqa: E402
import update_standings  # noqa: E402

MEASURED_SHA = "148f21b"
STAMPED_SHA = "f078f51"
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
    monkeypatch.setattr(
        update_standings.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, STAMPED_SHA, ""))
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
        "measured_at": MEASURED_SHA, "stamped_at": STAMPED_SHA,
        "reason": REASON,
        "evidence": {"kind": "model-hash", "before": PROOF, "after": PROOF}}

    assert check_standings.stale_axes(reg) == []
    assert check_standings.check_release(reg, "1.8.0") == []
    assert "carried-forward stamp" in capsys.readouterr().out
    ok, audit = check_standings.hash_skip("gpu-tall", entry)
    assert ok and REASON in audit and "model-hash evidence" in audit

    # Spent on the next move: a carry-forward is not a standing exemption.
    (tmp_path / "src" / "cuda" / "kernel.cu").write_text("device, faster\n")
    assert check_standings.stale_axes(reg) == ["gpu-tall"]


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
    assert "tolerance evidence" in check_standings.hash_skip(
        "gpu-tall", reg["gpu-tall"])[1]


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


# Supersession =====================================================================================

# Every supersession test asserts the whole registry file, byte for byte:
# this is provenance machinery, so "the entry looks right" is not the
# contract. The exact text is.

INSTALLED_REFS = {"xgboost": "3.3.0"}
LEDGER_REFS = {"xgboost": "3.2.0", "lightgbm": "4.6.0", "catboost": "1.2.10"}
STAMPED_REFS = {"xgboost": "3.3.0", "lightgbm": "4.6.0", "catboost": "1.2.10"}


def _pin_environment(monkeypatch, tmp_path) -> pathlib.Path:
    """Point the ref-version ledger and the installed-version probe at fakes.

    The installed reference libraries are an environment reading, so the
    written `refs` block is only assertable once both ends are pinned.
    """
    ledger = tmp_path / "reference_versions.json"
    ledger.write_text(json.dumps({"_": "doc", **LEDGER_REFS}, indent=2) + "\n")
    monkeypatch.setattr(check_standings, "REF_VERSIONS", ledger)
    monkeypatch.setattr(check_standings, "installed_ref_version",
                        INSTALLED_REFS.get)
    return ledger


def _results(monkeypatch, tmp_path, files: dict) -> pathlib.Path:
    """A results directory holding `files` (name to body), wired into the module."""
    results = tmp_path / "results"
    results.mkdir(exist_ok=True)
    for name, body in files.items():
        (results / name).write_text(body)
    monkeypatch.setattr(update_standings, "RESULTS", results)
    return results


def _rows(*rows: dict) -> str:
    """A results jsonl body."""
    return "".join(json.dumps(r) + "\n" for r in rows)


def _row(sha: str = "abc1234", host: str = "pod-blackwell",
         ts: str = "2026-09-01T00:00:00") -> dict:
    """One results row, carrying only the fields supersession reads."""
    return {"git_sha": sha, "host": {"name": host}, "ts": ts}


def _measured_axis(monkeypatch, tmp_path, axis: str, entry: dict) -> pathlib.Path:
    """A registry holding one axis, with the fake tree behind its digests."""
    _fake_tree(tmp_path)
    monkeypatch.setattr(check_standings, "REPO", tmp_path)
    monkeypatch.setattr(check_standings, "MODEL_HASH_SCRIPT",
                        tmp_path / "scripts" / "model_hash.py")
    _pin_environment(monkeypatch, tmp_path)
    registry = tmp_path / "standings.json"
    registry.write_text(json.dumps({axis: entry}, indent=2) + "\n")
    monkeypatch.setattr(update_standings, "REGISTRY", registry)
    return registry


def _expect(registry: pathlib.Path, axis: str, entry: dict) -> None:
    """Assert the registry file is exactly this axis, byte for byte."""
    assert registry.read_text() == json.dumps({axis: entry}, indent=2) + "\n"


def test_supersede_refuses_a_file_that_is_not_single_sha(monkeypatch, tmp_path):
    """A standings claim names one commit. Rows from two commits, or rows
    that name none at all, cannot back one, and the registry must not move."""
    registry = _measured_axis(monkeypatch, tmp_path, "gpu-tall",
                              {"file": "gpu-tall-2026-08.jsonl",
                               "sha": MEASURED_SHA, "plane": "gpu",
                               "refreshed_for": "1.7.0"})
    before = registry.read_text()
    _results(monkeypatch, tmp_path, {
        "two.jsonl": _rows(_row(sha="abc1234"), _row(sha="def5678")),
        "none.jsonl": _rows({"host": {"name": "pod"}, "ts": "2026-09-01"})})

    assert _run(monkeypatch, "--axis", "gpu-tall", "--file", "two.jsonl") == 1
    assert _run(monkeypatch, "--axis", "gpu-tall", "--file", "none.jsonl") == 1
    assert registry.read_text() == before


def test_supersede_refuses_rows_that_name_no_commit(monkeypatch, tmp_path):
    """"unknown" is what runlog writes when nothing states the commit. It is
    truthy, so it clears the single-sha check while leaving the rows
    unattributable; the registry would claim provenance it does not have."""
    registry = _measured_axis(monkeypatch, tmp_path, "gpu-tall",
                              {"file": "gpu-tall-2026-08.jsonl",
                               "sha": MEASURED_SHA, "plane": "gpu",
                               "refreshed_for": "1.7.0"})
    before = registry.read_text()
    _results(monkeypatch, tmp_path,
             {"new.jsonl": _rows(_row(sha="unknown"), _row(sha="unknown"))})

    assert _run(monkeypatch, "--axis", "gpu-tall", "--file", "new.jsonl") == 1
    assert registry.read_text() == before


def test_supersede_deletes_the_file_it_replaces_and_stamps_the_gate(
        monkeypatch, tmp_path, capsys):
    """The whole write on the ordinary path: the superseded file is deleted
    (git history is the archive), the note that described it goes with it,
    the newest row's date wins, and a planed axis regains its skip gate from
    the current sources and the installed reference libraries."""
    registry = _measured_axis(monkeypatch, tmp_path, "gpu-tall",
                              {"file": "gpu-tall-2026-08.jsonl",
                               "sha": MEASURED_SHA, "host": "old-pod",
                               "date": "2026-08-05", "plane": "gpu",
                               "refreshed_for": "1.7.0",
                               "note": "measured under a quota",
                               "hash_set": "stale",
                               "refs": {"xgboost": "3.1.0"}})
    results = _results(monkeypatch, tmp_path, {
        "gpu-tall-2026-08.jsonl": "old\n",
        "gpu-tall-2026-09.jsonl": _rows(_row(ts="2026-09-01T00:00:00"),
                                        _row(ts="2026-09-02T00:00:00"))})

    assert _run(monkeypatch, "--axis", "gpu-tall", "--file",
                "gpu-tall-2026-09.jsonl", "--version", "2.1.0") == 0

    _expect(registry, "gpu-tall", {
        "file": "gpu-tall-2026-09.jsonl", "sha": "abc1234",
        "host": "pod-blackwell", "date": "2026-09-02", "plane": "gpu",
        "refreshed_for": "2.1.0",
        "hash_set": check_standings.plane_digest(check_standings.PLANE_GPU),
        "refs": STAMPED_REFS})
    assert not (results / "gpu-tall-2026-08.jsonl").exists()
    assert json.loads((tmp_path / "reference_versions.json").read_text()) == {
        "_": "doc", **STAMPED_REFS}
    out = capsys.readouterr().out
    assert "superseded gpu-tall-2026-08.jsonl" in out
    assert "gpu-tall: gpu-tall-2026-09.jsonl at abc1234" in out


def test_supersede_keeps_the_note_when_the_file_name_does_not_move(monkeypatch,
                                                                   tmp_path):
    """A re-run onto the same dated file supersedes nothing, so there is no
    file to delete and no note to drop: only the measurement is rewritten."""
    registry = _measured_axis(monkeypatch, tmp_path, "gpu-tall",
                              {"file": "gpu-tall-2026-09.jsonl",
                               "sha": MEASURED_SHA, "host": "old-pod",
                               "date": "2026-08-05", "plane": "gpu",
                               "refreshed_for": "1.7.0",
                               "note": "measured under a quota",
                               "hash_set": "stale",
                               "refs": {"xgboost": "3.1.0"}})
    results = _results(monkeypatch, tmp_path,
                       {"gpu-tall-2026-09.jsonl": _rows(_row())})

    assert _run(monkeypatch, "--axis", "gpu-tall", "--file",
                "gpu-tall-2026-09.jsonl", "--version", "2.1.0") == 0

    _expect(registry, "gpu-tall", {
        "file": "gpu-tall-2026-09.jsonl", "sha": "abc1234",
        "host": "pod-blackwell", "date": "2026-09-01", "plane": "gpu",
        "refreshed_for": "2.1.0", "note": "measured under a quota",
        "hash_set": check_standings.plane_digest(check_standings.PLANE_GPU),
        "refs": STAMPED_REFS})
    assert (results / "gpu-tall-2026-09.jsonl").exists()


def test_supersede_swaps_the_companion_it_was_given(monkeypatch, tmp_path,
                                                    capsys):
    """The companion is measured with the axis and supersedes with it, so a
    new one deletes the old one and the registry records the new name."""
    registry = _measured_axis(monkeypatch, tmp_path, "gpu-tall",
                              {"file": "gpu-tall-2026-08.jsonl",
                               "sha": MEASURED_SHA, "plane": "gpu",
                               "refreshed_for": "1.7.0",
                               "companion": "parity-2026-08.jsonl"})
    results = _results(monkeypatch, tmp_path, {
        "parity-2026-08.jsonl": "old parity\n",
        "gpu-tall-2026-09.jsonl": _rows(_row())})

    assert _run(monkeypatch, "--axis", "gpu-tall", "--file",
                "gpu-tall-2026-09.jsonl", "--companion",
                "parity-2026-09.jsonl", "--version", "2.1.0") == 0

    _expect(registry, "gpu-tall", {
        "file": "gpu-tall-2026-09.jsonl", "sha": "abc1234", "plane": "gpu",
        "refreshed_for": "2.1.0", "companion": "parity-2026-09.jsonl",
        "host": "pod-blackwell", "date": "2026-09-01",
        "hash_set": check_standings.plane_digest(check_standings.PLANE_GPU),
        "refs": STAMPED_REFS})
    assert not (results / "parity-2026-08.jsonl").exists()
    assert "superseded parity-2026-08.jsonl" in capsys.readouterr().out


def test_supersede_leaves_a_repeated_companion_alone(monkeypatch, tmp_path):
    """Naming the companion the entry already carries deletes nothing: the
    file that would be removed is the one about to be claimed."""
    registry = _measured_axis(monkeypatch, tmp_path, "gpu-tall",
                              {"file": "gpu-tall-2026-08.jsonl",
                               "sha": MEASURED_SHA, "plane": "gpu",
                               "refreshed_for": "1.7.0",
                               "companion": "parity-2026-09.jsonl"})
    results = _results(monkeypatch, tmp_path, {
        "parity-2026-09.jsonl": "parity\n",
        "gpu-tall-2026-09.jsonl": _rows(_row())})

    assert _run(monkeypatch, "--axis", "gpu-tall", "--file",
                "gpu-tall-2026-09.jsonl", "--companion",
                "parity-2026-09.jsonl", "--version", "2.1.0") == 0

    assert (results / "parity-2026-09.jsonl").exists()
    assert json.loads(registry.read_text())["gpu-tall"]["companion"] == (
        "parity-2026-09.jsonl")


def test_supersede_records_no_host_when_the_rows_disagree(monkeypatch,
                                                          tmp_path):
    """Rows from two machines cannot name one host of record, so the field
    goes null rather than picking a winner. Rows with no timestamp leave the
    date null, an unnamed version falls back to the project's own, and an
    axis registering no plane keeps its skip gate unstamped."""
    registry = _measured_axis(monkeypatch, tmp_path, "gpu-tall",
                              {"file": None, "sha": None, "plane": None,
                               "refreshed_for": None})
    _results(monkeypatch, tmp_path, {"gpu-tall-2026-09.jsonl": _rows(
        {"git_sha": "abc1234", "host": {"name": "pod-a"}},
        {"git_sha": "abc1234", "host": "pod-b"})})

    assert _run(monkeypatch, "--axis", "gpu-tall", "--file",
                "gpu-tall-2026-09.jsonl") == 0

    _expect(registry, "gpu-tall", {
        "file": "gpu-tall-2026-09.jsonl", "sha": "abc1234", "plane": None,
        "refreshed_for": update_standings.project_version(),
        "host": None, "date": None})


def test_supersede_stamps_a_quality_axis_that_registers_no_plane(monkeypatch,
                                                                 tmp_path):
    """A quality axis carries no plane and is gated on the whole-implementation
    digest, which is the gpu plane's by construction."""
    registry = _measured_axis(monkeypatch, tmp_path, "quality-grinsztajn",
                              {"file": None, "sha": None,
                               "refreshed_for": None})
    _results(monkeypatch, tmp_path,
             {"quality-grinsztajn-2026-09.jsonl": _rows(_row())})

    assert _run(monkeypatch, "--axis", "quality-grinsztajn", "--file",
                "quality-grinsztajn-2026-09.jsonl", "--version", "2.1.0") == 0

    _expect(registry, "quality-grinsztajn", {
        "file": "quality-grinsztajn-2026-09.jsonl", "sha": "abc1234",
        "refreshed_for": "2.1.0", "host": "pod-blackwell",
        "date": "2026-09-01",
        "hash_set": check_standings.plane_digest(check_standings.PLANE_GPU),
        "refs": STAMPED_REFS})


def test_supersede_swaps_the_ab_file_it_was_given(monkeypatch, tmp_path,
                                                  capsys):
    """The release A/B is measured with the axis and supersedes with it,
    exactly as the companion does."""
    registry = _measured_axis(monkeypatch, tmp_path, "cpu-tall",
                              {"file": "cpu-tall-2026-08.jsonl",
                               "sha": MEASURED_SHA, "plane": "cpu",
                               "refreshed_for": "1.7.0",
                               "ab": "ab-cpu-2026-08.jsonl"})
    results = _results(monkeypatch, tmp_path, {
        "ab-cpu-2026-08.jsonl": "old ab\n",
        "cpu-tall-2026-09.jsonl": _rows(_row())})

    assert _run(monkeypatch, "--axis", "cpu-tall", "--file",
                "cpu-tall-2026-09.jsonl", "--ab", "ab-cpu-2026-09.jsonl",
                "--version", "2.1.0") == 0

    assert json.loads(registry.read_text())["cpu-tall"]["ab"] == (
        "ab-cpu-2026-09.jsonl")
    assert not (results / "ab-cpu-2026-08.jsonl").exists()
    assert "superseded ab-cpu-2026-08.jsonl" in capsys.readouterr().out


def test_supersede_keeps_the_ab_file_when_none_is_given(monkeypatch, tmp_path):
    """A refresh that fitted no arms leaves the last A/B registered: the
    gate keeps reading it until a later session replaces it."""
    registry = _measured_axis(monkeypatch, tmp_path, "cpu-tall",
                              {"file": "cpu-tall-2026-08.jsonl",
                               "sha": MEASURED_SHA, "plane": "cpu",
                               "refreshed_for": "1.7.0",
                               "ab": "ab-cpu-2026-08.jsonl"})
    results = _results(monkeypatch, tmp_path, {
        "ab-cpu-2026-08.jsonl": "old ab\n",
        "cpu-tall-2026-09.jsonl": _rows(_row())})

    assert _run(monkeypatch, "--axis", "cpu-tall", "--file",
                "cpu-tall-2026-09.jsonl", "--version", "2.1.0") == 0

    assert json.loads(registry.read_text())["cpu-tall"]["ab"] == (
        "ab-cpu-2026-08.jsonl")
    assert (results / "ab-cpu-2026-08.jsonl").exists()
