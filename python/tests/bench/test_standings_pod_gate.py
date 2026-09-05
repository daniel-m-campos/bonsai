"""Tests for the gates in scripts/standings_refresh_pod.sh.

These decide whether a pod may publish what it measured: which commit it
measured it at, whether the CPU plane was timed under a ceiling, and whether
this session is the one that owns the parity rows. All of them run only on a
rented pod, which is why two shipped versions were wrong for months, so every
test here lifts the script's own code out and runs it rather than restating
what it should do.

    pytest python/tests/bench/test_standings_pod_gate.py
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
POD_SCRIPT = REPO_ROOT / "scripts" / "standings_refresh_pod.sh"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import standings_refresh  # noqa: E402


def _throttle_pct_program() -> str:
    """The awk program the pod script uses to compute the throttled share.

    Returns
    -------
    str
        The single-quoted awk source, lifted from the script so the test
        cannot drift away from what the pod actually runs.

    Raises
    ------
    AssertionError
        If the script no longer contains exactly one such program.
    """
    text = POD_SCRIPT.read_text()
    found = re.findall(r"'(BEGIN \{printf \"%\.1f\"[^']*)'", text)
    assert len(found) == 1, f"expected one throttle-pct program, got {found}"
    return found[0]


def _run_pct(periods: int, throttled: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["awk", "-v", f"p={periods}", "-v", f"t={throttled}",
         _throttle_pct_program()],
        capture_output=True, text=True, cwd=REPO_ROOT)


@pytest.mark.parametrize("periods,throttled,expected",
                         [(169, 20, "11.8"), (558, 0, "0.0"),
                          (100, 100, "100.0"), (0, 0, "0.0")])
def test_throttle_pct_reaches_stdout(periods, throttled, expected):
    """A percentage the gate can read, not a file the gate never opens.

    `printf fmt, e > x` is redirection in awk, so an unparenthesized
    ternary here wrote the period count to a file named 0 and printed
    nothing; the gate then compared an empty string and passed.
    """
    done = _run_pct(periods, throttled)
    assert done.stdout == expected, done.stderr
    assert not (REPO_ROOT / "0").exists(), "awk redirected instead of printing"


def test_the_gate_fails_closed_when_the_probe_reads_nothing():
    """An unreadable probe must abort the axis, because nothing is not a pass."""
    text = POD_SCRIPT.read_text()
    guard = re.search(r"case \"\$pct\" in\s*\n\s*''\|\*\[!0-9\.\]\*\)\s*\n\s*"
                      r"fail_axis", text)
    assert guard, ("the pct guard is gone: an empty or non-numeric percentage "
                   "compares as a string against the limit and passes")


def test_the_limit_is_compared_as_a_number():
    """String comparison would rank "11.8" below "5" and pass a throttled pod."""
    text = POD_SCRIPT.read_text()
    assert "pct + 0 > max + 0" in text
    over = subprocess.run(["awk", "-v", "pct=11.8", "-v", "max=5",
                           "BEGIN {exit !(pct + 0 > max + 0)}"])
    under = subprocess.run(["awk", "-v", "pct=0.9", "-v", "max=5",
                            "BEGIN {exit !(pct + 0 > max + 0)}"])
    assert over.returncode == 0 and under.returncode == 1


# Provenance =======================================================================================

def _checkout_sha_source() -> str:
    """The pod script's checkout_sha function, lifted whole."""
    found = re.search(r"^checkout_sha\(\) \{.*?^\}$", POD_SCRIPT.read_text(),
                      re.S | re.M)
    assert found, f"checkout_sha() is gone from {POD_SCRIPT.name}"
    return found.group(0)


def _git(repo: pathlib.Path, *args: str) -> str:
    done = subprocess.run(["git", *args], cwd=repo, capture_output=True,
                          text=True, check=True)
    return done.stdout.strip()


def _fake_origin(root: pathlib.Path) -> dict:
    """An origin with two branches, and a single-branch clone of one of them.

    The clone is single-branch because that is what makes the second branch
    genuinely absent locally, which is the case the explicit refspec exists
    for and the case the runbook records hitting three times. It goes over
    file:// rather than a plain path: a local-path clone hardlinks the whole
    object store, so every commit would already be here and nothing would be
    missing to fetch.
    """
    origin = root / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main", ".")
    _git(origin, "config", "user.email", "pod@test")
    _git(origin, "config", "user.name", "pod")
    (origin / "a").write_text("a\n")
    _git(origin, "add", "a")
    _git(origin, "commit", "-qm", "first")
    _git(origin, "checkout", "-qb", "side")
    (origin / "b").write_text("b\n")
    _git(origin, "add", "b")
    _git(origin, "commit", "-qm", "second")
    side = _git(origin, "rev-parse", "HEAD")
    _git(origin, "checkout", "-q", "main")
    work = root / "work"
    subprocess.run(["git", "clone", "-q", "--single-branch", "--branch",
                    "main", origin.as_uri(), str(work)], check=True)
    return {"work": work, "side": side, "clone_head": _git(work, "rev-parse",
                                                           "HEAD")}


def _run_checkout(work: pathlib.Path, sha: str) -> subprocess.CompletedProcess:
    body = _checkout_sha_source()
    return subprocess.run(
        ["bash", "-c", f"set -euo pipefail\n{body}\ncheckout_sha \"$1\"",
         "_", sha],
        cwd=work, capture_output=True, text=True)


def test_the_checkout_reaches_a_commit_the_clone_never_fetched(tmp_path):
    """A single-branch clone has no ref for the requested commit; the explicit
    refspec is what puts it within reach, and HEAD must land exactly there."""
    repo = _fake_origin(tmp_path)
    done = _run_checkout(repo["work"], repo["side"])
    assert done.returncode == 0, done.stderr
    assert _git(repo["work"], "rev-parse", "HEAD") == repo["side"]


def test_an_unreachable_commit_fails_instead_of_measuring_the_clone(tmp_path):
    """The 1.14.0 sweep asked for one commit, measured another, and reported
    failures=0. The old form is run here to show it still would."""
    repo = _fake_origin(tmp_path)
    bogus = "0" * 39 + "1"
    old = subprocess.run(
        ["bash", "-c", "set -euo pipefail\n"
                       'git fetch origin "$1" && git checkout -f "$1"\n'
                       "git rev-parse HEAD\n", "_", bogus],
        cwd=repo["work"], capture_output=True, text=True)
    assert old.returncode == 0, "set -e exempts a non-final && list, as shipped"
    assert old.stdout.strip() == repo["clone_head"], (
        "the old form leaves HEAD wherever the clone landed")

    done = _run_checkout(repo["work"], bogus)
    assert done.returncode != 0, "an unreachable commit must abort the sweep"
    assert _git(repo["work"], "rev-parse", "HEAD") == repo["clone_head"]


def test_a_name_that_is_a_path_and_not_a_commit_fails(tmp_path):
    """`git checkout -f <tracked file>` exits 0 and leaves HEAD where it was,
    so the exit status alone cannot say a commit was reached. This is the case
    the HEAD assertion is for."""
    repo = _fake_origin(tmp_path)
    done = _run_checkout(repo["work"], "a")
    assert done.returncode != 0, "a path checkout is not a commit checkout"
    assert _git(repo["work"], "rev-parse", "HEAD") == repo["clone_head"]


# Parity ownership =================================================================================

def _parity_guard() -> str:
    """The condition guarding the pod script's parity block."""
    found = [ln for ln in POD_SCRIPT.read_text().splitlines()
             if ln.startswith("if ") and "PARITY_AXIS" in ln]
    assert len(found) == 1, f"expected one parity guard, got {found}"
    return found[0].removeprefix("if ").removesuffix("; then")


@pytest.mark.parametrize("plane,axes,taken", [
    ("gpu", "gpu-tall,gpu-wide", True),
    ("gpu", "gpu-tall", True),
    ("gpu", "gpu-early-stop,gpu-tall", True),
    ("gpu", "gpu-extreme", False),
    ("gpu", "gpu-tallest", False),
    ("cpu", "cpu-tall", False),
])
def test_parity_is_taken_only_by_the_session_that_anchors_it(plane, axes, taken):
    """Four runs at the anchor cell, and a file committed as that axis's
    companion, so a session measuring anything else must not produce them: its
    rows would be another host's, landing on the anchor's for the same month.
    The gpu-tallest case is why the match is comma-anchored."""
    done = subprocess.run(
        ["bash", "-c", f"PLANE={plane}; AXES={axes}; PARITY_AXIS=gpu-tall\n"
                       f"if {_parity_guard()}; then echo taken; else echo no; fi"],
        capture_output=True, text=True)
    assert done.stdout.strip() == ("taken" if taken else "no"), done.stderr


# A/B arms =========================================================================================

def _arm_list_source() -> str:
    """The lines that build AB_ARMS from which wheels the driver named."""
    lines = POD_SCRIPT.read_text().splitlines()
    start = lines.index("AB_ARMS=()")
    end = lines.index("AB_ARMS+=(new)", start)
    return "\n".join(lines[start:end + 1])


@pytest.mark.parametrize("anchor,prev,arms", [
    ("1.15.0", "2.0.0", "anchor old new"),
    ("1.15.0", "", "anchor new"),
    ("", "2.0.0", "old new"),
    ("", "", "new"),
])
def test_the_arms_are_the_wheels_the_driver_named_plus_head(anchor, prev, arms):
    """The anchor and the previous release are each optional; HEAD is always
    fitted, and with no wheel to compare against the A/B blocks (one arm)
    have nothing to do."""
    done = subprocess.run(
        ["bash", "-c", f"set -eu; ANCHOR_VERSION='{anchor}'; PREV_VERSION='{prev}'\n"
                       f"{_arm_list_source()}\n"
                       'echo "${AB_ARMS[*]}"'],
        capture_output=True, text=True)
    assert done.stdout.strip() == arms, done.stderr


def _cpu_ab_guard() -> str:
    """The condition under which run_cpu_axis takes the cpu plane's A/B."""
    found = [ln.strip() for ln in POD_SCRIPT.read_text().splitlines()
             if ln.strip().startswith("if ") and "CPU_AB_AXIS" in ln]
    assert len(found) == 1, f"expected one cpu A/B guard, got {found}"
    return found[0].removeprefix("if ").removesuffix("; then")


@pytest.mark.parametrize("axis,taken", [
    ("cpu-tall", True),
    ("cpu-wide", False),
    ("cpu-tallest", False),
])
def test_the_cpu_ab_rides_the_tall_axis_only(axis, taken):
    """One A/B file per plane, at the tall cell, after that axis cleared the
    cap gate: a session measuring only cpu-wide never writes it, so its rows
    cannot be another cell's under the same name."""
    done = subprocess.run(
        ["bash", "-c", f"axis={axis}; CPU_AB_AXIS=cpu-tall\n"
                       f"if {_cpu_ab_guard()}; then echo taken; else echo no; fi"],
        capture_output=True, text=True)
    assert done.stdout.strip() == ("taken" if taken else "no"), done.stderr


def test_the_driver_registers_the_cpu_ab_under_the_axis_the_pod_fits_it_at():
    """The pod writes ab-cpu.jsonl at CPU_AB_AXIS's cell; the driver dates and
    registers it under AB_AXES[cpu]. Two constants, one cell."""
    match = re.search(r"^CPU_AB_AXIS=(\S+)$", POD_SCRIPT.read_text(), re.M)
    assert match, "the pod script names no CPU_AB_AXIS"
    assert standings_refresh.AB_AXES[standings_refresh.PLANE_CPU] == match.group(1)
    assert standings_refresh.AB_AXES[standings_refresh.PLANE_GPU] == (
        standings_refresh.PARITY_AXIS)
