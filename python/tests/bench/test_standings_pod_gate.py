"""Tests for the throttle gate in scripts/standings_refresh_pod.sh.

The gate decides whether a pod may time the CPU plane at all, and it only
ever runs on a rented pod, so nothing here exercised it until a shipped
version computed no percentage and passed every axis anyway. These tests
run the script's own awk program rather than a copy of it, so the
precedence that broke it cannot come back unnoticed.

    pytest python/tests/bench/test_standings_pod_gate.py
"""

from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
POD_SCRIPT = REPO_ROOT / "scripts" / "standings_refresh_pod.sh"


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
