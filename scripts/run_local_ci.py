"""Run every CI gate this host can run, and name the ones it cannot.

CI runs twelve checks on a pull request. A laptop can run nine of them; the
rest need a sanitizer runtime that works, a GPU toolchain, or a second
architecture. Reporting is the point of this script as much as running is:
skipping a gate silently is the defect this repository keeps rediscovering,
a check whose negative result never reaches the person who needs it. Every
gate this host cannot run is printed as NOT CHECKED HERE with the reason and
where it does run.

The verification floor in CLAUDE.md lists these gates individually, and a
list a human reads is exactly the thing that gets half-run under time
pressure. This is that list, executed.

    make ci              # everything, including the slow clang-tidy pass
    make ci ARGS=--fast  # skip clang-tidy, which is the slow one

Exit status is non-zero when a gate that RAN failed. A gate that could not
run here is reported, never counted as a pass.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import platform
import shutil
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent


class Reason:
    """Why a gate cannot run on this host, phrased for the summary line."""

    ASAN_MACOS: str = (
        "homebrew LLVM sanitizers deadlock on macOS; runs in CI (sanitize)"
    )
    TSAN_MACOS: str = "ThreadSanitizer needs Linux; runs in CI (race)"
    NO_CUDA: str = "no nvcc on PATH; runs in CI (cuda-compile) and on a GPU host"
    NO_TIDY: str = "clang-tidy not found under LLVM_BIN"


@dataclasses.dataclass
class Gate:
    """One CI check and the local command that reproduces it.

    Parameters
    ----------
    name
        The CI job name, so a failure here points at the job that will fail.
    command
        Argv to run from the repository root.
    skip
        None when the gate can run here, else the reason it cannot.
    slow
        True for gates `--fast` drops.
    """

    name: str
    command: list[str]
    skip: str | None = None
    slow: bool = False


def _have_cuda() -> bool:
    """True when a CUDA compiler is visible on PATH."""
    return shutil.which("nvcc") is not None


def _have_tidy() -> bool:
    """True when clang-tidy resolves through the Makefile's toolchain root."""
    if shutil.which("clang-tidy"):
        return True
    return any(pathlib.Path(p).exists() for p in (
        "/opt/homebrew/opt/llvm@21/bin/clang-tidy",
        "/usr/lib/llvm-21/bin/clang-tidy",
    ))


def gates(python: str) -> list[Gate]:
    """Build the gate list for this host.

    Parameters
    ----------
    python
        Interpreter passed through to the Makefile's PYTHON variable.

    Returns
    -------
    list[Gate]
        Every CI check with a local equivalent, in the order CI would fail
        fastest: cheap prose and format checks first, builds after.
    """
    linux = platform.system() == "Linux"
    return [
        Gate("format", ["make", "format-check"]),
        Gate("python:lint", ["make", "lint-python"]),
        Gate("python:docs", ["make", "docs-check"]),
        Gate("build-test", ["make", "test"]),
        Gate("python:pytest", ["make", "python-test", f"PYTHON={python}"]),
        Gate("tidy", ["make", "lint"], skip=None if _have_tidy() else Reason.NO_TIDY,
             slow=True),
        Gate("sanitize", ["make", "test-asan"],
             skip=None if linux else Reason.ASAN_MACOS),
        Gate("race", ["make", "test-tsan"],
             skip=None if linux else Reason.TSAN_MACOS),
        Gate("cuda-compile", ["make", "build-cuda"],
             skip=None if _have_cuda() else Reason.NO_CUDA),
    ]


def run(gate: Gate) -> tuple[str, float]:
    """Run one gate and return its outcome and wall time.

    Returns
    -------
    tuple[str, float]
        Outcome is "pass" or "FAIL"; output is streamed only on failure, so
        a clean run stays readable.
    """
    started = time.monotonic()
    proc = subprocess.run(gate.command, cwd=REPO, capture_output=True, text=True)
    elapsed = time.monotonic() - started
    if proc.returncode != 0:
        print(f"\n----- {gate.name} FAILED: {' '.join(gate.command)}")
        tail = (proc.stdout + proc.stderr).strip().splitlines()
        for line in tail[-25:]:
            print(f"  {line}")
        print()
        return "FAIL", elapsed
    return "pass", elapsed


def main() -> int:
    """Run the gates and print a summary naming everything not checked."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true",
                        help="skip the slow gates (clang-tidy)")
    parser.add_argument("--python", default=sys.executable,
                        help="interpreter for the extension build and pytest")
    args = parser.parse_args()

    results = []
    for gate in gates(args.python):
        if gate.skip is not None:
            results.append((gate.name, "not checked here", gate.skip, 0.0))
            continue
        if gate.slow and args.fast:
            results.append((gate.name, "not checked here", "--fast", 0.0))
            continue
        print(f"running {gate.name} ...", flush=True)
        outcome, elapsed = run(gate)
        results.append((gate.name, outcome, "", elapsed))

    print("\n=== local CI ===")
    failed = 0
    unchecked = 0
    for name, outcome, note, elapsed in results:
        if outcome == "FAIL":
            failed += 1
            print(f"  FAIL              {name}  ({elapsed:.0f}s)")
        elif outcome == "pass":
            print(f"  pass              {name}  ({elapsed:.0f}s)")
        else:
            unchecked += 1
            print(f"  NOT CHECKED HERE  {name}  ({note})")

    ran = len(results) - unchecked
    print(f"\n{ran - failed}/{ran} gates passed here, {unchecked} not checkable "
          f"on this host.")
    if unchecked:
        print("The unchecked ones are real gates: CI will run them.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
