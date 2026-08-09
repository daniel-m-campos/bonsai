#!/usr/bin/env python3
"""Standings freshness gates (results-lifecycle policy, decision 92).

Two hard-fail chokepoints plus the refresh planner, stdlib only:

    python3 scripts/check_standings.py --decisions
        Claim-time gate (runs in make docs-check). A decisions-log entry that
        claims a perf change on an axis carries a `Standings: <axis>[, ...]`
        line; the gate fails while any tagged entry is newer than the axis's
        registered as_of_decision, forcing refresh-or-demote before the claim
        ships.

    python3 scripts/check_standings.py --release <version>
        Release-time gate (runs in the wheels publish job). Fails unless every
        registry entry was refreshed for exactly this release (and has a sha),
        bounding worst-case staleness at one release even for untagged
        changes, with one mechanical exception: the hash-unchanged skip below.

    python3 scripts/check_standings.py --stale
        The refresh driver's planner: one axis name per line, for the axes a
        refresh must actually measure. Everything else is provably current.

The hash-unchanged skip. An axis may pass the release gate without a fresh
refresh when its committed `hash_set` digest still matches its plane's current
digest AND none of its `refs` reference libraries has bumped a major version
since. Quality axes lean on decision 40's contract (CPU-only builds stay
bit-identical, so unmoved model-hash inputs cannot move a quality standing).
Perf axes lean on the plane split: a CUDA-only change cannot move a CPU-plane
wall clock, so the cpu axes stay current across it and the refresh skips them,
which is the whole point of registry v2's `plane` field.

Carried-forward stamps. An entry's `carried_forward` block says its `hash_set`
was moved to the current digest by an equivalence argument rather than a new
measurement (`update_standings.py --restamp-verified`). Such an entry clears
the gates like any other skip, so every report that clears it labels it
carried-forward rather than a run.

Reference-library majors are compared against the installed package in this
environment when available (`importlib.metadata`, no import needed); most
release-gate runs have no bench extras installed, so this falls back to
`benchmarks/reference_versions.json`, a hand-maintained ledger of the latest
known major per library. The bench extras are unpinned in `pyproject.toml`,
so there is no lockfile to read statically instead.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
REGISTRY = REPO / "benchmarks" / "standings.json"
DECISIONS = REPO / "docs" / "decisions.md"
REF_VERSIONS = REPO / "benchmarks" / "reference_versions.json"
MODEL_HASH_SCRIPT = REPO / "scripts" / "model_hash.py"

ENTRY_RE = re.compile(r"^## (\d+)\. ", re.M)
TAG_RE = re.compile(r"^Standings: (.+)$", re.M)

# Axes the scenario redesign retired (decision 103). Frozen decisions-log
# entries still carry `Standings:` tags naming them and the record is never
# rewritten, so the claim-time gate accepts the name and has nothing left to
# compare it against.
RETIRED_AXES = frozenset({"rows", "width", "shape", "frontier", "airline"})

PLANE_CPU = "cpu"
PLANE_GPU = "gpu"
# The device plane's sources, by repo-relative prefix: everything under them
# can only move a device timing, so a cpu-plane axis is current across a
# change confined here.
CUDA_PREFIXES = ("src/cuda/", "include/bonsai/cuda/")


def load_registry() -> dict:
    reg = json.loads(REGISTRY.read_text())
    reg.pop("_", None)
    return reg


def check_decisions(reg: dict) -> list[str]:
    text = DECISIONS.read_text()
    entries = list(ENTRY_RE.finditer(text))
    errors = []
    for i, m in enumerate(entries):
        n = int(m.group(1))
        body = text[m.end():entries[i + 1].start() if i + 1 < len(entries)
                    else len(text)]
        for tag in TAG_RE.finditer(body):
            for axis in (a.strip() for a in tag.group(1).split(",")):
                if axis in RETIRED_AXES:
                    continue
                if axis not in reg:
                    errors.append(f"decision {n}: unknown standings axis "
                                  f"{axis!r} (known: {sorted(reg)})")
                elif n > reg[axis]["as_of_decision"]:
                    errors.append(
                        f"decision {n} supersedes the {axis!r} standings "
                        f"(as_of_decision {reg[axis]['as_of_decision']}): "
                        "refresh the axis and bump the registry, or drop the "
                        "tag if the claim does not move the standings")
    return errors


def check_release(reg: dict, version: str) -> list[str]:
    errors = []
    for axis, e in reg.items():
        if not e.get("sha"):
            errors.append(f"{axis}: never refreshed (no measurement sha); "
                          "run the standings refresh before releasing")
            continue
        if e.get("refreshed_for") == version and not e.get("carried_forward"):
            print(f"{axis}: fresh refresh for {version!r}")
            continue
        ok, reason = hash_skip(axis, e)
        if ok:
            label = ("carried-forward stamp" if e.get("carried_forward")
                     else "hash-unchanged skip")
            print(f"{axis}: {label} ({reason})")
        else:
            errors.append(
                f"{axis}: refreshed for {e.get('refreshed_for')!r}, not "
                f"{version!r}, and the hash-unchanged skip does not apply "
                f"({reason}); run the standings refresh before releasing")
    return errors


def stale_axes(reg: dict) -> list[str]:
    """Axes a refresh must measure: never refreshed, or their plane moved.

    Parameters
    ----------
    reg : dict
        The loaded registry.

    Returns
    -------
    list[str]
        Registry keys in registry order.
    """
    return [axis for axis, e in reg.items()
            if not e.get("sha") or not hash_skip(axis, e)[0]]


def is_quality_axis(axis: str) -> bool:
    """Whether a registry key names a quality axis, by naming convention."""
    return axis.startswith("quality")


def plane_digest(plane: str) -> str:
    """sha256 over the sources that can move one plane's measured numbers.

    The gpu plane is the whole C++ implementation (`src/`, `include/`) plus
    the fixed-model harness `scripts/model_hash.py` it drives: that set is
    what the cross-arch CI hash jobs build and run, and any of it can move a
    device timing. The cpu plane is the same set minus the CUDA subtrees.
    `update_standings.py` computes this the same way when stamping an axis's
    `hash_set`, so the two are directly comparable.

    Parameters
    ----------
    plane : str
        `PLANE_CPU` or `PLANE_GPU`.

    Returns
    -------
    str
        A 16-character hex digest.
    """
    digest = hashlib.sha256()
    for path in _plane_paths(plane):
        digest.update(path.relative_to(REPO).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def hash_skip(axis: str, entry: dict) -> tuple[bool, str]:
    """Whether an axis may skip refresh, and why.

    Parameters
    ----------
    axis : str
        The registry key, which decides the plane when the entry names none.
    entry : dict
        The axis's registry entry; needs `hash_set` and `refs`.

    Returns
    -------
    tuple[bool, str]
        `(True, reason)` when the skip applies, `(False, reason)` otherwise;
        `reason` is always a human-readable audit line.
    """
    plane = entry.get("plane") or (PLANE_GPU if is_quality_axis(axis) else None)
    if plane is None:
        return False, "no plane registered, so nothing proves the axis current"
    current = plane_digest(plane)
    if entry.get("hash_set") != current:
        return False, f"{plane}-plane sources changed since this axis's refresh"
    bumped = bumped_ref_majors(entry.get("refs", {}))
    if bumped:
        return False, f"reference-library major bumped: {', '.join(bumped)}"
    carried = entry.get("carried_forward")
    if carried:
        kind = carried.get("evidence", {}).get("kind", "unrecorded")
        return True, (f"{plane} hash set {current} carried forward at "
                      f"{carried.get('stamped_at')} on {kind} evidence, still "
                      f"measured at {carried.get('measured_at')}: "
                      f"{carried.get('reason')}")
    return True, f"{plane} hash set {current} unchanged, refs current"


def bumped_ref_majors(refs: dict) -> list[str]:
    """Reference libraries whose current major version outruns `refs`.

    Parameters
    ----------
    refs : dict
        Library name to the version measured against, e.g.
        `{"xgboost": "3.2.0"}`.

    Returns
    -------
    list[str]
        Sorted names of libraries that have bumped a major version since.
    """
    recorded = recorded_ref_versions()
    bumped = []
    for name, measured in sorted(refs.items()):
        current = installed_ref_version(name) or recorded.get(name, measured)
        if _major(current) > _major(measured):
            bumped.append(name)
    return bumped


def installed_ref_version(name: str) -> str | None:
    """The installed version of a reference library, via package metadata.

    Reads distribution metadata without importing the package, so this stays
    cheap even when the heavy bench extras are not installed.
    """
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def recorded_ref_versions() -> dict:
    """The hand-maintained `reference_versions.json` ledger, sans its doc key."""
    versions = json.loads(REF_VERSIONS.read_text())
    versions.pop("_", None)
    return versions


def _plane_paths(plane: str) -> list[pathlib.Path]:
    """The digested files for one plane, in a stable order."""
    paths = sorted(p for p in (REPO / "src").rglob("*") if p.is_file())
    paths += sorted(p for p in (REPO / "include").rglob("*") if p.is_file())
    paths.append(MODEL_HASH_SCRIPT)
    if plane == PLANE_CPU:
        return [p for p in paths
                if not p.relative_to(REPO).as_posix().startswith(CUDA_PREFIXES)]
    return paths


def _major(version: str) -> int:
    return int(str(version).split(".")[0])


def main() -> int:
    reg = load_registry()
    if "--decisions" in sys.argv:
        errors = check_decisions(reg)
        label = "decision gate"
    elif "--release" in sys.argv:
        version = sys.argv[sys.argv.index("--release") + 1]
        errors = check_release(reg, version)
        label = f"release gate ({version})"
    elif "--stale" in sys.argv:
        # One name per line and nothing else: the refresh driver reads this.
        for axis in stale_axes(reg):
            print(axis)
        for axis, e in reg.items():
            if not e.get("carried_forward"):
                continue
            ok, reason = hash_skip(axis, e)
            if ok:
                print(f"note: carried-forward stamp, not a run: {axis}: "
                      f"{reason}", file=sys.stderr)
        return 0
    else:
        print("usage: check_standings.py --decisions | --release <version> "
              "| --stale", file=sys.stderr)
        return 2
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"standings {label}: ok ({len(reg)} axes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
