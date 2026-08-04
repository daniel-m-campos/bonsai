#!/usr/bin/env python3
"""Standings freshness gates (results-lifecycle policy, decision 92).

Two hard-fail chokepoints, stdlib only:

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
        changes. Quality axes (registry key starts with "quality") get one
        mechanical exception: the hash-unchanged skip, below.

The hash-unchanged skip. A quality axis may pass the release gate without a
fresh refresh when its committed `hash_set` digest still matches
`hash_set_digest()` AND none of its `refs` reference libraries has bumped a
major version since. Bit-identical CPU model-hash bytes cannot move quality
standings by construction (decision 40's contract: CPU-only builds stay
bit-identical), so an unmoved guard proves the axis current for free. Perf
axes carry no such proof (their numbers are wall-clock, not bytes) and are
unaffected: they still require `refreshed_for == version` exactly.

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
            errors.append(f"{axis}: standings have no measurement sha; "
                          "run the standings refresh before releasing")
        elif e.get("refreshed_for") == version:
            print(f"{axis}: fresh refresh for {version!r}")
        elif is_quality_axis(axis):
            ok, reason = quality_skip(e)
            if ok:
                print(f"{axis}: hash-unchanged skip ({reason})")
            else:
                errors.append(
                    f"{axis}: refreshed for {e.get('refreshed_for')!r}, not "
                    f"{version!r}, and the hash-unchanged skip does not "
                    f"apply ({reason}); run the standings refresh before "
                    "releasing")
        else:
            errors.append(
                f"{axis}: refreshed for {e.get('refreshed_for')!r}, not "
                f"{version!r}; run the standings refresh before releasing")
    return errors


def is_quality_axis(axis: str) -> bool:
    """Whether a registry key names a quality axis, by naming convention."""
    return axis.startswith("quality")


def hash_set_digest() -> str:
    """sha256 over every file that feeds the model-hash guard.

    Covers the fixed-model harness (`scripts/model_hash.py`) and the whole
    C++ implementation it drives (`src/`, `include/`): together these are
    what the cross-arch CI hash jobs build and run. `update_standings.py`
    computes this the same way when stamping a quality axis's `hash_set`, so
    the two are directly comparable.

    Returns
    -------
    str
        A 16-character hex digest.
    """
    paths = sorted(p for p in (REPO / "src").rglob("*") if p.is_file())
    paths += sorted(p for p in (REPO / "include").rglob("*") if p.is_file())
    paths.append(MODEL_HASH_SCRIPT)
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(REPO).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def quality_skip(entry: dict) -> tuple[bool, str]:
    """Whether a quality axis may skip refresh, and why.

    Parameters
    ----------
    entry : dict
        The axis's registry entry; needs `hash_set` and `refs`.

    Returns
    -------
    tuple[bool, str]
        `(True, reason)` when the skip applies, `(False, reason)` otherwise;
        `reason` is always a human-readable audit line.
    """
    current = hash_set_digest()
    if entry.get("hash_set") != current:
        return False, "model-hash guard inputs changed since this axis's refresh"
    bumped = bumped_ref_majors(entry.get("refs", {}))
    if bumped:
        return False, f"reference-library major bumped: {', '.join(bumped)}"
    return True, f"hash set {current} unchanged, refs current"


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
    else:
        print("usage: check_standings.py --decisions | --release <version>",
              file=sys.stderr)
        return 2
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"standings {label}: ok ({len(reg)} axes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
