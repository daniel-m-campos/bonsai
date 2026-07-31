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
        changes.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
REGISTRY = REPO / "benchmarks" / "standings.json"
DECISIONS = REPO / "docs" / "decisions.md"

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
        elif e.get("refreshed_for") != version:
            errors.append(
                f"{axis}: refreshed for {e.get('refreshed_for')!r}, not "
                f"{version!r}; run the standings refresh before releasing")
    return errors


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
