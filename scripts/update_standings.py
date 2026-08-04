#!/usr/bin/env python3
"""Point a standings axis at a freshly measured file (decision 92).

    python3 scripts/update_standings.py --axis rows --file rebaseline-2026-08.jsonl

Reads sha/host/date from the new file's rows, deletes the superseded file
(git history is the archive), and rewrites the registry entry with
refreshed_for taken from pyproject.toml (override with --version).
Supersession also drops the entry's note, since a note describes the
file being replaced. Run render_results.py afterwards; the generator
loads through the registry.

Superseding a quality axis (registry key starts with "quality") also stamps
`hash_set` (the model-hash guard digest, `check_standings.hash_set_digest()`)
and `refs` (the reference libraries' installed versions) so a later release
can use check_standings.py's hash-unchanged skip instead of a full refresh.
The refresh is the one moment the installed reference-library versions are
known for certain (the grinsztajn suite runs against them), so it also
refreshes benchmarks/reference_versions.json to match.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
REGISTRY = REPO / "benchmarks" / "standings.json"
RESULTS = REPO / "benchmarks" / "results"

sys.path.insert(0, str(REPO / "scripts"))
import check_standings  # noqa: E402

REF_LIBRARIES = ("xgboost", "lightgbm", "catboost")


def project_version() -> str:
    m = re.search(r'^version = "([^"]+)"',
                  (REPO / "pyproject.toml").read_text(), re.M)
    return m.group(1)


def stamp_quality_gate(entry: dict) -> None:
    """Record the hash-unchanged skip's inputs on a freshly refreshed axis."""
    entry["hash_set"] = check_standings.hash_set_digest()
    ref_versions = current_ref_versions()
    entry["refs"] = ref_versions
    refresh_ref_ledger(ref_versions)


def current_ref_versions() -> dict:
    """Installed reference-library versions, falling back to the ledger."""
    recorded = check_standings.recorded_ref_versions()
    return {name: check_standings.installed_ref_version(name) or recorded.get(name)
            for name in REF_LIBRARIES}


def refresh_ref_ledger(ref_versions: dict) -> None:
    """Fold newly seen reference-library versions into the recorded ledger."""
    ledger = json.loads(check_standings.REF_VERSIONS.read_text())
    ledger.update({name: v for name, v in ref_versions.items() if v})
    check_standings.REF_VERSIONS.write_text(json.dumps(ledger, indent=2) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True)
    ap.add_argument("--file", required=True,
                    help="new results file name (already in benchmarks/results)")
    ap.add_argument("--version", default=None)
    args = ap.parse_args()

    reg = json.loads(REGISTRY.read_text())
    if args.axis not in reg:
        print(f"ERROR: unknown axis {args.axis!r}", file=sys.stderr)
        return 1
    entry = reg[args.axis]

    rows = [json.loads(ln)
            for ln in (RESULTS / args.file).read_text().splitlines()
            if ln.strip()]
    shas = {r.get("git_sha") for r in rows if r.get("git_sha")}
    if len(shas) != 1:
        print(f"ERROR: {args.file} carries shas {sorted(shas)}; a standings "
              "file must be single-sha", file=sys.stderr)
        return 1
    hosts = {(r["host"].get("name") if isinstance(r.get("host"), dict)
              else r.get("host")) for r in rows if r.get("host")}
    dates = sorted({r["ts"][:10] for r in rows if r.get("ts")})

    old = entry["file"]
    if old != args.file:
        if (RESULTS / old).exists():
            (RESULTS / old).unlink()
            print(f"superseded {old} (git history is the archive)")
        entry.pop("note", None)
    entry.update(file=args.file, sha=shas.pop(),
                 host=sorted(hosts)[0] if len(hosts) == 1 else None,
                 date=dates[-1] if dates else None,
                 refreshed_for=args.version or project_version())
    if check_standings.is_quality_axis(args.axis):
        stamp_quality_gate(entry)
    REGISTRY.write_text(json.dumps(reg, indent=2) + "\n")
    print(f"{args.axis}: {args.file} at {entry['sha']} "
          f"(refreshed for {entry['refreshed_for']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
