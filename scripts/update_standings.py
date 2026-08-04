#!/usr/bin/env python3
"""Point a standings axis at a freshly measured file (decision 92).

    python3 scripts/update_standings.py --axis rows --file rebaseline-2026-08.jsonl

Reads sha/host/date from the new file's rows, deletes the superseded file
(git history is the archive), and rewrites the registry entry with
refreshed_for taken from pyproject.toml (override with --version).
Supersession also drops the entry's note, since a note describes the
file being replaced. Run render_results.py afterwards; the generator
loads through the registry.
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


def project_version() -> str:
    m = re.search(r'^version = "([^"]+)"',
                  (REPO / "pyproject.toml").read_text(), re.M)
    return m.group(1)


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
    REGISTRY.write_text(json.dumps(reg, indent=2) + "\n")
    print(f"{args.axis}: {args.file} at {entry['sha']} "
          f"(refreshed for {entry['refreshed_for']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
