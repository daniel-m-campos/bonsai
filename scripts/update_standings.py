#!/usr/bin/env python3
"""Point a standings axis at a freshly measured file (decision 92).

    python3 scripts/update_standings.py --axis rows --file rebaseline-2026-08.jsonl

Reads sha/host/date from the new file's rows, deletes the superseded file
(git history is the archive), and rewrites the registry entry with
refreshed_for taken from pyproject.toml (override with --version).
Supersession also drops the entry's note, since a note describes the
file being replaced. Run render_results.py afterwards; the generator
loads through the registry.

`--companion` registers a second file measured in the same session (the
gpu-tall axis carries its parity rows that way, which is where the perf
page's fused anchor comes from). It supersedes with its axis and the
renderer holds it to the same rendered-or-removed rule.

Superseding an axis that carries a plane (or a quality axis) also stamps
`hash_set` (its plane's digest, `check_standings.plane_digest()`) and `refs`
(the reference libraries' installed versions) so a later release can use
check_standings.py's hash-unchanged skip instead of a full refresh, and so
`--stale` can tell the next refresh which axes to measure. The refresh is the
one moment the installed reference-library versions are known for certain (the
suites run against them), so it also refreshes
benchmarks/reference_versions.json to match.

`--restamp-verified` is the other, much narrower way an axis's stamp moves:

    python3 scripts/update_standings.py --axis gpu-tall --restamp-verified \
        --reason "host-side fill only; src/cuda byte-identical" \
        --evidence-before 9f0a1c2d3e4f5061 --evidence-after 9f0a1c2d3e4f5061

It carries the current plane digest forward onto an axis the changed sources
provably cannot reach, with no new measurement. The measured sha, results
file, host, date, and refreshed_for are left exactly as the last real run set
them, and a `carried_forward` block records the equivalence argument, so the
entry stays visibly a carry-forward rather than a run. It refuses without a
reason and a pair of equal evidence hashes, refuses on an axis that is
already current, and refuses when the staleness comes from a reference-library
major rather than the source tree. Superseding an axis drops the block; a
carry-forward is spent the moment the plane moves again.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
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


def stamp_skip_gate(entry: dict) -> None:
    """Record the hash-unchanged skip's inputs on a freshly refreshed axis.

    A perf axis stamps its own plane's digest, so a later change confined to
    the other plane leaves it provably current; a quality axis has no plane
    and stamps the whole-implementation digest, as it always did.
    """
    entry["hash_set"] = check_standings.plane_digest(
        entry.get("plane") or check_standings.PLANE_GPU)
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


def head_sha() -> str:
    """The short sha of the checkout doing the stamping."""
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                          capture_output=True, text=True,
                          check=True).stdout.strip()


def restamp_verified(args: argparse.Namespace, reg: dict) -> int:
    """Carry the current plane digest forward onto an unreachable axis.

    Writes no measurement: the axis keeps the file, sha, host, date, and
    refreshed_for of its last real run and gains a `carried_forward` block
    holding the equivalence argument that justifies the new stamp.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments; needs `axis`, `reason`, `evidence`,
        `evidence_before`, and `evidence_after`.
    reg : dict
        The loaded registry, mutated and written on success.

    Returns
    -------
    int
        0 on success, 1 when the carry-forward is refused.
    """
    entry = reg[args.axis]
    missing = [name for name, value in (("--reason", args.reason),
                                        ("--evidence-before", args.evidence_before),
                                        ("--evidence-after", args.evidence_after))
               if not value]
    if missing:
        print(f"ERROR: --restamp-verified needs {', '.join(missing)}: a stamp "
              "without a recorded proof is a guess, not a carry-forward",
              file=sys.stderr)
        return 1
    if args.evidence_before != args.evidence_after:
        print(f"ERROR: evidence {args.evidence_before!r} and "
              f"{args.evidence_after!r} differ; that is a measured difference, "
              "not an equivalence proof", file=sys.stderr)
        return 1
    if not entry.get("sha"):
        print(f"ERROR: {args.axis} has never been measured; there is no "
              "measurement to carry forward", file=sys.stderr)
        return 1
    plane = entry.get("plane") or (check_standings.PLANE_GPU
                                   if check_standings.is_quality_axis(args.axis)
                                   else None)
    if plane is None:
        print(f"ERROR: {args.axis} registers no plane, so there is no digest "
              "to carry forward", file=sys.stderr)
        return 1
    ok, reason = check_standings.hash_skip(args.axis, entry)
    if ok:
        print(f"ERROR: {args.axis} is already current ({reason}); nothing to "
              "carry forward", file=sys.stderr)
        return 1
    bumped = check_standings.bumped_ref_majors(entry.get("refs", {}))
    if bumped:
        print(f"ERROR: {args.axis} is stale on reference-library majors "
              f"({', '.join(bumped)}), which no source-tree equivalence "
              "answers; re-measure the axis", file=sys.stderr)
        return 1

    entry["hash_set"] = check_standings.plane_digest(plane)
    entry["carried_forward"] = {
        "measured_at": entry["sha"],
        "stamped_at": head_sha(),
        "reason": args.reason,
        "evidence": {"kind": args.evidence,
                     "before": args.evidence_before,
                     "after": args.evidence_after},
    }
    REGISTRY.write_text(json.dumps(reg, indent=2) + "\n")
    print(f"{args.axis}: {plane} stamp carried forward to "
          f"{entry['carried_forward']['stamped_at']}, still measured at "
          f"{entry['sha']} ({args.reason})")
    return 0


def supersede(args: argparse.Namespace, reg: dict) -> int:
    """Point an axis at a freshly measured file and restamp its skip gate.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments; needs `axis`, `file`, `version`, and `companion`.
    reg : dict
        The loaded registry, mutated and written on success.

    Returns
    -------
    int
        0 on success, 1 when the new file cannot back a standings claim.
    """
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

    # file is null on an axis that has never been measured (registry v2
    # placeholder), so there is nothing to supersede on its first refresh.
    old = entry.get("file")
    if old != args.file:
        if old and (RESULTS / old).exists():
            (RESULTS / old).unlink()
            print(f"superseded {old} (git history is the archive)")
        entry.pop("note", None)
    old_companion = entry.get("companion")
    if args.companion and old_companion != args.companion:
        if old_companion and (RESULTS / old_companion).exists():
            (RESULTS / old_companion).unlink()
            print(f"superseded {old_companion} (git history is the archive)")
        entry["companion"] = args.companion
    entry.update(file=args.file, sha=shas.pop(),
                 host=sorted(hosts)[0] if len(hosts) == 1 else None,
                 date=dates[-1] if dates else None,
                 refreshed_for=args.version or project_version())
    # A measured stamp is never a carried-forward one.
    entry.pop("carried_forward", None)
    if check_standings.is_quality_axis(args.axis) or entry.get("plane"):
        stamp_skip_gate(entry)
    REGISTRY.write_text(json.dumps(reg, indent=2) + "\n")
    print(f"{args.axis}: {args.file} at {entry['sha']} "
          f"(refreshed for {entry['refreshed_for']})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True)
    ap.add_argument("--file", default=None,
                    help="new results file name (already in benchmarks/results)")
    ap.add_argument("--version", default=None)
    ap.add_argument("--companion", default=None,
                    help="evidence file measured with this axis (rendered "
                         "with it, superseded with it)")
    ap.add_argument("--restamp-verified", action="store_true",
                    help="carry the current plane digest forward with no new "
                         "measurement, for an axis the changed sources cannot "
                         "reach; needs --reason and the evidence hashes")
    ap.add_argument("--reason", default=None,
                    help="carry-forward: why the changed plane cannot reach "
                         "this axis")
    ap.add_argument("--evidence", default="model-hash",
                    help="carry-forward: what the equality proof is")
    ap.add_argument("--evidence-before", default=None,
                    help="carry-forward: the proof's value at the measured sha")
    ap.add_argument("--evidence-after", default=None,
                    help="carry-forward: the proof's value at the stamped sha")
    args = ap.parse_args()

    reg = json.loads(REGISTRY.read_text())
    if args.axis not in reg:
        print(f"ERROR: unknown axis {args.axis!r}", file=sys.stderr)
        return 1
    if args.restamp_verified:
        return restamp_verified(args, reg)
    if not args.file:
        print("ERROR: --file is required unless --restamp-verified is given",
              file=sys.stderr)
        return 1
    return supersede(args, reg)


if __name__ == "__main__":
    sys.exit(main())
