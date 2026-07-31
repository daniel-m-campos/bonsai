"""The unified bench CLI: python -m bonsai.bench <subcommand>.

    python -m bonsai.bench run --spec <bundled-name-or-path>
    python -m bonsai.bench run --axis rows,cols     # legacy scaling sugar
    python -m bonsai.bench plan --spec ...          # expansion only, no fits
    python -m bonsai.bench variants                 # the registry
    python -m bonsai.bench worker                   # child protocol (internal)

Spec mode resumes by default when the output file exists (ok/unsupported rows
are final; error/oom/timeout re-attempt); --no-resume forces a full rerun.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from . import params as rp
from . import runlog
from . import spec as spec_mod
from . import variants as vr
from .driver import run_jobs


def _default_out(spec: dict) -> str:
    repo = runlog.repo_root()
    name = f"{spec['name']}.jsonl"
    if repo is not None:
        return str(repo / "benchmarks" / "results" / name)
    return name


def _run(args) -> int:
    if args.spec is None and args.axis is None and not args.smoke:
        raise SystemExit("run needs --spec, --axis, or --smoke")
    if args.spec is not None and (args.axis is not None or args.smoke):
        raise SystemExit("--spec and --axis/--smoke are mutually exclusive")
    variants = args.variants.split(",") if args.variants else None

    if args.spec is None:
        # Legacy scaling sugar: same grid, knobs, and output file as
        # python -m bonsai.bench.scaling; resume stays opt-in there.
        from . import scaling
        axes = (["rows", "cols", "bins", "threads"]
                if args.axis in (None, "all")
                else [a.strip() for a in args.axis.split(",")])
        unknown_axes = set(axes) - {"rows", "cols", "bins", "threads"}
        if unknown_axes:
            raise SystemExit(f"unknown axes: {sorted(unknown_axes)}")
        for v in variants or []:
            if v not in scaling.VARIANTS:
                raise SystemExit(f"unknown scaling variant {v}")
        grid = scaling.build_grid(axes, args.smoke)
        repeats = args.repeats if args.repeats is not None else 1
        jobs = [{"cell": dict(cell), "variant": v, "threads": t,
                 "repeats": (max(repeats, 3)
                             if cell["axis"] == "base" and not args.smoke
                             else repeats)}
                for cell, t in grid
                for v in (variants or list(scaling.VARIANTS))]
        host = runlog.detect_host(args.host_name)
        knobs = dict(rp.SCALING, num_leaves_convention="full")
        return run_jobs(jobs, out=args.out or str(scaling.RESULTS),
                        suite="scaling", knobs=knobs, host=host,
                        run_label=args.run_label, dry_run=args.dry_run,
                        resume_path=args.resume, timeout_cap=args.timeout_cap,
                        gates={}, mem_sampler=not args.no_mem_sampler,
                        data_cache=args.data_cache)

    spec = spec_mod.load_spec(args.spec)
    jobs = spec_mod.expand(spec, variants=variants, repeats=args.repeats)
    out = args.out or _default_out(spec)
    resume = args.resume
    if resume is None and not args.no_resume and pathlib.Path(out).exists():
        resume = out
    host = runlog.detect_host(args.host_name)
    knobs = dict(spec.get("defaults", {}), num_leaves_convention="full")
    return run_jobs(jobs, out=out, suite=spec.get("suite", spec["name"]),
                    knobs=knobs, host=host,
                    run_label=args.run_label or spec["name"],
                    dry_run=args.dry_run, resume_path=resume,
                    timeout_cap=(args.timeout_cap
                                 if args.timeout_cap is not None
                                 else spec.get("timeout_cap", 3600)),
                    gates=spec.get("gates", {}),
                    mem_sampler=not args.no_mem_sampler,
                    data_cache=args.data_cache)


def _specs() -> int:
    for name in spec_mod.bundled_specs():
        s = spec_mod.load_spec(name)
        print(f"{name:28} suite={s.get('suite', s['name'])} "
              f"variants={len(s['variants'])}")
    return 0


def _variants(args) -> int:
    for v in vr.REGISTRY.values():
        if args.device and v.device != args.device:
            continue
        alias = f"  (aliases: {', '.join(v.aliases)})" if v.aliases else ""
        print(f"{v.name:28} {v.lib:10} {v.device}{alias}")
    return 0


def _worker() -> int:
    from .runners import worker
    child = json.loads(sys.stdin.read())
    print("RESULT " + json.dumps(worker(child)), flush=True)
    return 0


def _add_run_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--spec", default=None,
                   help="JSON spec file, or a bundled name (see `specs`)")
    p.add_argument("--axis", default=None,
                   help="legacy scaling grid: rows|cols|bins|threads|all")
    p.add_argument("--smoke", action="store_true",
                   help="legacy scaling smoke grid")
    p.add_argument("--out", default=None)
    p.add_argument("--run-label", default=None)
    p.add_argument("--host-name", default=None)
    p.add_argument("--variants", default=None)
    p.add_argument("--repeats", type=int, default=None)
    p.add_argument("--timeout-cap", type=int, default=None)
    p.add_argument("--resume", default=None,
                   help="jsonl of prior results (defaults to --out in spec mode)")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-mem-sampler", action="store_true",
                   help="disable the device-memory sampler")
    p.add_argument("--data-cache", default=None,
                   help="directory for memoized gen_data arrays (pods: /dev/shm)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bonsai.bench")
    sub = ap.add_subparsers(dest="command", required=True)
    _add_run_flags(sub.add_parser("run", help="execute a spec or legacy grid"))
    _add_run_flags(sub.add_parser("plan", help="print the expansion, run nothing"))
    pv = sub.add_parser("variants", help="print the variant registry")
    pv.add_argument("--device", choices=("cpu", "cuda"), default=None)
    sub.add_parser("specs", help="list the bundled campaign specs")
    sub.add_parser("worker", help="internal: job JSON on stdin, RESULT on stdout")
    args = ap.parse_args(argv)

    if args.command == "worker":
        return _worker()
    if args.command == "variants":
        return _variants(args)
    if args.command == "specs":
        return _specs()
    if args.command == "plan":
        args.dry_run = True
    if args.timeout_cap is None and args.spec is None:
        args.timeout_cap = 3600
    return _run(args)


if __name__ == "__main__":
    sys.exit(main())
