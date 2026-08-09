"""The unified bench CLI: python -m bonsai.bench <subcommand>.

    python -m bonsai.bench run --spec <bundled-name-or-path>
    python -m bonsai.bench plan --spec ...          # expansion only, no fits
    python -m bonsai.bench variants                 # the registry
    python -m bonsai.bench worker                   # child protocol (internal)

Runs resume by default when the output file exists (ok/unsupported rows are
final; error/oom/timeout re-attempt); --no-resume forces a full rerun.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from bonsai.bench import runlog
from bonsai.bench import spec as spec_mod
from bonsai.bench import variants as vr
from bonsai.bench.driver import run_jobs


def main(argv: list[str] | None = None) -> int:
    """Parse the subcommand and dispatch; `plan` is `run` with dry_run."""
    ap = argparse.ArgumentParser(prog="bonsai.bench")
    sub = ap.add_subparsers(dest="command", required=True)
    _add_run_flags(sub.add_parser("run", help="execute a spec"))
    _add_run_flags(sub.add_parser("plan", help="print the expansion, run nothing"))
    pv = sub.add_parser("variants", help="print the variant registry")
    pv.add_argument("--device", choices=(vr.Device.CPU, vr.Device.CUDA), default=None)
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
    return _run(args)


def _run(args) -> int:
    """Expand the spec, resume against the output file by default, run."""
    if args.spec is None:
        raise SystemExit("run needs --spec")
    variants = args.variants.split(",") if args.variants else None
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


def _default_out(spec: dict) -> str:
    """benchmarks/results/<name>.jsonl in a checkout, else the bare name."""
    repo = runlog.repo_root()
    name = f"{spec['name']}.jsonl"
    if repo is not None:
        return str(repo / "benchmarks" / "results" / name)
    return name


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
    from bonsai.bench.runners import worker
    child = json.loads(sys.stdin.read())
    print("RESULT " + json.dumps(worker(child)), flush=True)
    return 0


def _add_run_flags(p: argparse.ArgumentParser):
    p.add_argument("--spec", default=None,
                   help="JSON spec file, or a bundled name (see `specs`)")
    p.add_argument("--out", default=None)
    p.add_argument("--run-label", default=None)
    p.add_argument("--host-name", default=None)
    p.add_argument("--variants", default=None)
    p.add_argument("--repeats", type=int, default=None)
    p.add_argument("--timeout-cap", type=int, default=None)
    p.add_argument("--resume", default=None,
                   help="jsonl of prior results (defaults to --out)")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-mem-sampler", action="store_true",
                   help="disable the device-memory sampler")
    p.add_argument("--data-cache", default=None,
                   help="directory for memoized gen_data arrays (pods: /dev/shm)")


if __name__ == "__main__":
    sys.exit(main())
