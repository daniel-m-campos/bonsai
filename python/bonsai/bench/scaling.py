"""Scaling benchmark suite (perf division): synthetic regression swept over
rows / cols / bins / threads, bonsai vs xgboost / lightgbm / catboost.

    python -m bonsai.bench.scaling --smoke
    python -m bonsai.bench.scaling --axis all

This entry point is sugar over the unified CLI (python -m bonsai.bench run);
the grid definitions below are the historical axes it expands. Rows are
labeled division="perf", timing_mode="in_memory" (schema v1,
bonsai.bench.runlog); the synthetic recipe and its provenance live in
bonsai.bench.synth. See docs/method/benchmark-protocol.md.

Methodology (decision 46): in-memory float32 numpy arrays through each
library's Python API — bonsai via the nanobind module (make python-cuda on GPU
hosts; the Makefile bench-scaling target sets PYTHONPATH). fit_s times each
library's own ingestion (bonsai ColumnBatch + binning, xgboost QuantileDMatrix,
lgb.Dataset, catboost Pool) plus training; predict_s times prediction from a
raw test matrix. Quality is R^2 on train and a held-out test split.

Every (cell, variant, repeat) runs in a child process (the worker protocol in
bonsai.bench.runners): OOM and segfaults become jsonl status lines instead of
killing the sweep, and bonsai's exit-time profilers flush per-run breakdowns
to the child's stderr where the parent captures them.

    make bench-scaling ARGS="--axis all"
    uv run scripts/bench_scaling.py --smoke            # tiny grid (Mac)
    uv run scripts/bench_scaling.py --dry-run --axis all

Results append to the --out file (see the standings policy in
docs/method/benchmark-protocol.md). Grid corners the host cannot fit are recorded as
status="skipped" with the memory estimate — the feasibility frontier is data.
"""
import argparse
import pathlib
import sys

from . import runlog
from . import variants as vr
from .driver import GPU_MAX_COLS
from .runners import RUNNERS, worker
from .synth import gen_data

__all__ = ["AXES", "BASE", "GPU_MAX_COLS", "RESULTS", "RUNNERS", "VARIANTS",
           "gen_data", "main", "worker"]

# In-repo runs default next to the other results; wheel installs default to
# the working directory.
_repo = runlog.repo_root()
RESULTS = (_repo / "benchmarks" / "results" / "scaling.jsonl"
           if _repo is not None else pathlib.Path("scaling.jsonl"))

BASE = {"rows": 1_000_000, "cols": 100, "bins": 255, "depth": 8, "iters": 100,
        "lr": 0.1, "informative": 20, "seed": 42}

AXES = {
    "rows": [(250_000, 100), (1_000_000, 100), (4_000_000, 100), (16_000_000, 100)],
    # (rows, cols): rows shrink past 4k cols to cap cells at 2^31.
    "cols": [(1_000_000, 16), (1_000_000, 64), (1_000_000, 256), (1_000_000, 1024),
             (1_000_000, 4096), (131_072, 16_384), (32_768, 65_536)],
    "bins": [15, 63, 255, 1023, 4095, 16_383, 65_535],
    "threads": [1, 4, 16, 64],
}

# variant -> (library, device), a view of the registry in
# bonsai.bench.variants. lgbm_cuda needs the CUDA source build baked into the
# bonsai-ci image (the PyPI wheel is CPU-only).
VARIANTS = {n: (vr.resolve(n).lib, vr.resolve(n).device) for n in vr.SCALING}


def make_cell(axis: str, rows: int, cols: int, bins: int) -> dict:
    n_test = min(rows // 5, 500_000)
    return dict(axis=axis, rows=rows, cols=cols, bins=bins, bins_effective=bins,
                depth=BASE["depth"], iters=BASE["iters"], lr=BASE["lr"],
                informative=BASE["informative"], n_test=n_test, seed=BASE["seed"])


def build_grid(axes: list[str], smoke: bool) -> list[tuple[dict, int]]:
    """Returns [(cell, threads)]; the base cell appears once with axis='base'."""
    if smoke:
        cells = [make_cell("rows", r, c, b)
                 for r in (50_000, 100_000) for c in (16, 64) for b in (15, 63)]
        for cell in cells:
            cell.update(iters=20, n_test=10_000)
        return [(c, 16) for c in cells]
    grid: list[tuple[dict, int]] = [(make_cell("base", BASE["rows"], BASE["cols"],
                                               BASE["bins"]), 16)]
    if "rows" in axes:
        grid += [(make_cell("rows", r, c, BASE["bins"]), 16)
                 for r, c in AXES["rows"] if r != BASE["rows"]]
    if "cols" in axes:
        grid += [(make_cell("cols", r, c, BASE["bins"]), 16)
                 for r, c in AXES["cols"] if c != BASE["cols"]]
    if "bins" in axes:
        grid += [(make_cell("bins", BASE["rows"], BASE["cols"], b), 16)
                 for b in AXES["bins"] if b != BASE["bins"]]
    if "threads" in axes:
        grid += [(make_cell("threads", BASE["rows"], BASE["cols"], BASE["bins"]), t)
                 for t in AXES["threads"] if t != 16]
    return grid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--axis", default="all",
                    help="rows|cols|bins|threads|all (comma-separable)")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--host-name", default=None)
    ap.add_argument("--timeout-cap", type=int, default=3600)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", default=None,
                    help="jsonl of prior results; ok/unsupported runs are skipped")
    ap.add_argument("--out", default=str(RESULTS))
    args = ap.parse_args()

    if args.worker:
        from .cli import _worker
        return _worker()

    from .cli import main as cli_main
    argv = ["run", "--axis", args.axis, "--variants", args.variants,
            "--repeats", str(args.repeats), "--timeout-cap",
            str(args.timeout_cap), "--out", args.out]
    if args.host_name:
        argv += ["--host-name", args.host_name]
    if args.resume:
        argv += ["--resume", args.resume]
    if args.smoke:
        argv += ["--smoke"]
    if args.dry_run:
        argv += ["--dry-run"]
    return cli_main(argv)


if __name__ == "__main__":
    sys.exit(main())
