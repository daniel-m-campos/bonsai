"""The CPU counterpart: price a row view against a materialized copy.

Same arms as the CUDA probe, same invariant: every arm does m rows of
histogram work per level and differs only in how the fill reaches the bins.
The view arms go through `Dataset.subset(rows=)`, so the shape the fill sees
is the row descriptor's: --segments 1 is a Range, more is Segments, and the
scatter arm is a Gather.

Two CPU-specific effects this is built to expose, both from src/grower.cpp:

    dense fast path (:42)  `n == ds.n_rows()` picks `bins[k]` over
        `bins[rows[k]]`. A materialized copy takes it; a view never can,
        so a view pays one extra load per element even at density 1.0.

    fill cutoff (:68,:126)  `rows.size() * 4 >= ds.n_rows()` picks the
        column fill over the row-major fill, and the denominator is the
        DATASET, not the view. Above 25% both arms take the column fill.
        Below it a view drops to the row fill while an equal-sized copy
        does not, so the arms diverge by code path rather than locality.
        Sweep --fracs across 0.25 to see it.

Timing is the `populate` bucket of BONSAI_GROW_PROFILE, the host counterpart
of the CUDA root_hist/adv_hist spans, so split finding and partitioning are
excluded. Arms interleave within a rep; the figure is the median.

    PYTHONPATH=build/python python3 probe_cpu_view.py --rows 2000000
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys

WORKER = r"""
import json, sys
import numpy as np
import bonsai

spec = json.loads(sys.argv[1])
n, f, m, k = spec["n"], spec["f"], spec["m"], spec["segments"]
rng = np.random.default_rng(0)
X = rng.random((n, f), dtype=np.float32)
y = (X[:, 0] * 2 + X[:, 1] - X[:, 2] + rng.normal(0, 0.1, n)).astype(np.float32)

params = {"dispatch.grower_name": "depthwise", "booster.n_iters": spec["iters"],
          "tree.max_depth": spec["depth"], "booster.learning_rate": 0.1,
          "parallel.n_threads": spec["threads"]}


# m row ids as k evenly spaced contiguous runs; k=0 is a scatter.
def rows_of(k):
    if k <= 0:
        return np.sort(rng.choice(n, m, replace=False))
    per, stride = max(1, m // k), n // k
    runs = [np.arange(s * stride, min(s * stride + per, n)) for s in range(k)]
    return np.concatenate(runs)[:m]


# The Dataset constructor carries its OWN n_threads: leaving it at the
# default binds with every advertised core, which on a quota-metered
# container over-subscribes and throttles the run being measured.
if spec["arm"] == "copy":
    ds = bonsai.Dataset(X[:m], y[:m], n_threads=spec["threads"])
else:
    ds = bonsai.Dataset(X, y, n_threads=spec["threads"]).subset(rows=rows_of(k))
    print("VIEW", repr(ds), flush=True)

bonsai.train(bonsai.Params.from_dict({**params, "booster.n_iters": 2}), ds)
bonsai.train(bonsai.Params.from_dict(params), ds)
print("WORKER_DONE", flush=True)
"""

GROW = re.compile(r"grow-profile: (.+)")


def populate_seconds(stderr: str) -> tuple[float, float]:
    """The last grow-profile block's populate and finalize buckets.

    finalize holds `route_unsampled`, which walks every row NOT in the row
    list through the finished tree to keep its score current, and no-ops
    when the list covers the dataset. So the copy arm skips it and a view
    arm pays it: the gap between the two finalize numbers is that cost.
    """
    blocks = GROW.findall(stderr)
    if not blocks:
        raise SystemExit("no grow-profile block; is BONSAI_GROW_PROFILE set?")
    fields = dict(kv.split("=") for kv in blocks[-1].split())
    return (float(fields["populate"].rstrip("s")),
            float(fields["finalize"].rstrip("s")))


def run(arm: str, segments: int, spec: dict, python: str,
        mod: str = "") -> tuple[float, float]:
    env = {**os.environ, "BONSAI_GROW_PROFILE": "1"}
    env.pop("BONSAI_PROBE_SEGMENTS", None)
    if mod:
        env["PYTHONPATH"] = mod
    payload = json.dumps({**spec, "arm": arm, "segments": segments})
    proc = subprocess.run([python, "-c", WORKER, payload],
                          capture_output=True, text=True, env=env)
    if "WORKER_DONE" not in proc.stdout:
        print(proc.stdout[-1200:], proc.stderr[-2000:], sep="\n", file=sys.stderr)
        raise SystemExit(f"arm {arm}/{segments} failed")
    return populate_seconds(proc.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2_000_000)
    ap.add_argument("--cols", type=int, default=64)
    ap.add_argument("--fracs", default="0.8")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--segments", default="1,160,10000")
    ap.add_argument("--python", default=sys.executable)
    # A/B: label=path pairs, each arm run once per module, interleaved.
    ap.add_argument("--mods", default="")
    args = ap.parse_args()

    ks = [int(s) for s in args.segments.split(",") if s]
    shapes = ([("copy", "copy", 0)] + [(f"seg{k}", "view", k) for k in ks]
              + [("scatter", "view", 0)])
    mods = ([tuple(m.split("=")) for m in args.mods.split(",")] if args.mods
            else [("", "")])
    arms = [(f"{ml}:{label}" if ml else label, arm, seg, mp)
            for label, arm, seg in shapes for ml, mp in mods]

    for frac in [float(s) for s in args.fracs.split(",")]:
        spec = {"n": args.rows, "f": args.cols, "m": int(args.rows * frac),
                "iters": args.iters, "depth": args.depth, "threads": args.threads}
        cutoff = "column fill" if frac * 4 >= 1.0 else "ROW fill (view only)"
        print(f"\n=== rows={args.rows} cols={args.cols} frac={frac} "
              f"subset={spec['m']} -> view routes to {cutoff}", flush=True)
        pop: dict[str, list[float]] = {label: [] for label, _, _, _ in arms}
        fin: dict[str, list[float]] = {label: [] for label, _, _, _ in arms}
        for r in range(args.reps):
            for label, arm, seg, mp in arms:
                p_s, f_s = run(arm, seg, spec, args.python, mp)
                pop[label].append(p_s)
                fin[label].append(f_s)
                print(f"  rep{r} {label:8s} populate={p_s:.4f}s finalize={f_s:.4f}s",
                      flush=True)
        first = arms[0][0]
        base = statistics.median(pop[first])
        fbase = statistics.median(fin[first])
        print(f"\n  {'arm':10s} {'populate':>10s} {'vs copy':>9s} {'spread':>8s}"
              f" | {'finalize':>9s} {'vs copy':>9s}")
        for label, _, _, _ in arms:
            med = statistics.median(pop[label])
            fmed = statistics.median(fin[label])
            spread = (max(pop[label]) - min(pop[label])) / med * 100 if med else 0.0
            fdelta = fmed / fbase - 1 if fbase else 0.0
            print(f"  {label:10s} {med:9.4f}s {med / base - 1:+8.1%} {spread:7.1f}%"
                  f" | {fmed:8.4f}s {fdelta:+8.1%}")
        print("  finalize holds route_unsampled: the copy arm skips it (its rows "
              "are its whole dataset), a view pays it over the complement.")
    print("\ncopy takes the dense fast path (bins[k]); every view takes "
          "bins[rows[k]], so a view pays one extra load per element even at "
          "density 1.0. That floor is what seg1 measures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
