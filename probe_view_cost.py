"""Price row-subset VIEWS of varying fragmentation against a materialized copy.

Every arm does the SAME arithmetic: m rows of histogram work per level, the
same tree depth, the same iteration count. They differ only in how the
kernel reaches the bins.

    copy      Dataset built from X[:m]: m rows, root iota over a compact
              matrix. What every library does today, and the floor.
    seg<k>    Dataset of all n rows, sampler emits m rows as k evenly spaced
              contiguous runs. k=1 is a walk-forward prefix. k=160 is
              triple-barrier purging at H=1000. k=3300 is its worst case.
    scatter   Dataset of all n rows, bernoulli draw: m ascending but fully
              scattered indices. The upper bound on fragmentation.

Timing is the device-side histogram spans (root_hist + adv_hist) from
BONSAI_CUDA_PROFILE, so host sampling cost and the per-tree row upload are
excluded by construction. Arms interleave within each rep so drift cannot
masquerade as an effect; the reported figure is the median over reps.

    PYTHONPATH=build-cuda/python python3 probe_view_cost.py --rows 4000000
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
n, f, m = spec["n"], spec["f"], spec["m"]
rng = np.random.default_rng(0)
X = rng.random((n, f), dtype=np.float32)
y = (X[:, 0] * 2 + X[:, 1] - X[:, 2] + rng.normal(0, 0.1, n)).astype(np.float32)

base = {"dispatch.grower_name": "cuda_depthwise", "booster.n_iters": spec["iters"],
        "tree.max_depth": spec["depth"], "booster.learning_rate": 0.1}
if spec["arm"] == "copy":
    ds = bonsai.Dataset(X[:m], y[:m], device="cuda")
    params = base
else:
    ds = bonsai.Dataset(X, y, device="cuda")
    params = {**base, "dispatch.sampler_name": "bernoulli",
              "sampler.subsample": m / n}

warm = bonsai.Params.from_dict({**params, "booster.n_iters": 2})
bonsai.train(warm, ds)          # the cold block is never the one parsed
bonsai.train(bonsai.Params.from_dict(params), ds)
print("WORKER_DONE", flush=True)
"""

ROUND = re.compile(r"cuda-round-decomp: (.+)")


def hist_seconds(stderr: str) -> tuple[float, float]:
    """root_hist and adv_hist from the LAST round-decomp block (the timed fit)."""
    blocks = ROUND.findall(stderr)
    if not blocks:
        raise SystemExit("no cuda-round-decomp block: is this a CUDA build?")
    fields = dict(kv.split("=") for kv in blocks[-1].split())
    return (float(fields["root_hist"].rstrip("s")),
            float(fields["adv_hist"].rstrip("s")))


def run(arm: str, segments: int, spec: dict, python: str) -> tuple[float, float]:
    env = {**os.environ, "BONSAI_CUDA_PROFILE": "1"}
    if segments:
        env["BONSAI_PROBE_SEGMENTS"] = str(segments)
    else:
        env.pop("BONSAI_PROBE_SEGMENTS", None)
    proc = subprocess.run([python, "-c", WORKER, json.dumps({**spec, "arm": arm})],
                          capture_output=True, text=True, env=env)
    if "WORKER_DONE" not in proc.stdout:
        print(proc.stdout[-1500:], proc.stderr[-2500:], sep="\n", file=sys.stderr)
        raise SystemExit(f"arm {arm}/{segments} failed")
    return hist_seconds(proc.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4_000_000)
    ap.add_argument("--cols", type=int, default=128)
    ap.add_argument("--frac", type=float, default=0.8)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--segments", default="1,16,160,1024,3300")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    ks = [int(s) for s in args.segments.split(",") if s]
    spec = {"n": args.rows, "f": args.cols, "m": int(args.rows * args.frac),
            "iters": args.iters, "depth": args.depth}
    # (label, arm, segment count); copy first so it anchors each rep
    arms = [("copy", "copy", 0)]
    arms += [(f"seg{k}", "view", k) for k in ks]
    arms += [("scatter", "view", 0)]

    print(f"rows={args.rows} cols={args.cols} subset={spec['m']} "
          f"iters={args.iters} depth={args.depth} reps={args.reps}", flush=True)
    # root_hist is one node over all m rows at depth 0: pure access pattern,
    # no tree structure. adv_hist folds in how rows land across nodes, which
    # differs between a deterministic subset and a per-tree random draw, so
    # only root_hist isolates locality.
    root: dict[str, list[float]] = {label: [] for label, _, _ in arms}
    adv: dict[str, list[float]] = {label: [] for label, _, _ in arms}
    for r in range(args.reps):
        for label, arm, seg in arms:
            rt, ad = run(arm, seg, spec, args.python)
            root[label].append(rt)
            adv[label].append(ad)
            print(f"  rep{r} {label:8s} root={rt:.4f}s adv={ad:.4f}s", flush=True)

    rbase = statistics.median(root["copy"])
    abase = statistics.median(adv["copy"])
    print(f"\n{'arm':10s} {'root_hist':>10s} {'vs copy':>9s} {'spread':>7s}"
          f" | {'adv_hist':>9s} {'vs copy':>9s}")
    for label, _, _ in arms:
        rmed = statistics.median(root[label])
        amed = statistics.median(adv[label])
        rspread = (max(root[label]) - min(root[label])) / rmed * 100 if rmed else 0.0
        print(f"{label:10s} {rmed:9.4f}s {rmed / rbase - 1:+8.1%} {rspread:6.1f}%"
              f" | {amed:8.4f}s {amed / abase - 1:+8.1%}")
    print("\nroot_hist is the controlled comparison: identical work, identical "
          "node structure, only index locality varies.")
    print("adv_hist additionally reflects tree shape, which the arms do not "
          "share; read it as context, not as evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
