"""Device memory for k CV folds: materialized Datasets against one plus views.

The materialized arm is what every library does: build a Dataset per fold and
hold them all at once, which is what xgb.cv and lgb.cv both do. The view arm
is one Dataset plus k row-index arrays, so the plane is paid once.

Device memory is read per process from NVML, not device-wide totals, so a
co-tenant cannot contaminate the number.

    PYTHONPATH=build-cuda/python python3 probe_fold_memory.py --rows 16000000
"""
from __future__ import annotations

import argparse

import numpy as np
import pynvml

import bonsai


def used_mb() -> float:
    """This process's device memory, via NVML."""
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    mine = 0
    import os
    pid = os.getpid()
    for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h):
        if p.pid == pid:
            mine = p.usedGpuMemory or 0
    return mine / (1024 * 1024)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=16_000_000)
    ap.add_argument("--cols", type=int, default=128)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--frac", type=float, default=0.8)
    args = ap.parse_args()

    n, f, k = args.rows, args.cols, args.folds
    m = int(n * args.frac)
    rng = np.random.default_rng(0)
    X = rng.random((n, f), dtype=np.float32)
    y = X[:, 0].astype(np.float32)

    base = used_mb()
    parent = bonsai.Dataset(X, y, device="cuda")
    one = used_mb() - base
    print(f"rows={n} cols={f} folds={k} subset={m}")
    print(f"  one Dataset on device: {one:8.1f} MiB")

    # materialized: k fold Datasets alive at once, the xgb.cv / lgb.cv shape
    held = []
    for i in range(k):
        lo = int(i * n / k)
        rows = np.r_[0:lo, lo + (n // k):n][:m]
        held.append(bonsai.Dataset(X[rows], y[rows], device="cuda"))
    mat = used_mb() - base
    print(f"  + {k} materialized folds:  {mat:8.1f} MiB  "
          f"({mat / one:.1f}x the plane)")
    del held

    # view: the same k folds as row-index arrays against the one plane
    idx_mb = k * m * 4 / (1024 * 1024)
    print(f"  + {k} views (4B/row idx): {one + idx_mb:8.1f} MiB  "
          f"({(one + idx_mb) / one:.1f}x the plane)")
    print(f"\n  saving at k={k}: {mat - (one + idx_mb):.0f} MiB "
          f"({mat / (one + idx_mb):.1f}x less device memory)")
    print("  a range view needs no index array at all, just two integers.")
    del parent
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
