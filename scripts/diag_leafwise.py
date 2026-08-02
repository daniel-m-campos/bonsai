"""TEMPORARY stage-3 diagnosis driver: one profiled bonsai fit at a named cell.

Not for merge. Prints a RESULT line on stdout; the engine's profile lines
land on stderr at interpreter exit.
"""

from __future__ import annotations

import os
import pathlib
import sys
import time

import numpy as np


def _data(rows: int, cols: int, n_test: int):
    cache = pathlib.Path(os.environ.get("CACHE", "/dev/shm/diagcache"))
    cache.mkdir(parents=True, exist_ok=True)
    tag = f"{rows}_{cols}_{n_test}"
    paths = [cache / f"{name}_{tag}.npy" for name in ("X", "y", "Xte", "yte")]
    if all(p.exists() for p in paths):
        return [np.load(p, mmap_mode="r") for p in paths]
    from bonsai.bench.synth import gen_data

    arrays = gen_data(rows, cols, 42, n_test, 20)
    for p, a in zip(paths, arrays):
        np.save(p, a)
    return arrays


def main() -> int:
    rows = int(os.environ.get("ROWS", "16000000"))
    cols = int(os.environ.get("COLS", "100"))
    depth = int(os.environ.get("DEPTH", "8"))
    leaves = int(os.environ.get("LEAVES", "256"))
    iters = int(os.environ.get("ITERS", "100"))
    grower = os.environ.get("GROWER", "cuda_leafwise")
    n_test = int(os.environ.get("NTEST", "100000"))

    import bonsai
    from bonsai.bench import params as rp

    X, y, Xte, yte = _data(rows, cols, n_test)
    X = np.ascontiguousarray(X)
    y = np.ascontiguousarray(y)
    ds = bonsai.Dataset(X, y, max_bin=255)
    pairs = [
        (k, v)
        for k, v in rp.bonsai_core(
            learning_rate=0.1,
            max_depth=depth,
            num_leaves=leaves,
            min_data_in_leaf=20,
            lambda_l2=1.0,
            max_bin=255,
            seed=42,
            n_iters=iters,
            n_threads=16,
            grower=grower,
        )
        if not k.startswith("bin_mapper.")
    ]
    t0 = time.perf_counter()
    model = bonsai.train(pairs, ds)
    train_s = time.perf_counter() - t0
    pred = np.asarray(model.predict(np.ascontiguousarray(Xte)))
    yte = np.asarray(yte)
    r2 = 1.0 - ((yte - pred) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()
    print(
        f"RESULT grower={grower} rows={rows} cols={cols} depth={depth} "
        f"leaves={leaves} iters={iters} train_s={train_s:.3f} r2={r2:.6f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
