"""The synthetic dataset behind bonsai's perf-division benchmarks.

Provenance: a generalized form of Friedman #1 (Friedman, "Multivariate
Adaptive Regression Splines", Annals of Statistics 1991), the standard
synthetic regression for tree methods. The classic five-feature target

    10 sin(pi x0 x1) + 20 (x2 - 0.5)^2 + 10 x3 + 5 x4

is repeated over `informative` features in blocks of five with geometrically
decaying block weights (0.6^b), over uniform [0, 1) float32 features, plus
Gaussian noise with sigma = y.std() / 3, sizing the best-achievable test R^2
at roughly 0.9. Why synthetic for perf: rows, cols, and bins become free
axes (decision 46), and generation is deterministic in (seed, rows, cols,
n_test) alone, so bins/threads sweeps reuse byte-identical data.

Recipe 2 draws the rows in N_BLOCKS fixed blocks off spawned streams so
generation runs on every core the container may use. The block count is part
of the contract and the worker count is not: derive parallelism from cores
if you like, never the data. Rows generated under recipe 1, which drew the
whole matrix from one stream, are NOT byte-comparable with these; every row
records `data_recipe` for exactly that reason. See decision 112.

Note: scripts/model_hash.py contains a deliberately FROZEN linear variant of
this recipe. Its output feeds the cross-arch byte-identity CI gate; it must
never be edited or replaced with this function.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from bonsai.bench import runlog

# Bumped whenever the bytes change. Committed rows carry it, so a mixed
# results directory is readable as mixed instead of looking comparable.
DATA_RECIPE = 2

# How many row blocks the matrix is drawn in, one spawned stream each. This
# is DATA, not scheduling: change it and every byte changes, which is why it
# is a constant here and never a core count. Sixty-four covers every quota a
# rented pod has actually metered (13.6 and 27.2 cores) with room over.
N_BLOCKS = 64


def gen_data(rows: int, cols: int, seed: int, n_test: int, informative: int):
    """Generalized Friedman-1 regression data, byte-stable by contract.

    k informative features in blocks of 5, decaying block weights, noise
    sized for a best-achievable R^2 of ~0.9. Deterministic in (rows, cols,
    seed, n_test) only, so bins/threads sweeps reuse identical data, and
    invariant to how many threads draw it. The exact operation order is
    pinned by golden hashes in test_synth.py: any change here severs
    comparability with every committed perf row and needs a DATA_RECIPE
    bump beside it.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (X_train, y_train, X_test, y_test); X float32 row-major in [0, 1),
        train is the first `rows` rows, test the next `n_test`.
    """
    n = rows + n_test
    # Stream 0 is the whole-matrix draw; the rest belong to one block each.
    streams = np.random.SeedSequence([seed, rows, cols]).spawn(N_BLOCKS + 1)
    k = min(informative, cols)
    idx = np.random.default_rng(streams[0]).choice(cols, size=k, replace=False)

    X = np.empty((n, cols), dtype=np.float32)
    y = np.empty(n, dtype=np.float32)
    # Unit normals now, scaled once below: sigma needs the whole of y, and
    # sigma * N(0, 1) is the N(0, sigma) the recipe asks for either way.
    noise = np.empty(n, dtype=np.float32)
    bounds = [(i * n // N_BLOCKS, (i + 1) * n // N_BLOCKS)
              for i in range(N_BLOCKS)]

    def draw(block: int):
        lo, hi = bounds[block]
        rng = np.random.default_rng(streams[block + 1])
        rng.random(out=X[lo:hi], dtype=np.float32)
        y[lo:hi] = _target(X[lo:hi], idx, k)
        rng.standard_normal(hi - lo, dtype=np.float32, out=noise[lo:hi])

    with ThreadPoolExecutor(max_workers=_workers()) as pool:
        list(pool.map(draw, range(N_BLOCKS)))
    y += (y.std() / 3.0) * noise
    return X[:rows], y[:rows], X[rows:], y[rows:]


# Private Functions ================================================================================

def _target(block: np.ndarray, idx: np.ndarray, k: int) -> np.ndarray:
    """The Friedman-1 sum over one block's rows; row-independent by design."""
    y = np.zeros(len(block), dtype=np.float32)
    for b in range(k // 5):
        f = block[:, idx[b * 5:(b + 1) * 5]]
        term = (10.0 * np.sin(np.pi * f[:, 0] * f[:, 1])
                + 20.0 * (f[:, 2] - 0.5) ** 2 + 10.0 * f[:, 3] + 5.0 * f[:, 4])
        y += (0.6 ** b) * term.astype(np.float32)
    return y


def _workers() -> int:
    """Threads to walk the blocks with, which never reaches the output.

    Sized from the cgroup ceiling rather than the CPU count: a pod
    advertising 256 CPUs may meter 27 of them (issue #355), and running 256
    threads against that quota buys throttling, not throughput.
    """
    cap = runlog.cpu_quota() or runlog.usable_cpus()
    return max(1, min(N_BLOCKS, int(cap)))
