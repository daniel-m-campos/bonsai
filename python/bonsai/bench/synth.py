"""The synthetic dataset behind bonsai's perf-division benchmarks.

Provenance: a generalized form of Friedman #1 (Friedman, "Multivariate
Adaptive Regression Splines", Annals of Statistics 1991), the standard
synthetic regression for tree methods. The classic five-feature target

    10 sin(pi x0 x1) + 20 (x2 - 0.5)^2 + 10 x3 + 5 x4

is repeated over `informative` features in blocks of five with geometrically
decaying block weights (0.6^b), over uniform [0, 1) float32 features, plus
Gaussian noise with sigma = y.std() / 3, sizing the best-achievable test R^2
at roughly 0.9. Why synthetic for perf: rows, cols, and bins become free
axes (decision 46), and generation is deterministic in (seed, rows, cols)
via SeedSequence([seed, rows, cols]) only, so bins/threads sweeps reuse
byte-identical data.

Note: scripts/model_hash.py contains a deliberately FROZEN linear variant of
this recipe. Its output feeds the cross-arch byte-identity CI gate; it must
never be edited or replaced with this function.
"""

from __future__ import annotations

import numpy as np


def gen_data(rows: int, cols: int, seed: int, n_test: int, informative: int):
    """Generalized Friedman-1: k informative features in blocks of 5, decaying
    block weights, noise sized for a best-achievable R^2 of ~0.9. Deterministic
    in (rows, cols, seed) only, so bins/threads sweeps reuse identical data."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, rows, cols]))
    n = rows + n_test
    X = rng.random((n, cols), dtype=np.float32)
    k = min(informative, cols)
    idx = rng.choice(cols, size=k, replace=False)
    y = np.zeros(n, dtype=np.float32)
    for b in range(k // 5):
        f = X[:, idx[b * 5:(b + 1) * 5]]
        term = (10.0 * np.sin(np.pi * f[:, 0] * f[:, 1])
                + 20.0 * (f[:, 2] - 0.5) ** 2 + 10.0 * f[:, 3] + 5.0 * f[:, 4])
        y += (0.6 ** b) * term.astype(np.float32)
    y += rng.normal(0.0, y.std() / 3.0, size=n).astype(np.float32)
    return X[:rows], y[:rows], X[rows:], y[rows:]
