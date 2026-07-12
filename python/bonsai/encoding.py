"""Ordered target-statistics encoding — bonsai's categorical front door.

The core is numeric on purpose (decision 58): categorical columns enter
through encoding, and the *ordered* (causal) encoding is the one that does
not leak the label. Guide chapter 13 derives the math and the evidence;
`fit_transform`/`transform` are asymmetric for the same reason sklearn's
TargetEncoder's are — the training encoding must not see a row's own label.
"""

from __future__ import annotations

import numpy as np

__all__ = ["OrderedTargetEncoder"]


def _bucket_missing(col: np.ndarray) -> np.ndarray:
    # NaN != NaN would make every missing row its own category; +inf is a
    # single shared bucket that still sorts (and searchsorts) last.
    return np.where(np.isnan(col), np.float32(np.inf), col)


class OrderedTargetEncoder:
    """Replace categorical code columns with causal running target means.

    Categories must arrive as numeric codes in float cells (``.cat.codes``
    for pandas users); NaN is treated as one shared "missing" category.
    ``fit_transform`` encodes the training set causally — each row sees the
    label mean of *earlier* rows only, under a seeded permutation — while
    ``transform`` applies full-training-set statistics, matching what a
    deployed model sees. ``keep_codes`` appends the raw code columns after
    the features, giving trees both views (measured worth +0.005 AUC on the
    amazon benchmark). Binary {0,1} or regression targets.
    """

    def __init__(self, columns, prior_weight: float = 10.0,
                 n_permutations: int = 1, keep_codes: bool = True,
                 seed: int = 0):
        self.columns = list(columns)
        self.prior_weight = float(prior_weight)
        self.n_permutations = int(n_permutations)
        self.keep_codes = bool(keep_codes)
        self.seed = int(seed)
        self._stats: dict[int, tuple[np.ndarray, np.ndarray]] | None = None
        self._prior: float = 0.0

    def fit_transform(self, X, y) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float64).ravel()
        out = np.array(X, copy=True)
        self._prior = float(y.mean())
        self._stats = {}
        appended = []
        for c in self.columns:
            col = _bucket_missing(X[:, c])
            enc = np.zeros(len(y), dtype=np.float64)
            for p in range(self.n_permutations):
                enc += self._causal_means(col, y, self.seed + p)
            out[:, c] = (enc / self.n_permutations).astype(np.float32)
            self._stats[c] = self._full_stats(col, y)
            if self.keep_codes:
                appended.append(X[:, c])
        return np.column_stack([out, *appended]) if appended else out

    def transform(self, X) -> np.ndarray:
        if self._stats is None:
            raise RuntimeError("fit_transform must run before transform")
        X = np.asarray(X, dtype=np.float32)
        out = np.array(X, copy=True)
        appended = []
        for c in self.columns:
            col = _bucket_missing(X[:, c])
            vals, means = self._stats[c]
            idx = np.clip(np.searchsorted(vals, col), 0, len(vals) - 1)
            seen = vals[idx] == col
            # Unseen categories have zero evidence: the smoothed mean is
            # exactly the prior.
            out[:, c] = np.where(seen, means[idx], np.float32(self._prior))
            if self.keep_codes:
                appended.append(X[:, c])
        return np.column_stack([out, *appended]) if appended else out

    def _causal_means(self, col, y, seed) -> np.ndarray:
        n = len(y)
        pos = np.empty(n, dtype=np.int64)
        pos[np.random.default_rng(seed).permutation(n)] = np.arange(n)
        # Group-major, visit-order-minor sort turns "label sum of earlier
        # rows in my category" into a segmented shifted cumsum.
        order = np.lexsort((pos, col))
        y_s, col_s = y[order], col[order]
        cum = np.cumsum(y_s) - y_s
        starts = np.maximum.accumulate(
            np.where(np.r_[True, col_s[1:] != col_s[:-1]], np.arange(n), 0))
        prev_sum = cum - cum[starts]
        prev_cnt = np.arange(n) - starts
        enc_s = ((prev_sum + self.prior_weight * self._prior) /
                 (prev_cnt + self.prior_weight))
        enc = np.empty(n, dtype=np.float64)
        enc[order] = enc_s
        return enc

    def _full_stats(self, col, y) -> tuple[np.ndarray, np.ndarray]:
        vals, inverse, counts = np.unique(col, return_inverse=True,
                                          return_counts=True)
        sums = np.bincount(inverse, weights=y)
        means = ((sums + self.prior_weight * self._prior) /
                 (counts + self.prior_weight))
        return vals, means.astype(np.float32)
