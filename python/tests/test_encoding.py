"""OrderedTargetEncoder tests: run with  PYTHONPATH=build/python
.venv/bin/python python/tests/test_encoding.py  (pytest also works)."""

from __future__ import annotations

import pathlib

import numpy as np

import bonsai

REPO = pathlib.Path(__file__).resolve().parents[2]


def _toy():
    rng = np.random.default_rng(3)
    codes = rng.integers(0, 20, 2000).astype(np.float32)
    codes[::13] = np.nan
    y = (rng.random(2000) < 0.25 + 0.02 * np.nan_to_num(codes)).astype(np.float64)
    return codes.reshape(-1, 1), y


def test_deterministic_and_shapes():
    X, y = _toy()
    a = bonsai.OrderedTargetEncoder(columns=[0], seed=5).fit_transform(X, y)
    b = bonsai.OrderedTargetEncoder(columns=[0], seed=5).fit_transform(X, y)
    assert np.array_equal(a, b, equal_nan=True)
    # keep_codes appends one column per encoded column: the untouched
    # original codes (NaN preserved — bonsai routes missing natively).
    assert a.shape == (X.shape[0], X.shape[1] + 1)
    assert np.array_equal(a[:, 1], X[:, 0], equal_nan=True)
    slim = bonsai.OrderedTargetEncoder(columns=[0], keep_codes=False)
    assert slim.fit_transform(X, y).shape == X.shape


def test_first_visit_gets_the_prior():
    X, y = _toy()
    enc = bonsai.OrderedTargetEncoder(columns=[0], prior_weight=7.0, seed=1,
                                      keep_codes=False)
    out = enc.fit_transform(X, y)[:, 0]
    prior = y.mean()
    # For every category, the first-visited row has zero evidence: its
    # encoding is exactly the smoothed prior 7*p/7 = p.
    order = np.random.default_rng(1).permutation(len(y))
    seen: set[float] = set()
    firsts = []
    col = np.where(np.isnan(X[:, 0]), np.inf, X[:, 0])
    for i in order:
        if col[i] not in seen:
            firsts.append(i)
            seen.add(col[i])
    assert np.allclose(out[firsts], prior, atol=1e-6)


def test_causality_no_own_label_leak():
    X, y = _toy()
    enc = bonsai.OrderedTargetEncoder(columns=[0], seed=2, keep_codes=False)
    base = enc.fit_transform(X, y)[:, 0]
    # Flipping row j's label must not move row j's own encoding — a row
    # never sees its own target. (It legitimately moves later rows.)
    j = 137
    y2 = y.copy()
    y2[j] = 1.0 - y2[j]
    flipped = bonsai.OrderedTargetEncoder(columns=[0], seed=2,
                                          keep_codes=False).fit_transform(X, y2)
    # prior shifts by 1/n; remove that global effect by comparing against a
    # tolerance far below the smallest same-category step.
    assert abs(flipped[j, 0] - base[j]) < 2.0 / len(y)


def test_transform_full_stats_and_unseen():
    X, y = _toy()
    enc = bonsai.OrderedTargetEncoder(columns=[0], prior_weight=10.0,
                                      keep_codes=False)
    enc.fit_transform(X, y)
    col = np.where(np.isnan(X[:, 0]), np.inf, X[:, 0])
    c = col[0]
    mask = col == c
    expect = (y[mask].sum() + 10.0 * y.mean()) / (mask.sum() + 10.0)
    got = enc.transform(np.array([[c]], dtype=np.float32))[0, 0]
    assert np.isclose(got, expect, atol=1e-6)
    # Unseen category and NaN both resolve without error; unseen == prior.
    t = enc.transform(np.array([[1e9], [np.nan]], dtype=np.float32))
    assert np.isclose(t[0, 0], y.mean(), atol=1e-5)
    assert np.isfinite(t[1, 0])


def test_amazon_quality_pin():
    # The measured reason this module exists (decision 58): +0.03 AUC or
    # better over ordinal codes on the amazon access data, which clears the
    # lightgbm-native-set-splits line from feature_gap.md §18.
    def load(p):
        d = np.loadtxt(p, delimiter=",", skiprows=1, dtype=np.float32)
        return d[:, 1:], d[:, 0]

    Xtr, ytr = load(REPO / "tests/data/amazon_train.csv")
    Xte, yte = load(REPO / "tests/data/amazon_test.csv")

    def auc(y, s):
        order = np.argsort(s)
        r = np.empty(len(s))
        r[order] = np.arange(1, len(s) + 1)
        pos = y > 0.5
        n_pos, n_neg = pos.sum(), (~pos).sum()
        return (r[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    def fit_auc(Xa, Xb):
        m = bonsai.BonsaiRegressor(
            n_iters=200, learning_rate=0.05, objective="logloss", max_depth=6,
            grower="depthwise", random_seed=42,
            params={"tree.min_data_in_leaf": 20, "tree.lambda_l2": 1.0,
                    "bin_mapper.max_bin": 255})
        m.fit(Xa, ytr)
        return auc(yte, m.predict(Xb))

    ordinal = fit_auc(Xtr, Xte)
    enc = bonsai.OrderedTargetEncoder(columns=range(Xtr.shape[1]))
    encoded = fit_auc(enc.fit_transform(Xtr, ytr), enc.transform(Xte))
    assert encoded - ordinal > 0.03, (ordinal, encoded)
    assert encoded > 0.85, encoded


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("all encoding tests passed")
