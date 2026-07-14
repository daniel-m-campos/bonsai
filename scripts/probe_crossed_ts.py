# PROVENANCE NOTE (decision 69): this probe is a completed experiment whose
# committed evidence was produced by this exact code; it stays as-run rather
# than being refactored onto bonsai.bench. NEW probes must import their knobs
# and reference-library mappings from bonsai.bench.params and their metrics
# from bonsai.bench.metrics.
# Crossed-TS probe (crown week): can crossed-category ordered TS close
# catboost's remaining amazon edge from preprocessing alone?
# Uses the repo's pinned amazon split (tests/data) so numbers compose with
# the decision-58 table (ordinal 0.8109, singles-TS 0.8603 there).
import itertools
import sys
import time

import numpy as np

sys.path.insert(0, "build/python")
import bonsai


def load(p):
    d = np.loadtxt(p, delimiter=",", skiprows=1, dtype=np.float32)
    return d[:, 1:], d[:, 0]


Xtr, ytr = load("tests/data/amazon_train.csv")
Xte, yte = load("tests/data/amazon_test.csv")
n_cols = Xtr.shape[1]


def auc(y, s):
    order = np.argsort(s)
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    pos = y > 0.5
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return (r[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def fit_auc(Xa, Xb, label):
    t0 = time.time()
    m = bonsai.BonsaiRegressor(
        n_iters=200, learning_rate=0.05, objective="logloss", max_depth=6,
        grower="depthwise", random_seed=42,
        params={"tree.min_data_in_leaf": 20, "tree.lambda_l2": 1.0,
                "bin_mapper.max_bin": 255})
    m.fit(Xa, ytr)
    a = auc(yte, m.predict(Xb))
    print(f"{label:40s} auc {a:.4f}  ({Xa.shape[1]} cols, {time.time()-t0:.1f}s)",
          flush=True)
    return a


def crossed_cols(idx_tuples):
    """Dense pair/triple codes, factorized jointly over train+test keys
    (identity only, no labels) so float32 stays integer-exact."""
    tr_cols, te_cols = [], []
    for idx in idx_tuples:
        key_tr = np.zeros(len(Xtr), dtype=np.int64)
        key_te = np.zeros(len(Xte), dtype=np.int64)
        for c in idx:
            key_tr = key_tr * 100_000 + Xtr[:, c].astype(np.int64)
            key_te = key_te * 100_000 + Xte[:, c].astype(np.int64)
        vals = np.unique(np.concatenate([key_tr, key_te]))
        tr_cols.append(np.searchsorted(vals, key_tr).astype(np.float32))
        te_cols.append(np.searchsorted(vals, key_te).astype(np.float32))
    return np.column_stack(tr_cols), np.column_stack(te_cols)


# Baseline: singles TS + codes (the decision-58 configuration).
enc = bonsai.OrderedTargetEncoder(columns=range(n_cols))
b_tr, b_te = enc.fit_transform(Xtr, ytr), enc.transform(Xte)
fit_auc(b_tr, b_te, "singles TS + codes (baseline)")

# Pairs: TS over all 36 crossed pair codes appended.
pairs = list(itertools.combinations(range(n_cols), 2))
ptr, pte = crossed_cols(pairs)
enc2 = bonsai.OrderedTargetEncoder(columns=range(len(pairs)), keep_codes=False)
p_tr, p_te = enc2.fit_transform(ptr, ytr), enc2.transform(pte)
fit_auc(np.column_stack([b_tr, p_tr]), np.column_stack([b_te, p_te]),
        "singles + pair-TS (36)")

# Triples on top.
triples = list(itertools.combinations(range(n_cols), 3))
ttr, tte = crossed_cols(triples)
enc3 = bonsai.OrderedTargetEncoder(columns=range(len(triples)), keep_codes=False)
t_tr, t_te = enc3.fit_transform(ttr, ytr), enc3.transform(tte)
fit_auc(np.column_stack([b_tr, p_tr, t_tr]),
        np.column_stack([b_te, p_te, t_te]),
        "singles + pairs + triple-TS (84)")

# Reference: catboost native on this split, default and pairs-only ctr depth.
from catboost import CatBoostClassifier  # noqa: E402

for label, extra in (("catboost native (default ctr)", {}),
                     ("catboost native (max_ctr_complexity=1)",
                      {"max_ctr_complexity": 1})):
    t0 = time.time()
    Xtr_s = Xtr.astype(np.int64).astype(str)
    Xte_s = Xte.astype(np.int64).astype(str)
    m = CatBoostClassifier(iterations=200, learning_rate=0.05, depth=6,
                           l2_leaf_reg=1.0, border_count=255, random_seed=42,
                           verbose=False, allow_writing_files=False,
                           cat_features=list(range(n_cols)), **extra)
    m.fit(Xtr_s, ytr)
    a = auc(yte, m.predict_proba(Xte_s)[:, 1])
    print(f"{label:40s} auc {a:.4f}  ({time.time()-t0:.1f}s)", flush=True)

print("CROSSED-TS-PROBE-DONE")
