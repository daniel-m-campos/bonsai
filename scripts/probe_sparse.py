# PROVENANCE NOTE (decision 69): this probe is a completed experiment whose
# committed evidence was produced by this exact code; it stays as-run rather
# than being refactored onto bonsai.bench. NEW probes must import their knobs
# and reference-library mappings from bonsai.bench.params and their metrics
# from bonsai.bench.metrics.
# Sparse/EFB probe (crown week, feature-admission step 1-2): is sparse
# input the remaining forcing function, and does preprocessing rescue it?
#   Q1 width ladder on real text (rcv1): bonsai-dense vs lgbm/xgb sparse —
#      quality + fit time as columns grow.
#   Q2 lgbm sparse vs lgbm densified — how much sparsity handling itself buys.
#   Q3 EFB-as-preprocessing: how many conflict-free bundles exist in text
#      (tf-idf) vs one-hot data; the one-hot control shows bundling
#      collapsing to "use codes, not one-hot" — which bonsai already does.
import sys
import time

import numpy as np
from sklearn.datasets import fetch_rcv1
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "build/python")
import bonsai

N_ROWS = 40_000
HP = dict(n_estimators=200, learning_rate=0.05, max_depth=6,
          random_state=42)


def bonsai_fit_auc(Xtr_d, ytr, Xte_d, yte):
    t0 = time.time()
    m = bonsai.BonsaiRegressor(
        n_iters=200, learning_rate=0.05, objective="logloss", max_depth=6,
        grower="depthwise", random_seed=42,
        params={"tree.min_data_in_leaf": 20, "tree.lambda_l2": 1.0,
                "bin_mapper.max_bin": 255})
    m.fit(Xtr_d, ytr)
    fit_s = time.time() - t0
    return roc_auc_score(yte, m.predict(Xte_d)), fit_s


def greedy_bundles(X_csc, max_conflict=0):
    """EFB core: greedily pack columns whose nonzero row sets don't overlap
    (conflict budget 0 = exact). Returns number of bundles."""
    bundles = 0
    order = np.argsort(-np.diff(X_csc.indptr))  # densest first
    open_masks = []
    for c in order:
        rows = X_csc.indices[X_csc.indptr[c]:X_csc.indptr[c + 1]]
        placed = False
        for mask in open_masks:
            if not mask[rows].any():
                mask[rows] = True
                placed = True
                break
        if not placed:
            mask = np.zeros(X_csc.shape[0], dtype=bool)
            mask[rows] = True
            open_masks.append(mask)
            bundles += 1
    return bundles


print("fetching rcv1...", flush=True)
rcv = fetch_rcv1()
y_all = np.asarray(rcv.target[:, rcv.target_names.tolist().index("CCAT")]
                   .todense()).ravel().astype(np.float64)
X_all = rcv.data[:N_ROWS].tocsc()
y = y_all[:N_ROWS]
cut = int(N_ROWS * 0.8)
print(f"rcv1 subset: {N_ROWS} rows, {X_all.shape[1]} cols, "
      f"pos rate {y.mean():.3f}", flush=True)

import lightgbm as lgb  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

for k in (1000, 5000, 10000):
    df = np.diff(X_all.indptr)
    top = np.sort(np.argsort(-df)[:k])
    Xk = X_all[:, top]
    Xtr, Xte = Xk[:cut], Xk[cut:]
    ytr, yte = y[:cut], y[cut:]
    r = {}

    t0 = time.time()
    m = lgb.LGBMClassifier(**HP, num_leaves=63, min_child_samples=20,
                           reg_lambda=1.0, max_bin=255, verbose=-1)
    m.fit(Xtr, ytr)
    r["lgbm_sparse"] = (roc_auc_score(yte, m.predict_proba(Xte)[:, 1]),
                        time.time() - t0)

    t0 = time.time()
    m = XGBClassifier(**HP, min_child_weight=20, reg_lambda=1.0,
                      max_bin=255, tree_method="hist")
    m.fit(Xtr, ytr)
    r["xgb_sparse"] = (roc_auc_score(yte, m.predict_proba(Xte)[:, 1]),
                       time.time() - t0)

    Xtr_d = np.asarray(Xtr.todense(), dtype=np.float32)
    Xte_d = np.asarray(Xte.todense(), dtype=np.float32)
    t0 = time.time()
    m = lgb.LGBMClassifier(**HP, num_leaves=63, min_child_samples=20,
                           reg_lambda=1.0, max_bin=255, verbose=-1)
    m.fit(Xtr_d, ytr)
    r["lgbm_dense"] = (roc_auc_score(yte, m.predict_proba(Xte_d)[:, 1]),
                       time.time() - t0)

    auc, fs = bonsai_fit_auc(Xtr_d, ytr, Xte_d, yte)
    r["bonsai_dense"] = (auc, fs)
    del Xtr_d, Xte_d

    nb = greedy_bundles(Xk[:cut].tocsc()) if k <= 5000 else -1
    for name, (a, s) in r.items():
        print(f"k={k:6d} {name:14s} auc {a:.4f}  fit {s:6.1f}s", flush=True)
    print(f"k={k:6d} exact EFB bundles: {nb} "
          f"({'skipped' if nb < 0 else f'{k / max(nb, 1):.2f}x reduction'})",
          flush=True)

# One-hot control: EFB's home turf collapses to "use codes".
rng = np.random.default_rng(11)
n, n_cat, card = 200_000, 20, 50
codes = rng.integers(0, card, (n, n_cat)).astype(np.float32)
logit = ((codes[:, 0] % 7) - 3) * 0.5 + ((codes[:, 1] % 5) - 2) * 0.4 \
    + rng.normal(0, 1.0, n)
yc = (logit > 0).astype(np.float64)
from scipy import sparse  # noqa: E402

onehot = sparse.hstack(
    [sparse.csr_matrix(
        (np.ones(n, dtype=np.float32), (np.arange(n), codes[:, j])),
        shape=(n, card)) for j in range(n_cat)]).tocsr()
cutc = int(n * 0.8)
t0 = time.time()
m = lgb.LGBMClassifier(**HP, num_leaves=63, min_child_samples=20,
                       reg_lambda=1.0, max_bin=255, verbose=-1)
m.fit(onehot[:cutc], yc[:cutc])
print(f"onehot lgbm_sparse_efb  auc "
      f"{roc_auc_score(yc[cutc:], m.predict_proba(onehot[cutc:])[:, 1]):.4f}"
      f"  fit {time.time() - t0:6.1f}s ({onehot.shape[1]} cols)", flush=True)
auc, fs = bonsai_fit_auc(codes[:cutc], yc[:cutc], codes[cutc:], yc[cutc:])
print(f"onehot bonsai_codes     auc {auc:.4f}  fit {fs:6.1f}s "
      f"({n_cat} cols)", flush=True)
print("SPARSE-PROBE-DONE")
