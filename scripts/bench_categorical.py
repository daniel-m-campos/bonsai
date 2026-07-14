#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=1.26",
#     "pandas>=2.2",
#     "scikit-learn>=1.4",
#     "xgboost>=2.0",
#     "lightgbm>=4.3",
#     "catboost>=1.2",
# ]
# ///
"""Stage-1 categorical measurement on Amazon employee access.

The decision question: what would NATIVE categorical splits buy bonsai?
The cleanest isolation is within one library — lightgbm with
categorical_feature declared vs the same integer IDs treated as numeric.
Around that: bonsai's two practical options today (raw IDs as numeric;
K-fold target encoding as preprocessing) and every reference library's
native mode.

Run (needs the native module):
    PYTHONPATH=build/python .venv/bin/python scripts/bench_categorical.py
"""

import pathlib
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "build" / "python"))

# STUDY_KNOBS: this stage-1 study deliberately deviates from the campaign
# regime (bonsai.bench.params.CAMPAIGN: 200 iters, lr 0.05, depth 6). The
# committed evidence (categorical-tradeoff-2026-07.md, decision 58) was
# adjudicated at THESE knobs; per decision 69 they stay annotated rather
# than rerun, which would reopen a settled decision for zero information.
N_ITERS = 300
LR = 0.1
SEED = 42
COMMON = dict(max_depth=8, max_leaves=63)


def auc(pred, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, pred))


def as_category(Xtr, Xte):
    """Shared per-column category dictionary (train ∪ test) so xgboost's
    predict accepts test-only IDs; unseen categories still carry no signal."""
    out_tr, out_te = Xtr.copy(), Xte.copy()
    for col in Xtr.columns:
        dtype = pd.CategoricalDtype(
            sorted(set(Xtr[col]) | set(Xte[col])))
        out_tr[col] = out_tr[col].astype(dtype)
        out_te[col] = out_te[col].astype(dtype)
    return out_tr, out_te


def target_encode(train_X, train_y, test_X, n_folds=5, smooth=20.0):
    """K-fold out-of-fold target encoding (leakage-safe), the standard
    preprocessing for numeric-only GBTs on high-cardinality categoricals."""
    from sklearn.model_selection import KFold
    prior = train_y.mean()
    enc_train = pd.DataFrame(index=train_X.index)
    enc_test = pd.DataFrame(index=test_X.index)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    for col in train_X.columns:
        oof = np.full(len(train_X), prior)
        for tr_idx, va_idx in kf.split(train_X):
            grp = train_y.iloc[tr_idx].groupby(
                train_X[col].iloc[tr_idx]).agg(["mean", "count"])
            enc = ((grp["mean"] * grp["count"] + prior * smooth)
                   / (grp["count"] + smooth))
            oof[va_idx] = (train_X[col].iloc[va_idx].map(enc)
                           .fillna(prior).to_numpy())
        enc_train[col] = oof
        grp = train_y.groupby(train_X[col]).agg(["mean", "count"])
        enc = ((grp["mean"] * grp["count"] + prior * smooth)
               / (grp["count"] + smooth))
        enc_test[col] = test_X[col].map(enc).fillna(prior).to_numpy()
    return enc_train, enc_test


def run_bonsai(Xtr, ytr, Xte, label):
    import bonsai
    t0 = time.perf_counter()
    m = bonsai.BonsaiRegressor(
        n_iters=N_ITERS, learning_rate=LR, objective="logloss",
        grower="leafwise", **COMMON,
        params={"tree.min_data_in_leaf": 20},
    ).fit(np.ascontiguousarray(Xtr, dtype=np.float32),
          np.ascontiguousarray(ytr, dtype=np.float32))
    fit_s = time.perf_counter() - t0
    pred = m.predict(np.ascontiguousarray(Xte, dtype=np.float32))
    return label, fit_s, pred


def run_lightgbm(Xtr, ytr, Xte, label, categorical=False):
    import lightgbm as lgb
    params = {"objective": "binary", "learning_rate": LR,
              "max_depth": COMMON["max_depth"],
              "num_leaves": COMMON["max_leaves"], "verbose": -1,
              "seed": SEED}
    cat = list(Xtr.columns) if categorical else "auto"
    if categorical:
        Xtr, Xte = as_category(Xtr, Xte)
    t0 = time.perf_counter()
    booster = lgb.train(params, lgb.Dataset(Xtr, label=ytr,
                                            categorical_feature=cat),
                        num_boost_round=N_ITERS)
    fit_s = time.perf_counter() - t0
    return label, fit_s, booster.predict(Xte)


def run_xgboost(Xtr, ytr, Xte, label, categorical=False):
    import xgboost as xgb
    if categorical:
        Xtr, Xte = as_category(Xtr, Xte)
    params = {"objective": "binary:logistic", "learning_rate": LR,
              "max_depth": COMMON["max_depth"],
              "max_leaves": COMMON["max_leaves"], "tree_method": "hist",
              "seed": SEED}
    dtr = xgb.DMatrix(Xtr, label=ytr, enable_categorical=categorical)
    dte = xgb.DMatrix(Xte, enable_categorical=categorical)
    t0 = time.perf_counter()
    booster = xgb.train(params, dtr, num_boost_round=N_ITERS)
    fit_s = time.perf_counter() - t0
    return label, fit_s, booster.predict(dte)


def run_catboost(Xtr, ytr, Xte, label):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(iterations=N_ITERS, learning_rate=LR,
                           depth=COMMON["max_depth"], random_seed=SEED,
                           verbose=False,
                           cat_features=list(range(Xtr.shape[1])))
    t0 = time.perf_counter()
    m.fit(Xtr.astype(str), ytr)
    fit_s = time.perf_counter() - t0
    return label, fit_s, m.predict_proba(Xte.astype(str))[:, 1]


def main() -> int:
    train = pd.read_csv(REPO_ROOT / "tests/data/amazon_train.csv")
    test = pd.read_csv(REPO_ROOT / "tests/data/amazon_test.csv")
    ytr, yte = train.pop("label"), test.pop("label")

    enc_tr, enc_te = target_encode(train, ytr, test)

    rows = []
    for label, fit_s, pred in [
        run_bonsai(train, ytr, test, "bonsai (raw IDs as numeric)"),
        run_bonsai(enc_tr, ytr, enc_te, "bonsai (K-fold target encoding)"),
        run_lightgbm(train, ytr, test, "lightgbm (raw IDs as numeric)"),
        run_lightgbm(train, ytr, test, "lightgbm (native categorical)",
                     categorical=True),
        run_xgboost(train, ytr, test, "xgboost (native categorical)",
                    categorical=True),
        run_catboost(train, ytr, test, "catboost (native, ordered TS)"),
    ]:
        rows.append((label, auc(pred, yte), fit_s))
        print(f"{label:<36} auc={rows[-1][1]:.4f}  fit={fit_s:.2f}s",
              flush=True)

    out = REPO_ROOT / "benchmarks/results/amazon_cat.md"
    width = max(len(r[0]) for r in rows)
    lines = ["# amazon_cat (stage-1 categorical measurement)", "",
             f"| {'setup':<{width}} | auc    | fit_s |",
             f"|{'-' * (width + 2)}|--------|-------|"]
    for label, a, f in rows:
        lines.append(f"| {label:<{width}} | {a:.4f} | {f:5.1f} |")
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
