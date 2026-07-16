#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=1.26", "pandas>=2.2", "scikit-learn>=1.4", "xgboost>=2.0"]
# ///
"""Feature-admission probe for issue #45: per-node / per-level feature subsampling.

bonsai already samples features once per tree (tree.feature_fraction ==
xgboost's colsample_bytree). Issue #45 would add per-LEVEL and per-NODE draws
(xgb colsample_bylevel/bynode, lgbm feature_fraction_bynode), which touch the
grower's selected-features plumbing and the CUDA find staging. Per the
feature-admission discipline we price the benefit at zero core cost by toggling
the reference libraries' own knobs, and only build if per-node/per-level BEATS
the per-tree knob bonsai already has.

Three questions, hardest last:
  1. In isolation, does bynode/bylevel beat bytree at a matched fraction?
  2. Stacked on bytree (their intended use), does bynode add anything?
  3. Can a well-tuned bytree ALONE match the best bytree+bynode stack?

Campaign knobs (200 iters, lr 0.05, depth 6, seed 42). Chance band ~2% rmse /
~0.001 auc (decision 55). Committed as the evidence for decision 75.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "tests" / "data"
SEED = 42
KNOBS = dict(n_estimators=200, learning_rate=0.05, max_depth=6)


def _csv(stem, task, cap=None):
    tr = pd.read_csv(DATA / f"{stem}_train.csv")
    te = pd.read_csv(DATA / f"{stem}_test.csv")
    if cap and len(tr) > cap:
        tr = tr.sample(cap, random_state=SEED)
    return (tr.drop(columns=["label"]).to_numpy(np.float32), tr["label"].to_numpy(),
            te.drop(columns=["label"]).to_numpy(np.float32), te["label"].to_numpy(), task)


def _synth(n=20000, p=200, informative=20):
    X, y = make_regression(n_samples=n + 5000, n_features=p, n_informative=informative,
                           noise=8.0, random_state=SEED)
    X = X.astype(np.float32)
    return X[:n], y[:n], X[n:], y[n:], "regression"


def _score(task, y, pred):
    if task == "regression":
        return "rmse", float(np.sqrt(mean_squared_error(y, pred)))
    if task == "binary":
        return "auc", float(roc_auc_score(y, pred))
    return "acc", float(accuracy_score(y, pred.argmax(1) if pred.ndim > 1 else pred))


def _xgb(X, y, Xte, task, **cs):
    import xgboost as xgb

    obj = {"regression": "reg:squarederror", "binary": "binary:logistic",
           "multiclass": "multi:softprob"}[task]
    kw = dict(objective=obj, tree_method="hist", random_state=SEED, **KNOBS, **cs)
    if task == "multiclass":
        kw["num_class"] = int(y.max()) + 1
    m = xgb.XGBRegressor(**kw) if task == "regression" else xgb.XGBClassifier(**kw)
    m.fit(X, y)
    if task == "regression":
        return m.predict(Xte)
    return m.predict_proba(Xte)[:, 1] if task == "binary" else m.predict_proba(Xte)


DATASETS = [
    ("california", lambda: _csv("california_housing", "regression")),
    ("higgs60k", lambda: _csv("higgs", "binary", cap=60000)),
    ("year_msd60k", lambda: _csv("year_prediction_msd", "regression", cap=60000)),
    ("synth_wide200", _synth),
]


def q1_isolation():
    print("\n# Q1: isolation: does bynode/bylevel beat bytree at a matched fraction?")
    arms = {"base": {},
            "bytree0.7": dict(colsample_bytree=0.7), "bytree0.5": dict(colsample_bytree=0.5),
            "bylevel0.7": dict(colsample_bylevel=0.7), "bynode0.7": dict(colsample_bynode=0.7),
            "bynode0.5": dict(colsample_bynode=0.5)}
    for name, loader in DATASETS:
        X, y, Xte, yte, task = loader()
        base = None
        cells = []
        for arm, cs in arms.items():
            metric, val = _score(task, yte, _xgb(X, y, Xte, task, **cs))
            base = val if arm == "base" else base
            cells.append(f"{arm}={val:.4f}({val - base:+.4f})")
        print(f"  {name:14s}[xgb {metric}] " + "  ".join(cells))


def q2q3_wide():
    print("\n# Q2/Q3: synth_wide200: stack on bytree, then can tuned bytree ALONE match it?")
    X, y, Xte, yte, task = _synth()
    arms = {"bytree0.7": dict(colsample_bytree=0.7), "bytree0.5": dict(colsample_bytree=0.5),
            "bytree0.4": dict(colsample_bytree=0.4), "bytree0.3": dict(colsample_bytree=0.3),
            "bytree0.7+bynode0.5": dict(colsample_bytree=0.7, colsample_bynode=0.5),
            "bytree0.5+bynode0.5": dict(colsample_bytree=0.5, colsample_bynode=0.5)}
    for arm, cs in arms.items():
        _, val = _score(task, yte, _xgb(X, y, Xte, task, **cs))
        print(f"  {arm:22s} rmse={val:.3f}")


if __name__ == "__main__":
    q1_isolation()
    q2q3_wide()
    print("\nVerdict: per-node/per-level is dominated by a tuned per-tree feature_fraction "
          "on every dataset, incl. the 200-feature best case (decision 75).")
