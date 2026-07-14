# PROVENANCE NOTE (decision 69): this probe is a completed experiment whose
# committed evidence was produced by this exact code; it stays as-run rather
# than being refactored onto bonsai.bench. NEW probes must import their knobs
# and reference-library mappings from bonsai.bench.params and their metrics
# from bonsai.bench.metrics.
# Per-feature bin budget probe (issue #63 residual; decision 55 hypothesis):
# measure the BENEFIT side of per-feature binning with zero bonsai C++ changes.
#   1. Reference toggle: lightgbm's own max_bin_by_feature ON vs OFF at
#      campaign-matched knobs prices the feature in the library that ships it.
#   2. Zero-core-cost emulation: pre-discretize each feature to its budgeted
#      bin count and feed stock bonsai; the distinct-value cut rule (issue
#      #61) reproduces the budgets exactly. bin_mapper.n_samples is set to
#      n_rows so sampling cannot miss a rare bin (deviation from the campaign
#      default, applied to every bonsai variant for apples-to-apples).
# Budget policies share the SAME total budget as uniform-255 unless named
# "headroom" (which raises the cap on important features instead).
# Output: JSON of metric per dataset x variant.
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "build/python"))
import bonsai

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/binning_probe.json")
HP = dict(n_iters=200, learning_rate=0.05, max_depth=6, max_leaves=63,
          random_seed=42)
BONSAI_PARAMS = {"tree.min_data_in_leaf": 20, "tree.lambda_l2": 1.0,
                 "bin_mapper.max_bin": 255}
LGB_BASE = dict(num_iterations=200, learning_rate=0.05, max_depth=6,
                num_leaves=63, min_data_in_leaf=20, lambda_l2=1.0, max_bin=255,
                verbose=-1, seed=42, deterministic=True, num_threads=8)
FLOOR, CAP = 8, 1023


def synth(n=250_000, f=100, informative=20, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.random((n, f), dtype=np.float32)
    y = np.zeros(n, dtype=np.float64)
    for i in range(informative):
        y += np.sin(3 * np.pi * X[:, i]) / (1 + 0.2 * i)
    y += rng.normal(0, 0.25, n)
    half = int(n * 0.9)
    return (X[:half], y[:half].astype(np.float32), X[half:],
            y[half:].astype(np.float32), "r2")


def load_local(train_csv, test_csv):
    tr = np.loadtxt(REPO / train_csv, delimiter=",", skiprows=1, dtype=np.float32)
    te = np.loadtxt(REPO / test_csv, delimiter=",", skiprows=1, dtype=np.float32)
    return tr[:, 1:], tr[:, 0], te[:, 1:], te[:, 0], "r2"


def load_openml(oml_id, seed=42):
    d = fetch_openml(data_id=oml_id, as_frame=True, parser="auto")
    Xf, y = d.data, d.target
    y = (y == y.unique()[0]).to_numpy(np.float32)
    for c in Xf.columns:
        if not pd.api.types.is_numeric_dtype(Xf[c]):
            Xf[c] = Xf[c].astype("category").cat.codes
    X = Xf.to_numpy(np.float32)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    half = int(len(X) * 0.8)
    tr, te = idx[:half], idx[half:]
    return X[tr], y[tr], X[te], y[te], "auc"


def metric(kind, y_true, pred):
    if kind == "auc":
        return float(roc_auc_score(y_true, pred))
    ss = float(((y_true - pred) ** 2).sum())
    tot = float(((y_true - y_true.mean()) ** 2).sum())
    return 1.0 - ss / tot


def bonsai_fit_predict(Xtr, ytr, Xte, kind, n_samples):
    params = dict(BONSAI_PARAMS)
    params["bin_mapper.n_samples"] = n_samples
    obj = "logloss" if kind == "auc" else "mse"
    m = bonsai.BonsaiRegressor(objective=obj, grower="depthwise", **HP,
                               params=params).fit(Xtr, ytr)
    return np.asarray(m.predict(Xte)), m


def importance_budgets(Xtr, ytr, kind, mode):
    """Gain-importance-driven per-feature budgets from a quick 50-iter fit."""
    obj = "logloss" if kind == "auc" else "mse"
    probe = bonsai.BonsaiRegressor(objective=obj, grower="depthwise", n_iters=50,
                                   learning_rate=0.1, max_depth=6,
                                   random_seed=42).fit(Xtr, ytr)
    g = np.sqrt(probe.importance("gain") + 1e-12)
    f = len(g)
    if mode == "headroom":
        k = np.full(f, 255)
        k[np.argsort(g)[-10:]] = CAP
        return k
    if mode == "inverse":
        g = 1.0 / g
    share = g / g.sum()
    k = np.clip(np.round(share * 255 * f), FLOOR, CAP).astype(int)
    return k


class ManualBinner:
    """Quantile edges at per-feature budgets; transform emits bin ids as
    floats, so stock bonsai's one-cut-per-distinct rule reproduces the
    budgets exactly."""

    def fit(self, X, budgets):
        self.edges = [
            np.unique(np.quantile(X[:, j], np.linspace(0, 1, k + 1)[1:-1]))
            for j, k in enumerate(budgets)
        ]
        return self

    def transform(self, X):
        out = np.empty_like(X)
        for j, e in enumerate(self.edges):
            out[:, j] = np.searchsorted(e, X[:, j], side="right")
        return out.astype(np.float32)


def run_dataset(name, loader, results):
    Xtr, ytr, Xte, yte, kind = loader()
    n = len(Xtr)
    res = {}

    pred, _ = bonsai_fit_predict(Xtr, ytr, Xte, kind, n)
    res["bonsai_uniform255"] = metric(kind, yte, pred)

    for mode in ("importance", "inverse", "headroom"):
        k = importance_budgets(Xtr, ytr, kind, mode)
        mb = ManualBinner().fit(Xtr, k)
        pred, _ = bonsai_fit_predict(mb.transform(Xtr), ytr, mb.transform(Xte),
                                     kind, n)
        res[f"bonsai_{mode}"] = metric(kind, yte, pred)
        res[f"budgets_{mode}"] = dict(min=int(k.min()), max=int(k.max()),
                                      total=int(k.sum()))

    import lightgbm as lgb
    obj = "binary" if kind == "auc" else "regression"
    ds = lgb.Dataset(Xtr, label=ytr)
    m = lgb.train({**LGB_BASE, "objective": obj}, ds)
    res["lgbm_uniform255"] = metric(kind, yte, m.predict(Xte))

    k = importance_budgets(Xtr, ytr, kind, "importance")
    # lightgbm caps per-feature bins at 255 internally unless max_bin rises too
    lgb_params = {**LGB_BASE, "objective": obj, "max_bin": int(k.max()),
                  "max_bin_by_feature": [int(v) for v in k]}
    ds = lgb.Dataset(Xtr, label=ytr, params=lgb_params)
    m = lgb.train(lgb_params, ds)
    res["lgbm_importance"] = metric(kind, yte, m.predict(Xte))

    import xgboost as xgb
    xm = xgb.XGBRegressor if kind == "r2" else xgb.XGBClassifier
    m = xm(n_estimators=200, learning_rate=0.05, max_depth=6, max_bin=255,
           tree_method="hist", reg_lambda=1.0, random_state=42, n_jobs=8).fit(
        Xtr, ytr)
    pred = m.predict_proba(Xte)[:, 1] if kind == "auc" else m.predict(Xte)
    res["xgb_uniform255"] = metric(kind, yte, pred)

    results[name] = res
    print(name, json.dumps(res, indent=None, default=str), flush=True)
    OUT.write_text(json.dumps(results, indent=2))


def main():
    results = {}
    run_dataset("synth_20of100", synth, results)
    run_dataset("california", lambda: load_local(
        "tests/data/california_housing_train.csv",
        "tests/data/california_housing_test.csv"), results)
    run_dataset("adult", lambda: load_openml(1590), results)
    run_dataset("kick", lambda: load_openml(41162), results)
    msd = REPO / "tests/data/year_prediction_msd_train.csv"
    if msd.exists():
        run_dataset("year_msd", lambda: load_local(
            "tests/data/year_prediction_msd_train.csv",
            "tests/data/year_prediction_msd_test.csv"), results)
    else:
        print("year_msd skipped (run scripts/fetch_year_msd.py first)")
    print("done ->", OUT)


if __name__ == "__main__":
    main()
