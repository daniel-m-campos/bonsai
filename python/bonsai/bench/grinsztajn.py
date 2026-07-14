"""Grinsztajn tabular benchmark: bonsai's external quality-division standings
suite (decision 68). Four OpenML suites from "Why do tree-based models still
outperform deep learning on tabular data?" (2022): 297/298 numerical
regression/classification, 299/304 categorical regression/classification,
55 tasks total.

Protocol: the paper's "medium" setting (train truncated to 10k rows), test
capped at 50k, categorical features as ordinal codes for every library, the
CAMPAIGN knob set (bonsai.bench.params), 3 seeds per task, primary metric per
task (r2 regression / AUC binary). Deviations from the paper: fixed random
splits instead of its resampling protocol, and no per-model tuning (matched
knobs is the point). xgboost's min_child_weight follows the campaign mapping;
see decision 68's correction for the bracketing caveat.

Resumable: rows already in the output jsonl are skipped.

    python -m bonsai.bench.grinsztajn out.jsonl
    python -m bonsai.bench.grinsztajn out.jsonl --report

Needs the [bench] extra (xgboost, lightgbm, catboost, scikit-learn, pandas,
openml).
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np

from . import metrics, params, runlog

SUITES = {297: "num_reg", 298: "num_clf", 299: "cat_reg", 304: "cat_clf"}
SEEDS = (0, 1, 2)
TRAIN_CAP, TEST_CAP = 10_000, 50_000
VARIANTS = ("bonsai_dw", "bonsai_lw", "bonsai_obl", "xgb", "lgbm", "catboost")
C = params.CAMPAIGN


def fit_predict(variant, Xtr, ytr, Xte, kind):
    if variant.startswith("bonsai"):
        import bonsai
        grower = {"bonsai_dw": "depthwise", "bonsai_lw": "leafwise",
                  "bonsai_obl": "oblivious"}[variant]
        obj = "logloss" if kind == "auc" else "mse"
        m = bonsai.BonsaiRegressor(
            objective=obj, grower=grower, n_iters=C["iters"],
            learning_rate=C["lr"], max_depth=C["depth"],
            max_leaves=params.num_leaves_campaign(C["depth"]),
            random_seed=C["seed"],
            params=params.BONSAI_CAMPAIGN_PARAMS).fit(Xtr, ytr)
        return np.asarray(m.predict(Xte))
    if variant == "xgb":
        import xgboost as xgb
        cls = xgb.XGBClassifier if kind == "auc" else xgb.XGBRegressor
        core = params.xgb_core(
            learning_rate=C["lr"], max_depth=C["depth"],
            min_data_in_leaf=C["min_data_in_leaf"], lambda_l2=C["lambda_l2"],
            max_bin=C["bins"], seed=C["seed"])
        core["random_state"] = core.pop("seed")
        m = cls(n_estimators=C["iters"], n_jobs=8, **core).fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1] if kind == "auc" else m.predict(Xte)
    if variant == "lgbm":
        import lightgbm as lgb
        obj = "binary" if kind == "auc" else "regression"
        p = {**params.lgbm_core(
                 learning_rate=C["lr"], max_depth=C["depth"],
                 num_leaves=params.num_leaves_campaign(C["depth"]),
                 min_data_in_leaf=C["min_data_in_leaf"],
                 lambda_l2=C["lambda_l2"], max_bin=C["bins"], seed=C["seed"]),
             "objective": obj, "num_iterations": C["iters"],
             "deterministic": True, "num_threads": 8}
        m = lgb.train(p, lgb.Dataset(Xtr, label=ytr))
        return m.predict(Xte)
    if variant == "catboost":
        import catboost as cb
        cls = cb.CatBoostClassifier if kind == "auc" else cb.CatBoostRegressor
        m = cls(**params.catboost_core(
                    learning_rate=C["lr"], max_depth=C["depth"],
                    lambda_l2=C["lambda_l2"], max_bin=C["bins"] - 1,
                    seed=C["seed"], device="cpu"),
                iterations=C["iters"], verbose=False, thread_count=8,
                allow_writing_files=False).fit(Xtr, ytr)
        return (m.predict_proba(Xte)[:, 1] if kind == "auc"
                else m.predict(Xte))
    raise ValueError(variant)


def load_task(task):
    import pandas as pd
    ds = task.get_dataset()
    X, y, _, _ = ds.get_data(target=task.target_name, dataset_format="dataframe")
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category").cat.codes
    Xn = X.to_numpy(np.float32)
    if pd.api.types.is_numeric_dtype(y):
        yn, kind = y.to_numpy(np.float32), "r2"
    else:
        yn, kind = (y == y.cat.categories[0]).to_numpy(np.float32), "auc"
    return Xn, yn, kind, ds.name


def _value(row) -> float | None:
    """Metric value across schema generations: v1 rows carry `value`, the
    pre-schema rows carried the number in `metric`."""
    v = row.get("value")
    if v is None and not isinstance(row.get("metric"), str):
        v = row.get("metric")
    return v


def run(out_path):
    import openml
    out = pathlib.Path(out_path)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["suite"], r["dataset"], r["variant"], r["seed"]))
    host = runlog.detect_host()
    knobs = dict(C, num_leaves_convention="campaign", train_cap=TRAIN_CAP)
    for sid, sname in SUITES.items():
        suite = openml.study.get_suite(sid)
        for tid in suite.tasks:
            try:
                X, y, kind, name = load_task(openml.tasks.get_task(
                    tid, download_splits=False))
            except Exception as e:
                print(f"skip task {tid}: {e!r}", flush=True)
                continue
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                idx = rng.permutation(len(X))
                n_tr = min(TRAIN_CAP, int(len(X) * 0.8))
                tr = idx[:n_tr]
                te = idx[n_tr:n_tr + TEST_CAP]
                for v in VARIANTS:
                    if (sname, name, v, seed) in done:
                        continue
                    t0 = time.time()
                    try:
                        pred = fit_predict(v, X[tr], y[tr], X[te], kind)
                        fn = metrics.auc if kind == "auc" else metrics.r2
                        value, status = fn(y[te], pred), "ok"
                    except Exception as e:
                        value, status = None, f"error: {e!r}"[:200]
                    runlog.emit_row(
                        out, division="quality", suite=sname, knobs=knobs,
                        host=host, dataset=name, task=kind, variant=v,
                        seed=seed, kind=kind, metric=kind, value=value,
                        status=status, n_train=len(tr),
                        n_features=int(X.shape[1]),
                        fit_s=round(time.time() - t0, 2))
                    shown = value if value is None else round(value, 4)
                    print(f"{sname:8s} {name[:28]:28s} {v:11s} s{seed} "
                          f"{shown}", flush=True)


def report(out_path):
    import pandas as pd
    rows = [json.loads(x)
            for x in pathlib.Path(out_path).read_text().splitlines()]
    ok = [dict(r, value=_value(r)) for r in rows if r["status"] == "ok"]
    df = pd.DataFrame(ok)
    mean = (df.groupby(["suite", "dataset", "variant"])["value"].mean()
            .reset_index())
    mean["lib"] = mean["variant"].map(
        lambda v: "bonsai" if v.startswith("bonsai") else v)
    lib = (mean.groupby(["suite", "dataset", "lib"])["value"].max()
           .reset_index())
    ranks = []
    for _k, g in lib.groupby(["suite", "dataset"]):
        g = g.copy()
        g["rank"] = g["value"].rank(ascending=False)
        ranks.append(g)
    rk = pd.concat(ranks)
    print("== library mean rank (best variant per lib; lower is better) ==")
    print(rk.groupby("lib")["rank"].mean().sort_values().round(3).to_string())
    print("\n== outright wins ==")
    print(rk[rk["rank"] == 1.0].groupby("lib").size()
          .sort_values(ascending=False).to_string())
    print("\n== per suite library mean rank ==")
    print(rk.groupby(["suite", "lib"])["rank"].mean().round(3).unstack()
          .to_string())
    print("\ndatasets per suite:\n"
          + rk.groupby(["suite"])["dataset"].nunique().to_string())


def main():
    if "--report" in sys.argv:
        report(sys.argv[1])
    else:
        run(sys.argv[1])


if __name__ == "__main__":
    main()
