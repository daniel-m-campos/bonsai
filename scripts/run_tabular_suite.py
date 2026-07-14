# Grinsztajn tabular benchmark runner (decision 68 candidate): the external
# standings suite. Four OpenML suites from "Why do tree-based models still
# outperform deep learning on tabular data?" (2022): 297/298 numerical
# regression/classification, 299/304 categorical regression/classification,
# 55 tasks total.
#
# Protocol: the paper's "medium" setting (train truncated to 10k rows), test
# capped at 50k, categorical features as ordinal codes for every library
# (campaign convention), campaign knobs (200 iters, lr 0.05, depth 6,
# 255 bins), 3 seeds per task, single-metric per task (r2 / AUC).
# Deviation from the paper: fixed random splits instead of their resampling
# protocol, and no per-model tuning (matched knobs is the campaign's point).
#
# Resumable: rows already in the output jsonl are skipped, so a killed run
# continues where it left off.
#
#   PYTHONPATH=build/python python3 scripts/run_tabular_suite.py \
#       benchmarks/results/grinsztajn-2026-07.jsonl
#   PYTHONPATH=build/python python3 scripts/run_tabular_suite.py \
#       benchmarks/results/grinsztajn-2026-07.jsonl --report
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "build/python"))
import bonsai

SUITES = {297: "num_reg", 298: "num_clf", 299: "cat_reg", 304: "cat_clf"}
SEEDS = (0, 1, 2)
TRAIN_CAP, TEST_CAP = 10_000, 50_000
HP = dict(n_iters=200, learning_rate=0.05, max_depth=6, max_leaves=63,
          random_seed=42)
BONSAI_PARAMS = {"tree.min_data_in_leaf": 20, "tree.lambda_l2": 1.0,
                 "bin_mapper.max_bin": 255}
VARIANTS = ("bonsai_dw", "bonsai_lw", "bonsai_obl", "xgb", "lgbm", "catboost")


def fit_predict(variant, Xtr, ytr, Xte, kind):
    if variant.startswith("bonsai"):
        grower = {"bonsai_dw": "depthwise", "bonsai_lw": "leafwise",
                  "bonsai_obl": "oblivious"}[variant]
        obj = "logloss" if kind == "auc" else "mse"
        m = bonsai.BonsaiRegressor(objective=obj, grower=grower, **HP,
                                   params=BONSAI_PARAMS).fit(Xtr, ytr)
        return np.asarray(m.predict(Xte))
    if variant == "xgb":
        import xgboost as xgb
        cls = xgb.XGBClassifier if kind == "auc" else xgb.XGBRegressor
        m = cls(n_estimators=200, learning_rate=0.05, max_depth=6, max_bin=255,
                tree_method="hist", reg_lambda=1.0, min_child_weight=1,
                random_state=42, n_jobs=8).fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1] if kind == "auc" else m.predict(Xte)
    if variant == "lgbm":
        import lightgbm as lgb
        obj = "binary" if kind == "auc" else "regression"
        m = lgb.train(dict(objective=obj, num_iterations=200, learning_rate=0.05,
                           max_depth=6, num_leaves=63, min_data_in_leaf=20,
                           lambda_l2=1.0, max_bin=255, verbose=-1, seed=42,
                           deterministic=True, num_threads=8),
                      lgb.Dataset(Xtr, label=ytr))
        return m.predict(Xte)
    if variant == "catboost":
        import catboost as cb
        cls = cb.CatBoostClassifier if kind == "auc" else cb.CatBoostRegressor
        m = cls(iterations=200, learning_rate=0.05, depth=6, l2_leaf_reg=1.0,
                border_count=254, random_seed=42, verbose=False,
                thread_count=8, allow_writing_files=False).fit(Xtr, ytr)
        return (m.predict_proba(Xte)[:, 1] if kind == "auc"
                else m.predict(Xte))
    raise ValueError(variant)


def metric(kind, y, pred):
    if kind == "auc":
        return float(roc_auc_score(y, pred))
    ss = float(((y - pred) ** 2).sum())
    tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss / tot


def load_task(task):
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


def run(out_path):
    import openml
    out = pathlib.Path(out_path)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["suite"], r["dataset"], r["variant"], r["seed"]))
    with out.open("a") as f:
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
                        key = (sname, name, v, seed)
                        if key in done:
                            continue
                        t0 = time.time()
                        try:
                            pred = fit_predict(v, X[tr], y[tr], X[te], kind)
                            m = metric(kind, y[te], pred)
                            status = "ok"
                        except Exception as e:
                            m, status = None, f"error: {e!r}"[:200]
                        row = dict(suite=sname, dataset=name, variant=v,
                                   seed=seed, kind=kind, metric=m,
                                   status=status, n_train=len(tr),
                                   n_features=int(X.shape[1]),
                                   fit_s=round(time.time() - t0, 2))
                        f.write(json.dumps(row) + "\n")
                        f.flush()
                        print(f"{sname:8s} {name[:28]:28s} {v:11s} s{seed} "
                              f"{m if m is None else round(m, 4)}", flush=True)


def report(out_path):
    rows = [json.loads(x) for x in pathlib.Path(out_path).read_text().splitlines()]
    df = pd.DataFrame([r for r in rows if r["status"] == "ok"])
    mean = df.groupby(["suite", "dataset", "variant"])["metric"].mean().reset_index()
    ranks = []
    for (_s, _d), g in mean.groupby(["suite", "dataset"]):
        g = g.copy()
        g["rank"] = g["metric"].rank(ascending=False)
        ranks.append(g)
    rk = pd.concat(ranks)
    print("== mean rank per variant (lower is better) ==")
    print(rk.groupby("variant")["rank"].mean().sort_values().round(3).to_string())
    print("\n== wins (rank 1) per variant ==")
    print(rk[rk["rank"] == 1.0].groupby("variant").size().sort_values(
        ascending=False).to_string())
    print("\n== per suite mean rank ==")
    print(rk.groupby(["suite", "variant"])["rank"].mean().round(3).unstack()
          .to_string())
    n = rk.groupby(["suite"])["dataset"].nunique()
    print("\ndatasets per suite:\n" + n.to_string())


if __name__ == "__main__":
    if "--report" in sys.argv:
        report(sys.argv[1])
    else:
        run(sys.argv[1])
