# PROVENANCE NOTE (decision 69): this probe is a completed experiment whose
# committed evidence was produced by this exact code; it stays as-run rather
# than being refactored onto bonsai.bench. NEW probes must import their knobs
# and reference-library mappings from bonsai.bench.params and their metrics
# from bonsai.bench.metrics.
# Categorical trade-off probe (decision 58): measure the BENEFIT side of the
# native-categoricals complexity trade with zero bonsai C++ changes.
#   1. Benefit ceiling: each reference library with its categorical machinery
#      ON vs OFF (ordinal codes) at campaign-matched knobs.
#   2. Zero-core-cost alternative: ordered target statistics as preprocessing
#      feeding today's bonsai (single perm, multi-perm average, TS+code).
# Output: one JSON of AUCs per dataset x variant.
import json
import pathlib
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "build/python"))
import bonsai

OUT = pathlib.Path(sys.argv[1])
HP = dict(n_iters=200, lr=0.05, depth=6, leaves=63, min_leaf=20, l2=1.0,
          max_bin=255, seed=42)

# name -> (openml id, all_cols_categorical)
DATASETS = {
    "amazon": (4135, True),    # 32.7k x 9, all high-card ID columns
    "adult": (1590, False),    # 48.8k x 14, mixed
    "kick": (41162, False),    # 72.9k x 32, mixed
}


def ordered_ts(codes_tr, y_tr, codes_te, prior_w, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y_tr))
    prior = float(y_tr.mean())
    sums, cnts = defaultdict(float), defaultdict(int)
    enc = np.empty(len(y_tr), dtype=np.float32)
    for i in perm:
        c = codes_tr[i]
        enc[i] = (sums[c] + prior_w * prior) / (cnts[c] + prior_w)
        sums[c] += y_tr[i]
        cnts[c] += 1
    te = np.array([(sums[c] + prior_w * prior) / (cnts[c] + prior_w)
                   for c in codes_te], dtype=np.float32)
    return enc, te


def encode_ts(Xtr_codes, y_tr, Xte_codes, cat_cols, prior_w, seeds):
    tr, te = Xtr_codes.copy(), Xte_codes.copy()
    for c in cat_cols:
        accs_tr = np.zeros(len(tr), dtype=np.float64)
        accs_te = np.zeros(len(te), dtype=np.float64)
        for s in seeds:
            a, b = ordered_ts(Xtr_codes[c].to_numpy(), y_tr,
                              Xte_codes[c].to_numpy(), prior_w, s)
            accs_tr += a
            accs_te += b
        tr[c] = (accs_tr / len(seeds)).astype(np.float32)
        te[c] = (accs_te / len(seeds)).astype(np.float32)
    return tr, te


def run_bonsai(Xtr, ytr, Xte):
    m = bonsai.BonsaiRegressor(
        n_iters=HP["n_iters"], learning_rate=HP["lr"], objective="logloss",
        max_depth=HP["depth"], max_leaves=HP["leaves"], grower="depthwise",
        random_seed=HP["seed"],
        params={"tree.min_data_in_leaf": HP["min_leaf"],
                "tree.lambda_l2": HP["l2"],
                "bin_mapper.max_bin": HP["max_bin"]})
    m.fit(Xtr.to_numpy(np.float32), ytr.astype(np.float32))
    return m.predict(Xte.to_numpy(np.float32))


results = {}
for name, (did, all_cat) in DATASETS.items():
    t0 = time.time()
    ds = fetch_openml(data_id=did, as_frame=True, parser="auto")
    frame = ds.frame
    target = ds.target_names[0]
    y = frame[target].astype("category").cat.codes.to_numpy().astype(np.float64)
    X_raw = frame.drop(columns=[target])
    cat_cols = list(X_raw.columns) if all_cat else [
        c for c in X_raw.columns
        if X_raw[c].dtype.name in ("category", "object", "bool", "string")]
    # ordinal code frame (NaN -> its own category 0 via +1 shift)
    X_codes = pd.DataFrame({
        c: (X_raw[c].astype("category").cat.codes.astype(np.int64) + 1
            if c in cat_cols
            else pd.to_numeric(X_raw[c], errors="coerce").astype(np.float32))
        for c in X_raw.columns})
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(y))
    cut = int(len(y) * 0.8)
    itr, ite = idx[:cut], idx[cut:]
    ytr, yte = y[itr], y[ite]
    Xtr_c, Xte_c = X_codes.iloc[itr].reset_index(drop=True), \
        X_codes.iloc[ite].reset_index(drop=True)
    r = {}

    import lightgbm as lgb
    base = dict(n_estimators=HP["n_iters"], learning_rate=HP["lr"],
                max_depth=HP["depth"], num_leaves=HP["leaves"],
                min_child_samples=HP["min_leaf"], reg_lambda=HP["l2"],
                max_bin=HP["max_bin"], random_state=HP["seed"], verbose=-1)
    m = lgb.LGBMClassifier(**base).fit(Xtr_c, ytr)
    r["lgbm_ordinal"] = roc_auc_score(yte, m.predict_proba(Xte_c)[:, 1])
    Xtr_n = Xtr_c.copy()
    Xte_n = Xte_c.copy()
    for c in cat_cols:
        Xtr_n[c] = Xtr_n[c].astype("category")
        Xte_n[c] = pd.Categorical(Xte_n[c],
                                  categories=Xtr_n[c].cat.categories)
    m = lgb.LGBMClassifier(**base).fit(Xtr_n, ytr,
                                       categorical_feature=cat_cols)
    r["lgbm_native"] = roc_auc_score(yte, m.predict_proba(Xte_n)[:, 1])

    from catboost import CatBoostClassifier
    cb = dict(iterations=HP["n_iters"], learning_rate=HP["lr"],
              depth=HP["depth"], l2_leaf_reg=HP["l2"],
              border_count=HP["max_bin"], random_seed=HP["seed"],
              verbose=False, allow_writing_files=False)
    m = CatBoostClassifier(**cb).fit(Xtr_c, ytr)
    r["catboost_ordinal"] = roc_auc_score(yte, m.predict_proba(Xte_c)[:, 1])
    Xtr_s = Xtr_c.copy()
    Xte_s = Xte_c.copy()
    for c in cat_cols:
        Xtr_s[c] = Xtr_s[c].astype(str)
        Xte_s[c] = Xte_s[c].astype(str)
    m = CatBoostClassifier(**cb, cat_features=cat_cols).fit(Xtr_s, ytr)
    r["catboost_native"] = roc_auc_score(yte, m.predict_proba(Xte_s)[:, 1])

    try:
        from xgboost import XGBClassifier
        xb = dict(n_estimators=HP["n_iters"], learning_rate=HP["lr"],
                  max_depth=HP["depth"], min_child_weight=HP["min_leaf"],
                  reg_lambda=HP["l2"], max_bin=HP["max_bin"],
                  tree_method="hist", random_state=HP["seed"])
        m = XGBClassifier(**xb).fit(Xtr_c, ytr)
        r["xgb_ordinal"] = roc_auc_score(yte, m.predict_proba(Xte_c)[:, 1])
        m = XGBClassifier(**xb, enable_categorical=True).fit(Xtr_n, ytr)
        r["xgb_native"] = roc_auc_score(yte, m.predict_proba(Xte_n)[:, 1])
    except Exception as e:
        r["xgb_native_error"] = str(e)

    Xtr_f = Xtr_c.astype(np.float32)
    Xte_f = Xte_c.astype(np.float32)
    r["bonsai_ordinal"] = roc_auc_score(yte, run_bonsai(Xtr_f, ytr, Xte_f))
    for label, (w, seeds) in {
            "bonsai_ts_a1": (1.0, [0]),
            "bonsai_ts_a10": (10.0, [0]),
            "bonsai_ts4_a10": (10.0, [0, 1, 2, 3])}.items():
        Ttr, Tte = encode_ts(Xtr_c, ytr, Xte_c, cat_cols, w, seeds)
        r[label] = roc_auc_score(
            yte, run_bonsai(Ttr.astype(np.float32), ytr,
                            Tte.astype(np.float32)))
    # TS + raw code side by side: the tree sees both views.
    Ttr, Tte = encode_ts(Xtr_c, ytr, Xte_c, cat_cols, 10.0, [0])
    for c in cat_cols:
        Ttr[c + "__code"] = Xtr_c[c]
        Tte[c + "__code"] = Xte_c[c]
    r["bonsai_ts_plus_code"] = roc_auc_score(
        yte, run_bonsai(Ttr.astype(np.float32), ytr, Tte.astype(np.float32)))

    r["_seconds"] = round(time.time() - t0, 1)
    r["_n_cat"] = len(cat_cols)
    r["_shape"] = list(X_raw.shape)
    results[name] = r
    print(f"{name}: {json.dumps(r)}", flush=True)

OUT.write_text(json.dumps(results, indent=1))
print("PROBE-DONE")
