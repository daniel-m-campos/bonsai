#!/usr/bin/env python3
"""Ordered-boosting admission probe, rung 0: decompose CatBoost's small-data lead.

NOTE ON THE NAME. The campaign brief named this deliverable
`scripts/probe_ordered_boosting.py`, but that path is already a committed
decision-69 artifact (a 16M synthetic-scale ordered/plain/bonsai study, marked
"stays as-run"). To avoid clobbering committed evidence this rung-0 probe lives
in a distinct file. Different experiment entirely: small pure-numeric TabArena
data, AUC/rmse, plus a honest-gradient reachability prototype.

Feature-admission step 1 (measure the benefit at zero core cost). The TabArena-Lite
gauge and the categorical reopener probe both left CatBoost ahead of bonsai on PURE
NUMERIC small data with its categorical machinery ablated (decision 80). The leading
candidate mechanism for that residual is ORDERED BOOSTING: gradients computed by
models that never saw the row, which removes the prediction shift / target leakage of
ordinary boosting. That mechanism is confounded with CatBoost's strong tuned defaults.
This probe decomposes the lead with three arms, all at zero bonsai-core cost:

  ARM A  (defaults toggle): CatBoost boosting_type=Ordered vs =Plain, otherwise the
         library defaults. Ordered is CatBoost's own small-data default; this prices
         ordered boosting by CatBoost's own toggle at its defaults.
  ARM B  (matched-knobs toggle): CatBoost Ordered vs Plain at the quality-campaign
         shape (depth 6, lr 0.05, 1000 iters, early_stopping_rounds 50). Separates the
         strong-defaults share (defaults-vs-matched) from the mechanism share.
  ARM C  (reachability prototype): a hand-rolled 2-fold honest-gradient GBM vs a plain
         single-booster control at the SAME knobs. Does honest-gradient boosting,
         implementable outside bonsai's core, close part of the Plain->Ordered gap?

ARM C, exactly what is built (see the .md for the full derivation):
  Cross-fitted honest-gradient boosting with sklearn DecisionTreeRegressor weak learners
  (LightGBM is not in the gauge venv; a hand-rolled loop also isolates the mechanism more
  transparently, and the plain control uses the identical weak learner so the comparison
  is apples-to-apples). Train rows are split into fold0/fold1. Two SUPPORTING ensembles
  are grown, E0 on fold1 only (honest for fold0) and E1 on fold0 only (honest for fold1);
  each supporting model is grown with the OTHER model's honest gradients on its own fold,
  so no row's gradient ever depends on a model that trained on that row. Each round: the
  honest per-row negative gradient (E0 for fold0 rows, E1 for fold1 rows) becomes the
  target, and a single readout tree is fit on ALL rows with those honest targets; the
  readout ensemble is the final model (used only at predict time, so it never leaks). The
  plain control is the same loop except its gradients come from its own (biased,
  prediction-shifted) running margin. Regression uses the squared-error negative gradient
  (y - F); binary uses the logistic form (y - sigmoid(F)).

Everything runs in one uniform own-harness: a single train/val/test split per dataset
(gauge fold-0 split where available, else a fixed stratified holdout), single-model fits
(no inner bagging) so every arm shares the identical protocol and the CatBoost mechanism
delta and the arm-C mechanism delta are measured on the same rows. bonsai is run fresh on
the same split as the numeric baseline (== bonsai_ts on pure-numeric data).

Environment (the TabArena-Lite harness + venv are not vendored; run inside the gauge venv):
  TABARENA_DIR   tabarena checkout whose `packages` are importable + curated metadata.
  BONSAI_PYTHON  bonsai build python dir (default: <repo>/build-tabarena/python).
  CACHED_RESULTS results_per_split.csv from the gauge (for the external cross-check).
  OUT_JSONL      raw output rows (default: benchmarks/results/ordered-boosting-probe-...jsonl).
  PROBE_DATASETS optional comma-separated dataset override (smoke).
  ARMC_MAX_ITERS optional cap on arm-C / matched-CatBoost iterations (default 1000).

Verdict logic and the evidence tables live in
benchmarks/ordered-boosting-probe-2026-07.md.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
TABARENA_DIR = Path(
    os.environ.get("TABARENA_DIR", "/Users/danielmcampos/.claude/jobs/65493dc2/tmp/tabarena")
)
BONSAI_PYTHON = Path(os.environ.get("BONSAI_PYTHON", str(REPO / "build-tabarena/python")))
CACHED_RESULTS = Path(
    os.environ.get(
        "CACHED_RESULTS",
        str(TABARENA_DIR / "tmp_scripts/eval/bonsai_lite_lite/results_per_split.csv"),
    )
)
OUT_JSONL = Path(
    os.environ.get(
        "OUT_JSONL", str(REPO / "benchmarks/results/ordered-boosting-probe-2026-07.jsonl")
    )
)
MAX_ITERS = int(os.environ.get("ARMC_MAX_ITERS", "1000"))

for pkg in sorted((TABARENA_DIR / "packages").glob("*/src")):
    sys.path.insert(0, str(pkg))
sys.path.insert(0, str(BONSAI_PYTHON))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.datasets import fetch_openml, load_breast_cancer  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import LabelEncoder  # noqa: E402
from sklearn.tree import DecisionTreeRegressor  # noqa: E402

SEED = 42
LR = 0.05
DEPTH = 6
ES_ROUNDS = 50
MIN_LEAF = 20  # arm-C regularization (both honest and plain use it; does not bias the delta)
ROW_CAP = 30000

# ------------------------------------------------------------------ pool selection
# Rule: pure-numeric (zero categorical features) AND small (<= ~30k rows).
#   BASE       : every gauge-lite dataset with percentage_cat_features == 0, under the
#                30k cap (drops physiochemical_protein at 45730). All are regression.
#   EXTENSION  : easily-loadable pure-numeric OpenML/sklearn small datasets, added to
#                reach 10-14 and to bring binary classification (the gauge pure-numeric
#                subset is regression-only, so the logistic-gradient path and the AUC-
#                scale criterion would otherwise be untested). Each is asserted pure-
#                numeric (no category/object columns) and <= 30k rows at load time.
GAUGE_DATASETS = [
    ("QSAR_fish_toxicity", 363698, "regression"),
    ("concrete_compressive_strength", 363625, "regression"),
    ("QSAR-TID-11", 363697, "regression"),
    ("houses", 363678, "regression"),
    ("superconductivity", 363705, "regression"),
]
EXTENSION_DATASETS = [
    ("breast_cancer", ("sklearn", None), "binary"),
    ("pima_diabetes", ("openml", ("diabetes", 1)), "binary"),
    ("banknote", ("openml", ("banknote-authentication", 1)), "binary"),
    ("phoneme", ("openml", ("phoneme", 1)), "binary"),
    ("spambase", ("openml", ("spambase", 1)), "binary"),
    ("MagicTelescope", ("openml", ("MagicTelescope", 1)), "binary"),
    ("wind", ("openml", ("wind", 1)), "regression"),
]
GAUGE_NAMES = {d[0] for d in GAUGE_DATASETS}
GAUGE_TASK = {d[0]: d[1] for d in GAUGE_DATASETS}
EXT_SPEC = {d[0]: d[1] for d in EXTENSION_DATASETS}
PTYPE = {d[0]: d[2] for d in GAUGE_DATASETS + [(a, b, c) for (a, b, c) in EXTENSION_DATASETS]}


def _assert_numeric(name, X):
    cats = list(X.select_dtypes(include=["category", "object"]).columns)
    assert not cats, f"{name}: expected pure-numeric, found categorical columns {cats}"
    assert len(X) <= ROW_CAP, f"{name}: {len(X)} rows exceeds the {ROW_CAP} cap"


def load_dataset(name):
    """Return (Xtr, ytr, Xval, yval, Xte, yte) as float32/float64; y encoded for binary."""
    ptype = PTYPE[name]
    if name in GAUGE_NAMES:
        from tabarena.benchmark.task.openml.task_wrapper import OpenMLTaskWrapper

        w = OpenMLTaskWrapper.from_task_id(GAUGE_TASK[name])
        X, y = w.get_X_y()
        _assert_numeric(name, X)
        tr_idx, te_idx = w.get_split_indices(fold=0, repeat=0, sample=0)
        X = X.to_numpy(dtype=np.float32)
        y = np.asarray(y)
        Xtr_full, ytr_full, Xte, yte = X[tr_idx], y[tr_idx], X[te_idx], y[te_idx]
    else:
        kind, arg = EXT_SPEC[name]
        if kind == "sklearn":
            d = load_breast_cancer(as_frame=True)
        else:
            oname, ver = arg
            d = fetch_openml(oname, version=ver, as_frame=True, parser="auto")
        _assert_numeric(name, d.data)
        X = d.data.to_numpy(dtype=np.float32)
        y = np.asarray(d.target)
        strat = y if ptype == "binary" else None
        Xtr_full, Xte, ytr_full, yte = train_test_split(
            X, y, test_size=0.25, random_state=SEED, stratify=strat
        )

    if ptype == "binary":
        le = LabelEncoder().fit(np.concatenate([ytr_full, yte]))
        ytr_full = le.transform(ytr_full).astype(np.float64)
        yte = le.transform(yte).astype(np.float64)
    else:
        ytr_full = ytr_full.astype(np.float64)
        yte = yte.astype(np.float64)

    strat = ytr_full if ptype == "binary" else None
    Xtr, Xval, ytr, yval = train_test_split(
        Xtr_full, ytr_full, test_size=0.2, random_state=SEED, stratify=strat
    )
    return Xtr, ytr, Xval, yval, Xte, yte


def metric_error(ptype, y_true, pred):
    """Lower is better, matching TabArena: rmse (reg) / 1-roc_auc (binary)."""
    if ptype == "regression":
        return float(np.sqrt(np.mean((y_true - pred) ** 2)))
    return float(1.0 - roc_auc_score(y_true, pred))


# ------------------------------------------------------------------ bonsai baseline
def run_bonsai(ptype, Xtr, ytr, Xval, yval, Xte, yte):
    import bonsai

    cls = bonsai.BonsaiRegressor if ptype == "regression" else bonsai.BonsaiClassifier
    est = cls(
        n_iters=1000, learning_rate=LR, max_depth=DEPTH, grower="depthwise",
        early_stopping_rounds=ES_ROUNDS, n_threads=os.cpu_count(), random_seed=SEED,
    )
    t0 = time.time()
    est.fit(Xtr, ytr, eval_set=(Xval, yval))
    dt = time.time() - t0
    pred = est.predict(Xte) if ptype == "regression" else est.predict_proba(Xte)[:, 1]
    return metric_error(ptype, yte, pred), dt


# ------------------------------------------------------------------ CatBoost arms
def run_catboost(ptype, boosting_type, matched, Xtr, ytr, Xval, yval, Xte, yte):
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool

    params = dict(
        boosting_type=boosting_type, random_seed=SEED, early_stopping_rounds=ES_ROUNDS,
        verbose=0, allow_writing_files=False,
    )
    if matched:
        params.update(depth=DEPTH, learning_rate=LR, iterations=1000)
    cls = CatBoostRegressor if ptype == "regression" else CatBoostClassifier
    est = cls(**params)
    t0 = time.time()
    est.fit(Pool(Xtr, ytr), eval_set=Pool(Xval, yval))
    dt = time.time() - t0
    pred = est.predict(Xte) if ptype == "regression" else est.predict_proba(Xte)[:, 1]
    return metric_error(ptype, yte, pred), dt, int(est.tree_count_)


# ------------------------------------------------------------------ ARM C
def _neg_grad(ptype, y, margin):
    if ptype == "regression":
        return y - margin
    return y - 1.0 / (1.0 + np.exp(-margin))


def _base_score(ptype, y):
    if ptype == "regression":
        return float(np.mean(y))
    p = float(np.clip(np.mean(y), 1e-6, 1 - 1e-6))
    return float(np.log(p / (1 - p)))


def _val_loss(ptype, y, margin):
    if ptype == "regression":
        return float(np.sqrt(np.mean((y - margin) ** 2)))
    p = np.clip(1.0 / (1.0 + np.exp(-margin)), 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _impute(Xtr, *others):
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)

    def fill(A):
        A = A.copy()
        idx = np.where(np.isnan(A))
        A[idx] = np.take(med, idx[1])
        return A

    return (fill(Xtr), *[fill(o) for o in others])


def _new_tree():
    return DecisionTreeRegressor(max_depth=DEPTH, min_samples_leaf=MIN_LEAF, random_state=SEED)


def run_armc_plain(ptype, Xtr, ytr, Xval, yval, Xte, yte, max_iters):
    Xtr, Xval, Xte = _impute(Xtr, Xval, Xte)
    base = _base_score(ptype, ytr)
    m = np.full(len(ytr), base)
    mv = np.full(len(yval), base)
    trees = []
    best_loss, best_k, since = np.inf, 0, 0
    t0 = time.time()
    for _ in range(max_iters):
        ng = _neg_grad(ptype, ytr, m)
        tree = _new_tree().fit(Xtr, ng)
        m += LR * tree.predict(Xtr)
        mv += LR * tree.predict(Xval)
        trees.append(tree)
        loss = _val_loss(ptype, yval, mv)
        if loss < best_loss - 1e-9:
            best_loss, best_k, since = loss, len(trees), 0
        else:
            since += 1
            if since >= ES_ROUNDS:
                break
    dt = time.time() - t0
    mte = np.full(len(yte), base)
    for tree in trees[:best_k]:
        mte += LR * tree.predict(Xte)
    pred = mte if ptype == "regression" else 1.0 / (1.0 + np.exp(-mte))
    return metric_error(ptype, yte, pred), dt, best_k


def _make_folds(ptype, ytr, rng):
    n = len(ytr)
    fold = np.zeros(n, dtype=int)
    if ptype == "binary":
        for c in (0, 1):
            ci = np.where(ytr == c)[0]
            fold[ci[rng.permutation(len(ci))[: len(ci) // 2]]] = 1
    else:
        fold[rng.permutation(n)[: n // 2]] = 1
    return fold


def run_armc_couple(ptype, Xtr, ytr, Xval, yval, Xte, yte, max_iters):
    """The FAITHFUL honest-gradient prototype (the campaign's primary form).

    Two boosters F0, F1 carry the running prediction (init_score); the final margin
    is base + F0 + F1. A row in fold0 gets its per-round gradient from the booster
    trained only on the OTHER fold (base + F1, which never saw fold0), and vice versa,
    so no row's gradient depends on a model that trained on it. Each round fits fold0's
    tree on fold0 rows with fold0's honest gradient and fold1's tree on fold1 rows.
    This is exactly ordered boosting's unbiased-gradient idea reduced to 2 permutation
    folds. It does NOT converge at 2 folds: a fold's honest gradient excludes its own
    accumulator entirely (all of that accumulator's trees are contaminated for the fold),
    so the gradient never vanishes and the running prediction overshoots. Reported as-is
    with early stopping, which is the honest reachability finding, not a bug to hide."""
    Xtr, Xval, Xte = _impute(Xtr, Xval, Xte)
    n = len(ytr)
    fold = _make_folds(ptype, ytr, np.random.RandomState(SEED))
    f0, f1 = np.where(fold == 0)[0], np.where(fold == 1)[0]
    base = _base_score(ptype, ytr)
    F0 = np.zeros(n)
    F1 = np.zeros(n)
    F0v = np.zeros(len(yval))
    F1v = np.zeros(len(yval))
    F0t = np.zeros(len(yte))
    F1t = np.zeros(len(yte))
    best_loss, best_k, since = np.inf, 0, 0
    best_margin_te = np.full(len(yte), base)
    t0 = time.time()
    for k in range(max_iters):
        g = np.empty(n)
        g[f0] = _neg_grad(ptype, ytr[f0], base + F1[f0])  # honest: exclude F0
        g[f1] = _neg_grad(ptype, ytr[f1], base + F0[f1])  # honest: exclude F1
        t_0 = _new_tree().fit(Xtr[f0], g[f0])
        F0 += LR * t_0.predict(Xtr)
        F0v += LR * t_0.predict(Xval)
        F0t += LR * t_0.predict(Xte)
        t_1 = _new_tree().fit(Xtr[f1], g[f1])
        F1 += LR * t_1.predict(Xtr)
        F1v += LR * t_1.predict(Xval)
        F1t += LR * t_1.predict(Xte)
        loss = _val_loss(ptype, yval, base + F0v + F1v)
        if loss < best_loss - 1e-9:
            best_loss, best_k, since = loss, k + 1, 0
            best_margin_te = base + F0t.copy() + F1t.copy()
        else:
            since += 1
            if since >= ES_ROUNDS:
                break
    dt = time.time() - t0
    pred = best_margin_te if ptype == "regression" else 1.0 / (1.0 + np.exp(-best_margin_te))
    return metric_error(ptype, yte, pred), dt, best_k


def run_armc_bag(ptype, Xtr, ytr, Xval, yval, Xte, yte, max_iters):
    """The converging simpler form the campaign flagged as acceptable: two independent
    proper boosters, each trained on one fold with ordinary within-fold gradients,
    margins averaged at predict. Converges (each is a proper GBM) and is out-of-fold
    honest at test, but its per-fold trees use ordinary (biased) within-fold gradients,
    so it is 2-fold bagging and does NOT inject the honest mechanism into training."""
    Xtr, Xval, Xte = _impute(Xtr, Xval, Xte)
    fold = _make_folds(ptype, ytr, np.random.RandomState(SEED))
    t0 = time.time()
    margins_te = []
    total_k = 0
    for fv in (0, 1):
        idx = np.where(fold == fv)[0]
        Xf, yf = Xtr[idx], ytr[idx]
        base = _base_score(ptype, yf)
        m = np.full(len(yf), base)
        mv = np.full(len(yval), base)
        mte = np.full(len(yte), base)
        best_loss, since = np.inf, 0
        best_mte = mte.copy()
        for k in range(max_iters):
            ng = _neg_grad(ptype, yf, m)
            tree = _new_tree().fit(Xf, ng)
            m += LR * tree.predict(Xf)
            mv += LR * tree.predict(Xval)
            mte += LR * tree.predict(Xte)
            loss = _val_loss(ptype, yval, mv)
            if loss < best_loss - 1e-9:
                best_loss, since, best_mte = loss, 0, mte.copy()
                total_k = max(total_k, k + 1)
            else:
                since += 1
                if since >= ES_ROUNDS:
                    break
        margins_te.append(best_mte)
    dt = time.time() - t0
    margin = 0.5 * (margins_te[0] + margins_te[1])
    pred = margin if ptype == "regression" else 1.0 / (1.0 + np.exp(-margin))
    return metric_error(ptype, yte, pred), dt, total_k


# ------------------------------------------------------------------ driver
def cached_reference():
    try:
        res = pd.read_csv(CACHED_RESULTS)
    except Exception:
        return {}
    wanted = {"CAT (default)": "cat", "[New] BONSAI (default)": "bonsai"}
    sub = res[res["method"].isin(wanted)]
    out = {}
    for ds, g in sub.groupby("dataset"):
        out[ds] = {wanted[r["method"]]: float(r["metric_error"]) for _, r in g.iterrows()}
    return out


def main():
    t_start = time.time()
    override = os.environ.get("PROBE_DATASETS", "").strip()
    if override:
        names = [d.strip() for d in override.split(",") if d.strip()]
    else:
        names = [g[0] for g in GAUGE_DATASETS] + [e[0] for e in EXTENSION_DATASETS]

    cached = cached_reference()
    rows = []
    for name in names:
        ptype = PTYPE[name]
        is_gauge = name in GAUGE_NAMES
        print(f"\n=== {name} ({ptype}, {'gauge' if is_gauge else 'ext'}) ===", flush=True)
        Xtr, ytr, Xval, yval, Xte, yte = load_dataset(name)
        print(f"  rows tr/val/te = {len(ytr)}/{len(yval)}/{len(yte)}  feats={Xtr.shape[1]}",
              flush=True)

        bonsai_err, bonsai_t = run_bonsai(ptype, Xtr, ytr, Xval, yval, Xte, yte)
        ord_def, ord_def_t, ord_def_k = run_catboost(
            ptype, "Ordered", False, Xtr, ytr, Xval, yval, Xte, yte)
        plain_def, plain_def_t, plain_def_k = run_catboost(
            ptype, "Plain", False, Xtr, ytr, Xval, yval, Xte, yte)
        ord_m, ord_m_t, ord_m_k = run_catboost(
            ptype, "Ordered", True, Xtr, ytr, Xval, yval, Xte, yte)
        plain_m, plain_m_t, plain_m_k = run_catboost(
            ptype, "Plain", True, Xtr, ytr, Xval, yval, Xte, yte)
        c_couple, c_couple_t, c_couple_k = run_armc_couple(
            ptype, Xtr, ytr, Xval, yval, Xte, yte, MAX_ITERS)
        c_bag, c_bag_t, c_bag_k = run_armc_bag(
            ptype, Xtr, ytr, Xval, yval, Xte, yte, MAX_ITERS)
        c_plain, c_plain_t, c_plain_k = run_armc_plain(
            ptype, Xtr, ytr, Xval, yval, Xte, yte, MAX_ITERS)

        row = {
            "dataset": name, "source": "gauge" if is_gauge else "ext",
            "problem_type": ptype,
            "metric": "rmse" if ptype == "regression" else "one_minus_auc",
            "n_train": len(ytr), "n_val": len(yval), "n_test": len(yte),
            "n_features": int(Xtr.shape[1]),
            "bonsai": bonsai_err,
            "cat_ordered_def": ord_def, "cat_plain_def": plain_def,
            "cat_ordered_matched": ord_m, "cat_plain_matched": plain_m,
            "armc_couple": c_couple, "armc_bag": c_bag, "armc_plain": c_plain,
            "ordered_share_def": plain_def - ord_def,
            "ordered_share_matched": plain_m - ord_m,
            "honest_share_couple": c_plain - c_couple,   # faithful honest form vs plain
            "honest_share_bag": c_plain - c_bag,         # converging simpler form vs plain
            "catboost_lead": bonsai_err - ord_def,
            "plain_vs_bonsai": bonsai_err - plain_def,
            "defaults_share": plain_m - plain_def,
            "time_bonsai_s": bonsai_t,
            "time_cat_ordered_def_s": ord_def_t, "time_cat_plain_def_s": plain_def_t,
            "time_cat_ordered_matched_s": ord_m_t, "time_cat_plain_matched_s": plain_m_t,
            "time_armc_couple_s": c_couple_t, "time_armc_bag_s": c_bag_t,
            "time_armc_plain_s": c_plain_t,
            "trees_cat_ordered_def": ord_def_k, "trees_cat_plain_def": plain_def_k,
            "trees_cat_ordered_matched": ord_m_k, "trees_cat_plain_matched": plain_m_k,
            "trees_armc_couple": c_couple_k, "trees_armc_bag": c_bag_k,
            "trees_armc_plain": c_plain_k,
        }
        if name in cached:
            row["cached_cat_default"] = cached[name].get("cat")
            row["cached_bonsai_default"] = cached[name].get("bonsai")
        rows.append(row)
        print(f"  bonsai={bonsai_err:.5f} ord_def={ord_def:.5f} plain_def={plain_def:.5f} "
              f"| ord_m={ord_m:.5f} plain_m={plain_m:.5f} "
              f"| C_couple={c_couple:.5f} C_bag={c_bag:.5f} C_plain={c_plain:.5f}", flush=True)
        print(f"  shares: ord_def={row['ordered_share_def']:+.5f} "
              f"ord_matched={row['ordered_share_matched']:+.5f} "
              f"honest_couple={row['honest_share_couple']:+.5f} "
              f"honest_bag={row['honest_share_bag']:+.5f} "
              f"cat_lead={row['catboost_lead']:+.5f}", flush=True)

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 260)
    print("\n\n########## SUMMARY ##########")
    for pt, g in df.groupby("problem_type"):
        print(f"\n=== {pt} pool (n={len(g)}, metric={g['metric'].iloc[0]}, lower better) ===")
        show = ["dataset", "bonsai", "cat_ordered_def", "cat_plain_def",
                "cat_ordered_matched", "cat_plain_matched", "armc_couple", "armc_bag",
                "armc_plain", "ordered_share_def", "ordered_share_matched",
                "honest_share_couple", "honest_share_bag", "catboost_lead"]
        print(g[show].to_string(index=False))
        print(f"  MEAN ordered_share_def     : {g['ordered_share_def'].mean():+.5f}")
        print(f"  MEAN ordered_share_matched : {g['ordered_share_matched'].mean():+.5f}")
        print(f"  MEAN honest_share_couple   : {g['honest_share_couple'].mean():+.5f}")
        print(f"  MEAN honest_share_bag      : {g['honest_share_bag'].mean():+.5f}")
        print(f"  MEAN catboost_lead         : {g['catboost_lead'].mean():+.5f}")
        print(f"  MEAN defaults_share        : {g['defaults_share'].mean():+.5f}")
        print(f"  MEAN plain_vs_bonsai       : {g['plain_vs_bonsai'].mean():+.5f}")

    print(f"\nwrote {OUT_JSONL}  ({len(rows)} rows)  wall {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
