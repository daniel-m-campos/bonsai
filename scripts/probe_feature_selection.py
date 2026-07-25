#!/usr/bin/env python3
"""Feature-selection admission probe: does refit-based honest feature selection
(a shadow-feature prototype) buy measurable accuracy, and how does it price
against CatBoost's select_features, the only in-library selector among the
reference libraries?

Feature-admission step 1 (measure the benefit at zero core cost). The shadow
prototype lives entirely in Python (append permuted shadow copies, fit stock
bonsai, threshold real importances against the shadow importances); if it earns
its keep the recommendation is a bonsai.select module, no bonsai-core change.
CatBoost's select_features (RecursiveByLossFunctionChange) is the in-library
reference point, priced by its own toggle. XGBoost and bonsai top-k-by-gain are
the truncation controls that isolate the shadow machinery from a plain ranking.

Two regimes, all drawn from the rung-0 12-dataset pool (scripts/probe_ordered_
boosting_rung0.py, imported read-only for its loader and splits):

  REAL-WIDE      the feature count is the point: QSAR-TID-11 (1024), superconduct-
                 ivity (81), spambase (57). Selection is measured on real data
                 with whatever redundancy the features already carry.
  NOISE-INJECTED houses, concrete_compressive_strength, wind, breast_cancer,
                 pima_diabetes, MagicTelescope, each augmented with shuffled-copy
                 noise columns equal to the original feature count. Each real
                 column is permuted independently (fixed seed rng) to destroy the
                 target association while keeping the marginal, so the injected
                 columns are known-noise ground truth for a precision/recall table.

Six arms per dataset (k is the count arm 4 keeps, so the truncation arms 3/5/6
run at arm 4's budget and 3-vs-4 isolates the shadow mechanism from truncation):

  1 bonsai_all      all features, the baseline; also the gain ranking arm 3 uses.
  2 cat_all         CatBoost Plain matched, all features (library-vs-selection).
  3 bonsai_topk_gain refit bonsai on the top-k features by bonsai's own gain
                    importance from arm 1 (equal-budget truncation control).
  4 bonsai_shadow   the prototype: over 5 seeds, append one permuted shadow copy
                    of every feature, fit, keep every real feature whose gain
                    exceeds the 95th percentile of all shadow importances; keep
                    features selected in >= 3 of 5 seeds; refit on the kept set.
  5 cat_select      CatBoost select_features RecursiveByLossFunctionChange at the
                    matched knobs, down to k, final model refit on the chosen set.
  6 xgb_topk_gain   xgboost matched all-features fit, refit on its top-k by gain.

Matched knobs throughout (the quality-campaign shape): depth 6, learning_rate
0.05, iterations cap 1000, early_stopping_rounds 50, min_data_in_leaf 20,
lambda_l2 1.0, max_bin 255 bonsai / 254 CatBoost, CatBoost boosting_type Plain.
Metric_error is lower-better: rmse (regression), 1 minus roc_auc (binary). Chance
band per decision 55: about 2% relative of the metric for rmse, 0.001 absolute AUC.

Verdict logic and the evidence tables live in
benchmarks/feature-selection-probe-2026-07.md.

Environment (the TabArena-Lite harness + venv are not vendored; run inside the
gauge venv, CatBoost 1.2.10 / xgboost / scikit-learn 1.7.2):
  BONSAI_PYTHON  bonsai build python dir (a real build; a worktree may not have one).
  TABARENA_DIR   tabarena checkout (importable packages + curated metadata).
  --out          raw output jsonl (default benchmarks/results/feature-selection-...jsonl).
  PROBE_DATASETS optional comma-separated dataset override (smoke runs).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

import numpy as np  # noqa: E402

# The rung-0 probe sets sys.path for tabarena + bonsai at import time using its own
# BONSAI_PYTHON default; a worktree has no build there, so the caller sets the env.
# bonsai.bench is imported lazily inside the fit helpers, only after this line runs.
import probe_ordered_boosting_rung0 as rung0  # noqa: E402

SEED = 42
LR = 0.05
DEPTH = 6
ES_ROUNDS = 50
LAMBDA_L2 = 1.0
MIN_DATA_IN_LEAF = 20
MAX_BIN_BONSAI = 255
MAX_BIN_CAT = 254
ITERS = 1000
THREADS = os.cpu_count()

# Shadow prototype knobs.
SHADOW_SEEDS = [42, 43, 44, 45, 46]
SHADOW_PERCENTILE = 95.0
SHADOW_AGREE = 3  # keep a feature selected in at least this many seeds

# CatBoost select_features knobs (recorded so the wall-time comparison is honest).
CAT_SELECT_STEPS = 5
CAT_SELECT_ALGO = "RecursiveByLossFunctionChange"

REAL_WIDE = ["QSAR-TID-11", "superconductivity", "spambase"]
NOISE_INJECTED = [
    "houses", "concrete_compressive_strength", "wind",
    "breast_cancer", "pima_diabetes", "MagicTelescope",
]
NOISE_SEED = 42

PTYPE = rung0.PTYPE

BONSAI_MATCHED_PARAMS = {
    "tree.min_data_in_leaf": MIN_DATA_IN_LEAF,
    "tree.lambda_l2": LAMBDA_L2,
    "bin_mapper.max_bin": MAX_BIN_BONSAI,
}


def metric_error(ptype, y_true, pred):
    """Lower is better, matching TabArena and the sibling probes."""
    from bonsai.bench import metrics as bench_metrics

    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    if ptype == "regression":
        return bench_metrics.rmse(y, p)
    return 1.0 - bench_metrics.auc(y, p)


def band(ptype, reference):
    """Chance band: about 2% relative of the metric for rmse, 0.001 absolute AUC."""
    return 0.001 if ptype == "binary" else 0.02 * abs(reference)


def inject_noise(X, seed):
    """Append one permuted copy of every column: X_noise[:, j] is a row-shuffle of
    X[:, j], which destroys the target association while keeping the marginal."""
    rng = np.random.RandomState(seed)
    noise = np.empty_like(X)
    for j in range(X.shape[1]):
        noise[:, j] = X[rng.permutation(X.shape[0]), j]
    return np.concatenate([X, noise], axis=1)


def shadow_augment(X, seed):
    """Append one permuted shadow copy of every column (per-seed rng). Returns the
    (n, 2p) matrix; columns [p, 2p) are the shadows."""
    return inject_noise(X, seed)


# ------------------------------------------------------------------ bonsai
def new_bonsai(ptype):
    import bonsai

    cls = bonsai.BonsaiRegressor if ptype == "regression" else bonsai.BonsaiClassifier
    return cls(
        n_iters=ITERS, learning_rate=LR, max_depth=DEPTH, grower="depthwise",
        early_stopping_rounds=ES_ROUNDS, n_threads=THREADS, random_seed=SEED,
        params=dict(BONSAI_MATCHED_PARAMS),
    )


def fit_bonsai(ptype, Xtr, ytr, Xval, yval, Xte):
    est = new_bonsai(ptype)
    est.fit(Xtr, ytr, eval_set=(Xval, yval))
    pred = est.predict(Xte) if ptype == "regression" else est.predict_proba(Xte)[:, 1]
    return pred, est


# ------------------------------------------------------------------ CatBoost
def fit_catboost_all(ptype, Xtr, ytr, Xval, yval, Xte):
    from bonsai.bench.params import catboost_core
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool

    params = dict(catboost_core(
        learning_rate=LR, max_depth=DEPTH, lambda_l2=LAMBDA_L2,
        max_bin=MAX_BIN_CAT, seed=SEED, device="cpu"))
    params.update(
        boosting_type="Plain", iterations=ITERS, early_stopping_rounds=ES_ROUNDS,
        verbose=0, allow_writing_files=False, thread_count=THREADS,
    )
    cls = CatBoostRegressor if ptype == "regression" else CatBoostClassifier
    est = cls(**params)
    est.fit(Pool(Xtr, ytr), eval_set=Pool(Xval, yval))
    pred = est.predict(Xte) if ptype == "regression" else est.predict_proba(Xte)[:, 1]
    return pred


def cat_select(ptype, Xtr, ytr, Xval, yval, Xte, k):
    """CatBoost select_features down to k, final model refit on the chosen set.
    Returns (pred, selected_indices, wall_s) or raises for the caller to record."""
    from bonsai.bench.params import catboost_core
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool

    params = dict(catboost_core(
        learning_rate=LR, max_depth=DEPTH, lambda_l2=LAMBDA_L2,
        max_bin=MAX_BIN_CAT, seed=SEED, device="cpu"))
    params.update(
        boosting_type="Plain", iterations=ITERS, early_stopping_rounds=ES_ROUNDS,
        verbose=0, allow_writing_files=False, thread_count=THREADS,
    )
    cls = CatBoostRegressor if ptype == "regression" else CatBoostClassifier
    est = cls(**params)
    n_feat = Xtr.shape[1]
    t0 = time.time()
    summary = est.select_features(
        Pool(Xtr, ytr), eval_set=Pool(Xval, yval),
        features_for_select=list(range(n_feat)),
        num_features_to_select=k, algorithm=CAT_SELECT_ALGO,
        steps=CAT_SELECT_STEPS, train_final_model=True, verbose=0,
    )
    dt = time.time() - t0
    selected = sorted(int(i) for i in summary["selected_features"])
    pred = est.predict(Xte) if ptype == "regression" else est.predict_proba(Xte)[:, 1]
    return pred, selected, dt


# ------------------------------------------------------------------ XGBoost
def fit_xgb(ptype, Xtr, ytr, Xval, yval, Xte):
    """All-features matched xgboost fit; returns (pred, gain_importance_vector)."""
    from bonsai.bench.params import xgb_core
    from xgboost import XGBClassifier, XGBRegressor

    core = xgb_core(
        learning_rate=LR, max_depth=DEPTH, min_data_in_leaf=MIN_DATA_IN_LEAF,
        lambda_l2=LAMBDA_L2, max_bin=MAX_BIN_BONSAI, seed=SEED)
    common = dict(
        n_estimators=ITERS, early_stopping_rounds=ES_ROUNDS, n_jobs=THREADS,
        importance_type="gain",
    )
    cls = XGBRegressor if ptype == "regression" else XGBClassifier
    est = cls(**core, **common)
    est.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    pred = est.predict(Xte) if ptype == "regression" else est.predict_proba(Xte)[:, 1]
    gain = np.zeros(Xtr.shape[1], dtype=np.float64)
    booster = est.get_booster()
    for feat, val in booster.get_score(importance_type="gain").items():
        gain[int(feat[1:])] = float(val)  # feature names are f0, f1, ...
    return pred, gain


def topk_indices(importance, k):
    """The k feature indices with the largest importance (ties broken by index)."""
    order = np.argsort(-np.asarray(importance, dtype=np.float64), kind="stable")
    return sorted(int(i) for i in order[:k])


# ------------------------------------------------------------------ shadow arm
def shadow_select(ptype, Xtr, ytr, Xval, yval):
    """The prototype. Returns (kept_indices, per_seed_masks, votes, wall_s). Only
    training and validation data are touched; the holdout is untouched by selection."""
    p = Xtr.shape[1]
    votes = np.zeros(p, dtype=int)
    per_seed = []
    t0 = time.time()
    for s in SHADOW_SEEDS:
        Xtr_s = shadow_augment(Xtr, s)
        Xval_s = shadow_augment(Xval, s)
        est = new_bonsai(ptype)
        est.fit(Xtr_s, ytr, eval_set=(Xval_s, yval))
        imp = est.importance("gain")
        shadow_imp = imp[p:]
        threshold = float(np.percentile(shadow_imp, SHADOW_PERCENTILE))
        selected = imp[:p] > threshold
        votes += selected.astype(int)
        per_seed.append([int(i) for i in np.where(selected)[0]])
    kept = sorted(int(i) for i in np.where(votes >= SHADOW_AGREE)[0])
    dt = time.time() - t0
    return kept, per_seed, votes.tolist(), dt


# ------------------------------------------------------------------ driver
def run_dataset(name):
    ptype = PTYPE[name]
    regime = "real_wide" if name in REAL_WIDE else "noise_injected"
    Xtr, ytr, Xval, yval, Xte, yte = rung0.load_dataset(name)
    orig_feats = int(Xtr.shape[1])

    n_injected = 0
    if regime == "noise_injected":
        Xtr = inject_noise(Xtr, NOISE_SEED)
        Xval = inject_noise(Xval, NOISE_SEED)
        Xte = inject_noise(Xte, NOISE_SEED)
        n_injected = orig_feats  # columns [orig_feats, 2*orig_feats) are noise
    total_feats = int(Xtr.shape[1])
    print(f"  rows tr/val/te={len(ytr)}/{len(yval)}/{len(yte)} "
          f"orig_feats={orig_feats} total_feats={total_feats} injected={n_injected}",
          flush=True)

    timings = {}

    def timed(key, fn):
        t0 = time.time()
        out = fn()
        timings[key] = time.time() - t0
        return out

    # Arm 1: bonsai all features (baseline + gain ranking source).
    bonsai_all_pred, bonsai_all_est = timed(
        "bonsai_all", lambda: fit_bonsai(ptype, Xtr, ytr, Xval, yval, Xte))
    bonsai_all_err = metric_error(ptype, yte, bonsai_all_pred)
    bonsai_gain = bonsai_all_est.importance("gain")

    # Arm 2: CatBoost all features.
    cat_all_err = metric_error(ptype, yte, timed(
        "cat_all", lambda: fit_catboost_all(ptype, Xtr, ytr, Xval, yval, Xte)))

    # Arm 4: the shadow prototype (fixes k for the truncation arms).
    kept, per_seed, votes, shadow_wall = shadow_select(ptype, Xtr, ytr, Xval, yval)
    timings["bonsai_shadow_select"] = shadow_wall
    k = len(kept)
    k_refit = max(k, 1)
    if k >= 1:
        shadow_pred, _ = timed("bonsai_shadow_refit", lambda: fit_bonsai(
            ptype, Xtr[:, kept], ytr, Xval[:, kept], yval, Xte[:, kept]))
        bonsai_shadow_err = metric_error(ptype, yte, shadow_pred)
    else:
        bonsai_shadow_err = float("nan")
    shadow_wall_total = timings["bonsai_shadow_select"] + timings.get(
        "bonsai_shadow_refit", 0.0)

    # Arm 3: bonsai top-k by its own gain, k = arm-4 kept count.
    topk_bonsai = topk_indices(bonsai_gain, k_refit)
    topk_pred, _ = timed("bonsai_topk_gain", lambda: fit_bonsai(
        ptype, Xtr[:, topk_bonsai], ytr, Xval[:, topk_bonsai], yval, Xte[:, topk_bonsai]))
    bonsai_topk_err = metric_error(ptype, yte, topk_pred)

    # Arm 5: CatBoost select_features down to k.
    cat_select_err = float("nan")
    cat_selected = None
    cat_select_wall = None
    cat_select_error_msg = None
    try:
        cs_pred, cat_selected, cat_select_wall = cat_select(
            ptype, Xtr, ytr, Xval, yval, Xte, k_refit)
        cat_select_err = metric_error(ptype, yte, cs_pred)
    except Exception as exc:  # record the failure honestly and continue
        cat_select_error_msg = f"{type(exc).__name__}: {exc}"
        print(f"  cat_select FAILED: {cat_select_error_msg}", flush=True)

    # Arm 6: xgboost all features + top-k by its own gain.
    xgb_all_pred, xgb_gain = timed(
        "xgb_all", lambda: fit_xgb(ptype, Xtr, ytr, Xval, yval, Xte))
    xgb_all_err = metric_error(ptype, yte, xgb_all_pred)
    topk_xgb = topk_indices(xgb_gain, k_refit)
    xgb_topk_pred, _ = timed("xgb_topk_gain", lambda: fit_xgb(
        ptype, Xtr[:, topk_xgb], ytr, Xval[:, topk_xgb], yval, Xte[:, topk_xgb]))
    xgb_topk_err = metric_error(ptype, yte, xgb_topk_pred)

    b = band(ptype, bonsai_all_err)

    row = {
        "dataset": name, "regime": regime, "problem_type": ptype,
        "metric": "rmse" if ptype == "regression" else "one_minus_auc",
        "n_train": len(ytr), "n_val": len(yval), "n_test": len(yte),
        "orig_features": orig_feats, "total_features": total_feats,
        "n_injected_noise": n_injected, "noise_seed": NOISE_SEED,
        "shadow_seeds": SHADOW_SEEDS, "shadow_percentile": SHADOW_PERCENTILE,
        "shadow_agree": SHADOW_AGREE,
        "cat_select_algo": CAT_SELECT_ALGO, "cat_select_steps": CAT_SELECT_STEPS,
        "band": b,
        # accuracy per arm
        "bonsai_all": bonsai_all_err,
        "cat_all": cat_all_err,
        "bonsai_topk_gain": bonsai_topk_err,
        "bonsai_shadow": bonsai_shadow_err,
        "cat_select": cat_select_err,
        "xgb_all": xgb_all_err,
        "xgb_topk_gain": xgb_topk_err,
        # selection outputs
        "k": k, "k_refit": k_refit,
        "shadow_kept": kept,
        "shadow_votes": votes,
        "shadow_per_seed_counts": [len(s) for s in per_seed],
        "bonsai_topk_selected": topk_bonsai,
        "cat_selected": cat_selected,
        "xgb_topk_selected": topk_xgb,
        "cat_select_error": cat_select_error_msg,
        # decompositions (lower-better; positive share = selection lowers error)
        "shadow_vs_all": bonsai_all_err - bonsai_shadow_err,
        "topk_vs_all": bonsai_all_err - bonsai_topk_err,
        "shadow_vs_topk": bonsai_topk_err - bonsai_shadow_err,
        "shadow_vs_catselect": cat_select_err - bonsai_shadow_err,
        # wall time
        "shadow_select_wall_s": shadow_wall,
        "shadow_total_wall_s": shadow_wall_total,
        "cat_select_wall_s": cat_select_wall,
        "timings_s": timings,
    }

    # Noise-recovery precision/recall for the injected regime (arms 4 and 5).
    if regime == "noise_injected":
        noise_ids = set(range(orig_feats, total_feats))
        real_ids = set(range(orig_feats))

        def recovery(selected):
            if selected is None:
                return None
            kept_set = set(selected)
            dropped = (real_ids | noise_ids) - kept_set
            noise_kept = len(kept_set & noise_ids)
            noise_dropped = len(noise_ids & dropped)
            real_kept = len(kept_set & real_ids)
            real_dropped = len(real_ids & dropped)
            n_dropped = len(dropped)
            # noise detection = dropping a noise column (positive class = noise).
            noise_recall = noise_dropped / len(noise_ids) if noise_ids else None
            noise_precision = noise_dropped / n_dropped if n_dropped else None
            return {
                "kept_total": len(kept_set),
                "noise_kept": noise_kept, "noise_dropped": noise_dropped,
                "real_kept": real_kept, "real_dropped": real_dropped,
                "noise_detection_recall": noise_recall,
                "noise_detection_precision": noise_precision,
            }

        row["recovery_shadow"] = recovery(kept)
        row["recovery_catselect"] = recovery(cat_selected)

    print(f"  bonsai_all={bonsai_all_err:.5f} cat_all={cat_all_err:.5f} "
          f"topk={bonsai_topk_err:.5f} shadow={bonsai_shadow_err:.5f} "
          f"cat_select={cat_select_err:.5f} xgb_all={xgb_all_err:.5f} "
          f"xgb_topk={xgb_topk_err:.5f}", flush=True)
    cs_wall_s = cat_select_wall if cat_select_wall is None else round(cat_select_wall, 1)
    print(f"  k={k}/{total_feats} shadow_wall={shadow_wall_total:.1f}s "
          f"cat_select_wall={cs_wall_s}s "
          f"shadow_vs_all={row['shadow_vs_all']:+.5f} band={b:.5f}", flush=True)
    if regime == "noise_injected" and row.get("recovery_shadow"):
        rs = row["recovery_shadow"]
        print(f"  shadow recovery: noise_kept={rs['noise_kept']}/{n_injected} "
              f"real_dropped={rs['real_dropped']}/{orig_feats}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default=str(REPO / "benchmarks/results/feature-selection-probe-2026-07.jsonl"),
        help="output jsonl path")
    args = ap.parse_args()
    out_path = Path(args.out)

    t_start = time.time()
    override = os.environ.get("PROBE_DATASETS", "").strip()
    if override:
        names = [d.strip() for d in override.split(",") if d.strip()]
    else:
        names = REAL_WIDE + NOISE_INJECTED

    rows = []
    for name in names:
        regime = "real_wide" if name in REAL_WIDE else "noise_injected"
        print(f"\n=== {name} ({PTYPE[name]}, {regime}) ===", flush=True)
        rows.append(run_dataset(name))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print("\n\n########## SUMMARY ##########")
    for regime in ("real_wide", "noise_injected"):
        g = [r for r in rows if r["regime"] == regime]
        if not g:
            continue
        print(f"\n=== {regime} (n={len(g)}) ===")
        for r in g:
            print(f"  {r['dataset']:32s} all={r['bonsai_all']:.5f} "
                  f"shadow={r['bonsai_shadow']:.5f} topk={r['bonsai_topk_gain']:.5f} "
                  f"cat_sel={r['cat_select']:.5f} vs_all={r['shadow_vs_all']:+.5f} "
                  f"band={r['band']:.5f} k={r['k']}/{r['total_features']}")

    print(f"\nwrote {out_path}  ({len(rows)} rows)  wall {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
