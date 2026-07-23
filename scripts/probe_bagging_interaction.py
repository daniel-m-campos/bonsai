#!/usr/bin/env python3
"""Bagging-interaction admission probe: does CatBoost's small-data edge live in
the BAGGED protocol, as a randomization interaction bonsai's deterministic
ensemble cannot match?

This resolves the reopener left by the ordered-boosting rung-0 probe. That probe
found CatBoost's pure-numeric small-data lead over bonsai is real under the
TabArena gauge's 8-fold BAGGED protocol (positive cached lead on all 5 gauge
datasets) but largely evaporates under single-model fits (bonsai competitive to
better at matched knobs), and ordered boosting explained none of it. The single
surviving hypothesis: the bagged protocol INTERACTS with CatBoost's non-rate
randomization defaults (Bayesian bootstrap, random_strength) so its ensemble
members decorrelate and average better than bonsai's deterministic ones. This
probe prices that interaction at zero bonsai-core cost.

The pool, the splits, and the dataset loader are imported wholesale from the
rung-0 probe (scripts/probe_ordered_boosting_rung0.py, a completed as-run
experiment imported read-only): same 12 pure-numeric small datasets, same gauge
fold-0 / stratified-holdout splits, same metric_error convention. Knobs come from
bonsai.bench.params (catboost_core) and metrics from bonsai.bench.metrics, per the
one-source-of-truth provenance rule; a one-knob drift there has produced a false
conclusion twice.

Seven arms per dataset, two protocols:

  SINGLE  identical to the rung-0 matched arms: a 20% stratified validation slice
          (seed 42) off the train side drives early stopping; fit on the other 80%.
  BAG8    8-fold CV on the train side (KFold for regression, StratifiedKFold for
          binary, shuffle=True, random_state=42). Each fold model trains on 7/8 and
          early-stops on its held-out fold; test prediction is the mean over the 8
          models (mean probability for binary, mean prediction for regression). This
          mirrors the AutoGluon bagged protocol the gauge runs.

  1 bonsai_single       matched knobs, deterministic defaults, seed 42.
  2 bonsai_bag8         BAG8, all folds seed 42, no subsampling; members differ only
                        through fold data (stock deterministic bonsai).
  3 bonsai_bag8_rand    BAG8 with randomization from existing knobs, per fold f:
                        booster.random_seed = 42+f, bernoulli sampler,
                        sampler.subsample = 0.8, tree.feature_fraction = 0.8,
                        tree.feature_seed = 42+f.
  4 cat_single          CatBoost Plain matched, randomization at library defaults.
                        Reproduces the rung-0 "cat plain (matched)" column (harness
                        validation; the one expected difference is that catboost_core
                        completes the match with l2_leaf_reg 1.0, which rung-0 left at
                        CatBoost's default 3.0).
  5 cat_bag8            BAG8, arm-4 config (stock Bayesian bootstrap + random_strength),
                        seed 42 all folds.
  6 cat_bag8_neut       arm 5 with randomization neutralized: bootstrap_type="No",
                        random_strength=0, rsm=1.
  7 cat_bag8_def        BAG8 at CatBoost library defaults (its own boosting_type, lr,
                        iterations; only seed 42 and thread count set, early stopping
                        per fold). The gauge-reproduction arm.

Decompositions (all lower-better metric_error; positive share = the named lever
lowers error):
  bagging_gain_lib = single - bag8               (positive = bagging helps)
  interaction      = (cat_single - cat_bag8) - (bonsai_single - bonsai_bag8)  HEADLINE
  randomization_share = cat_bag8_neut - cat_bag8 (positive = stock randomization helps)
  bonsai_reach     = bonsai_bag8 - bonsai_bag8_rand (positive = existing knobs buy it)
  gauge repro      = cat_bag8_def - bonsai_bag8 on the gauge 5, vs the cached leads.

Chance band: about 2% relative of the metric for rmse, 0.001 absolute for 1-roc_auc.

Environment (the TabArena-Lite harness + venv are not vendored; run inside the gauge
venv, CatBoost 1.2.10 / scikit-learn 1.7.2):
  BONSAI_PYTHON  bonsai build python dir (a real build; a worktree may not have one).
  TABARENA_DIR   tabarena checkout (importable packages + curated metadata).
  --out          raw output jsonl (default benchmarks/results/bagging-interaction-...jsonl).
  PROBE_DATASETS optional comma-separated dataset override (smoke runs).

Verdict logic and the evidence tables live in
benchmarks/bagging-interaction-probe-2026-07.md.
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
from sklearn.model_selection import KFold, StratifiedKFold  # noqa: E402

SEED = 42
LR = 0.05
DEPTH = 6
ES_ROUNDS = 50
LAMBDA_L2 = 1.0
MIN_DATA_IN_LEAF = 20
MAX_BIN_BONSAI = 255
MAX_BIN_CAT = 254
ITERS = 1000
N_FOLDS = 8
THREADS = os.cpu_count()

GAUGE_DATASETS = rung0.GAUGE_DATASETS
EXTENSION_DATASETS = rung0.EXTENSION_DATASETS
GAUGE_NAMES = rung0.GAUGE_NAMES
PTYPE = rung0.PTYPE

# Provenance: the campaign shape as dotted bonsai config keys (min_data_in_leaf,
# lambda_l2, max_bin); these equal bonsai's defaults, so the deterministic arms
# reproduce the rung-0 bonsai baseline while stating the match explicitly.
BONSAI_MATCHED_PARAMS = {
    "tree.min_data_in_leaf": MIN_DATA_IN_LEAF,
    "tree.lambda_l2": LAMBDA_L2,
    "bin_mapper.max_bin": MAX_BIN_BONSAI,
}


def metric_error(ptype, y_true, pred):
    """Lower is better, matching TabArena and rung-0: rmse (reg) / 1-auc (binary)."""
    from bonsai.bench import metrics as bench_metrics

    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(pred, dtype=np.float64)
    if ptype == "regression":
        return bench_metrics.rmse(y, p)
    return 1.0 - bench_metrics.auc(y, p)


def band(ptype, reference):
    """Chance band: ~2% relative of the metric for rmse, 0.001 absolute for auc."""
    return 0.001 if ptype == "binary" else 0.02 * abs(reference)


def pool_mean(pool, key):
    return sum(r[key] for r in pool) / len(pool)


# ------------------------------------------------------------------ bonsai arms
def fit_bonsai(ptype, Xtr, ytr, Xval, yval, Xte, seed, randomize):
    import bonsai

    cls = bonsai.BonsaiRegressor if ptype == "regression" else bonsai.BonsaiClassifier
    params = dict(BONSAI_MATCHED_PARAMS)
    sampler = "all_rows"
    if randomize:
        sampler = "bernoulli"
        params["booster.random_seed"] = seed
        params["sampler.subsample"] = 0.8
        params["tree.feature_fraction"] = 0.8
        params["tree.feature_seed"] = seed
    est = cls(
        n_iters=ITERS, learning_rate=LR, max_depth=DEPTH, grower="depthwise",
        sampler=sampler, early_stopping_rounds=ES_ROUNDS, n_threads=THREADS,
        random_seed=seed, params=params,
    )
    est.fit(Xtr, ytr, eval_set=(Xval, yval))
    return est.predict(Xte) if ptype == "regression" else est.predict_proba(Xte)[:, 1]


# ------------------------------------------------------------------ CatBoost arms
def fit_catboost(ptype, Xtr, ytr, Xval, yval, Xte, seed, mode):
    from bonsai.bench.params import catboost_core
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool

    if mode == "defaults":
        params = dict(
            random_seed=seed, thread_count=THREADS, early_stopping_rounds=ES_ROUNDS,
            verbose=0, allow_writing_files=False,
        )
    else:
        params = dict(catboost_core(
            learning_rate=LR, max_depth=DEPTH, lambda_l2=LAMBDA_L2,
            max_bin=MAX_BIN_CAT, seed=seed, device="cpu"))
        params.update(
            boosting_type="Plain", iterations=ITERS, early_stopping_rounds=ES_ROUNDS,
            verbose=0, allow_writing_files=False, thread_count=THREADS,
        )
        if mode == "neut":
            params.update(bootstrap_type="No", random_strength=0, rsm=1)
    cls = CatBoostRegressor if ptype == "regression" else CatBoostClassifier
    est = cls(**params)
    est.fit(Pool(Xtr, ytr), eval_set=Pool(Xval, yval))
    return est.predict(Xte) if ptype == "regression" else est.predict_proba(Xte)[:, 1]


# ------------------------------------------------------------------ BAG8 protocol
def bag8(ptype, Xfull, yfull, Xte, fit_fold):
    """8-fold CV bag on the train side; each fold early-stops on its held-out slice,
    predictions averaged. fit_fold(Xt, yt, Xv, yv, fold) returns test predictions."""
    if ptype == "binary":
        splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        it = splitter.split(Xfull, yfull)
    else:
        splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        it = splitter.split(Xfull)
    preds = []
    for f, (tr_idx, val_idx) in enumerate(it):
        p = fit_fold(Xfull[tr_idx], yfull[tr_idx], Xfull[val_idx], yfull[val_idx], f)
        preds.append(np.asarray(p, dtype=np.float64))
    return np.mean(preds, axis=0)


# ------------------------------------------------------------------ driver
def run_dataset(name):
    ptype = PTYPE[name]
    is_gauge = name in GAUGE_NAMES
    Xtr, ytr, Xval, yval, Xte, yte = rung0.load_dataset(name)
    # The train side (rung-0 split it 80/20 internally); BAG8 folds the whole side.
    Xfull = np.concatenate([Xtr, Xval], axis=0)
    yfull = np.concatenate([ytr, yval], axis=0)
    print(f"  rows tr/val/te={len(ytr)}/{len(yval)}/{len(yte)} full={len(yfull)} "
          f"feats={Xtr.shape[1]}", flush=True)

    timings = {}

    def timed(key, fn):
        t0 = time.time()
        out = fn()
        timings[key] = time.time() - t0
        return out

    # SINGLE arms (identical protocol to rung-0 matched).
    bonsai_single = metric_error(ptype, yte, timed(
        "bonsai_single",
        lambda: fit_bonsai(ptype, Xtr, ytr, Xval, yval, Xte, SEED, False)))
    cat_single = metric_error(ptype, yte, timed(
        "cat_single",
        lambda: fit_catboost(ptype, Xtr, ytr, Xval, yval, Xte, SEED, "matched")))

    # BAG8 arms.
    bonsai_bag8 = metric_error(ptype, yte, timed("bonsai_bag8", lambda: bag8(
        ptype, Xfull, yfull, Xte,
        lambda Xt, yt, Xv, yv, f: fit_bonsai(ptype, Xt, yt, Xv, yv, Xte, SEED, False))))
    bonsai_bag8_rand = metric_error(ptype, yte, timed("bonsai_bag8_rand", lambda: bag8(
        ptype, Xfull, yfull, Xte,
        lambda Xt, yt, Xv, yv, f: fit_bonsai(ptype, Xt, yt, Xv, yv, Xte, SEED + f, True))))
    cat_bag8 = metric_error(ptype, yte, timed("cat_bag8", lambda: bag8(
        ptype, Xfull, yfull, Xte,
        lambda Xt, yt, Xv, yv, f: fit_catboost(ptype, Xt, yt, Xv, yv, Xte, SEED, "matched"))))
    cat_bag8_neut = metric_error(ptype, yte, timed("cat_bag8_neut", lambda: bag8(
        ptype, Xfull, yfull, Xte,
        lambda Xt, yt, Xv, yv, f: fit_catboost(ptype, Xt, yt, Xv, yv, Xte, SEED, "neut"))))
    cat_bag8_def = metric_error(ptype, yte, timed("cat_bag8_def", lambda: bag8(
        ptype, Xfull, yfull, Xte,
        lambda Xt, yt, Xv, yv, f: fit_catboost(ptype, Xt, yt, Xv, yv, Xte, SEED, "defaults"))))

    bagging_gain_bonsai = bonsai_single - bonsai_bag8
    bagging_gain_cat = cat_single - cat_bag8
    interaction = bagging_gain_cat - bagging_gain_bonsai
    randomization_share = cat_bag8_neut - cat_bag8
    bonsai_reach = bonsai_bag8 - bonsai_bag8_rand
    b = band(ptype, bonsai_single)

    row = {
        "dataset": name, "source": "gauge" if is_gauge else "ext",
        "problem_type": ptype,
        "metric": "rmse" if ptype == "regression" else "one_minus_auc",
        "n_train": len(ytr), "n_val": len(yval),
        "n_full": len(yfull), "n_test": len(yte),
        "n_features": int(Xtr.shape[1]),
        "bonsai_single": bonsai_single, "bonsai_bag8": bonsai_bag8,
        "bonsai_bag8_rand": bonsai_bag8_rand,
        "cat_single": cat_single, "cat_bag8": cat_bag8,
        "cat_bag8_neut": cat_bag8_neut, "cat_bag8_def": cat_bag8_def,
        "bagging_gain_bonsai": bagging_gain_bonsai,
        "bagging_gain_cat": bagging_gain_cat,
        "interaction": interaction,
        "randomization_share": randomization_share,
        "bonsai_reach": bonsai_reach,
        "gauge_repro_lead": cat_bag8_def - bonsai_bag8,
        "band": b,
        "interaction_in_band": bool(abs(interaction) <= b),
        "timings_s": timings,
    }
    if is_gauge:
        cached = rung0.cached_reference().get(name, {})
        cc, cb = cached.get("cat"), cached.get("bonsai")
        row["cached_cat_default"] = cc
        row["cached_bonsai_default"] = cb
        row["cached_lead"] = (cb - cc) if (cc is not None and cb is not None) else None

    print(f"  bonsai s/b8/b8r={bonsai_single:.5f}/{bonsai_bag8:.5f}/{bonsai_bag8_rand:.5f} "
          f"cat s/b8/neut/def={cat_single:.5f}/{cat_bag8:.5f}/{cat_bag8_neut:.5f}/"
          f"{cat_bag8_def:.5f}", flush=True)
    print(f"  interaction={interaction:+.5f} band={b:.5f} in_band={row['interaction_in_band']} "
          f"rand_share={randomization_share:+.5f} reach={bonsai_reach:+.5f}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default=str(REPO / "benchmarks/results/bagging-interaction-probe-2026-07.jsonl"),
        help="output jsonl path")
    args = ap.parse_args()
    out_path = Path(args.out)

    t_start = time.time()
    override = os.environ.get("PROBE_DATASETS", "").strip()
    if override:
        names = [d.strip() for d in override.split(",") if d.strip()]
    else:
        names = [g[0] for g in GAUGE_DATASETS] + [e[0] for e in EXTENSION_DATASETS]

    rows = []
    for name in names:
        ptype = PTYPE[name]
        print(f"\n=== {name} ({ptype}, {'gauge' if name in GAUGE_NAMES else 'ext'}) ===",
              flush=True)
        rows.append(run_dataset(name))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print("\n\n########## SUMMARY ##########")
    for pt in ("regression", "binary"):
        g = [r for r in rows if r["problem_type"] == pt]
        if not g:
            continue
        metric = g[0]["metric"]
        print(f"\n=== {pt} pool (n={len(g)}, metric={metric}, lower better) ===")
        print(f"{'dataset':32s} {'inter':>9s} {'band':>8s} {'inband':>6s} "
              f"{'randsh':>9s} {'reach':>9s} {'bag_c':>9s} {'bag_b':>9s}")
        for r in g:
            print(f"{r['dataset']:32s} {r['interaction']:+9.5f} {r['band']:8.5f} "
                  f"{r['interaction_in_band']!s:>6s} {r['randomization_share']:+9.5f} "
                  f"{r['bonsai_reach']:+9.5f} {r['bagging_gain_cat']:+9.5f} "
                  f"{r['bagging_gain_bonsai']:+9.5f}")
        n_in = sum(1 for r in g if r["interaction_in_band"])
        m_int = pool_mean(g, "interaction")
        print(f"  interaction in band: {n_in}/{len(g)}   MEAN interaction {m_int:+.5f}")
        print(f"  MEAN randomization_share {pool_mean(g, 'randomization_share'):+.5f}  "
              f"MEAN bonsai_reach {pool_mean(g, 'bonsai_reach'):+.5f}")
        print(f"  MEAN bagging_gain cat {pool_mean(g, 'bagging_gain_cat'):+.5f}  "
              f"bonsai {pool_mean(g, 'bagging_gain_bonsai'):+.5f}")

    n_all = len(rows)
    n_in_all = sum(1 for r in rows if r["interaction_in_band"])
    print(f"\ninteraction in band overall: {n_in_all}/{n_all}")
    gauge = [r for r in rows if r["source"] == "gauge"]
    if gauge:
        print("\ngauge reproduction (cat_bag8_def - bonsai_bag8) vs cached lead:")
        for r in gauge:
            print(f"  {r['dataset']:32s} repro={r['gauge_repro_lead']:+.5f} "
                  f"cached={r.get('cached_lead')}")
    print(f"\nwrote {out_path}  ({len(rows)} rows)  wall {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
