#!/usr/bin/env python3
"""Learning-rate rule probe: how much of CatBoost's default-vs-default lead is the lr?

Decision 81 re-attributed CatBoost's small-data lead over bonsai to defaults and
protocol, not the ordered mechanism. This probe isolates the LEARNING-RATE slice of
that defaults residual: bonsai ships a fixed default lr of 0.05, while CatBoost
resolves its default lr per dataset from a size/iterations heuristic. Five bonsai
arms on the ordered-probe pool price what an lr rule alone could close; the CatBoost
default-arm numbers come from the committed ordered-probe jsonl, not fresh runs.

  ARM 1  bonsai_default : lr 0.05 (the reproduction arm; fidelity-gated against the
         committed ordered-probe `bonsai` column, which is deterministic).
  ARM 2  bonsai_lr_flat10 : lr 0.1, everything else identical (is-0.05-simply-wrong
         control).
  ARM 3  bonsai_oracle : per-dataset sweep lr in {0.01,0.02,0.03,0.05,0.08,0.12,0.2,
         0.3}, select on the VALIDATION split only, report test. The ceiling of any
         rule.
  ARM 4  bonsai_cat_rule : transplant CatBoost's own auto-lr. For each dataset a
         CatBoost fit at ITS defaults on the train split is stopped after one tree
         via a fit callback, the resolved learning_rate is read from
         get_all_params(), and bonsai runs at that lr. The callback stop (not
         iterations=1) matters: the iteration count enters CatBoost's auto-lr
         formula, so a literal 1-iteration fit resolves lr=0.5, not the default.
  ARM 5  bonsai_loo_rule : fit ln(lr) = a + b*ln(n_train) by least squares on the
         oracle-chosen lrs of the other 11 datasets (leave-one-out), apply to the
         held-out dataset, cycle. The honest generalization test of a size rule.

Pool, splits, seeds, and knobs are the ordered-boosting rung-0 probe's, imported
from scripts/probe_ordered_boosting_rung0.py so split identity holds by
construction: single split, 20% validation slice, early_stopping_rounds 50,
depth 6, 1000-iteration cap, single-model fits.

Environment (run inside the TabArena-Lite gauge venv; see the rung-0 probe):
  TABARENA_DIR   tabarena checkout (packages importable + curated metadata).
  BONSAI_PYTHON  bonsai build python dir (default: <repo>/build-tabarena/python).
  REF_JSONL      committed ordered-probe rows (bonsai + CatBoost default arms).
  OUT_JSONL      raw output rows (default: benchmarks/results/lr-rule-probe-...jsonl).
  PROBE_DATASETS optional comma-separated dataset override (smoke; skips the LOO arm
                 and the verdict when fewer than 4 datasets run).

Verdict logic and the evidence tables live in benchmarks/lr-rule-probe-2026-07.md.
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
sys.path.insert(0, str(REPO / "scripts"))

# Importing the rung-0 probe wires up TABARENA_DIR / BONSAI_PYTHON sys.path entries
# and hands over its loader, metric, pool tables, and knobs unchanged.
import numpy as np  # noqa: E402
import probe_ordered_boosting_rung0 as rung0  # noqa: E402

SEED = rung0.SEED
DEPTH = rung0.DEPTH
ES_ROUNDS = rung0.ES_ROUNDS
MAX_ITERS = 1000
LR_DEFAULT = 0.05
LR_FLAT10 = 0.10
LR_GRID = (0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3)

REF_JSONL = Path(
    os.environ.get(
        "REF_JSONL", str(REPO / "benchmarks/results/ordered-boosting-probe-2026-07.jsonl")
    )
)
OUT_JSONL = Path(
    os.environ.get("OUT_JSONL", str(REPO / "benchmarks/results/lr-rule-probe-2026-07.jsonl"))
)
REPRO_RTOL = 1e-9  # bonsai is deterministic; anything beyond this is flagged loudly


def fit_bonsai(ptype, lr, splits):
    """One bonsai fit at the probe knobs; returns val/test error, trees kept, wall."""
    import bonsai

    Xtr, ytr, Xval, yval, Xte, yte = splits
    cls = bonsai.BonsaiRegressor if ptype == "regression" else bonsai.BonsaiClassifier
    est = cls(
        n_iters=MAX_ITERS, learning_rate=lr, max_depth=DEPTH, grower="depthwise",
        early_stopping_rounds=ES_ROUNDS, n_threads=os.cpu_count(), random_seed=SEED,
    )
    t0 = time.time()
    est.fit(Xtr, ytr, eval_set=(Xval, yval))
    dt = time.time() - t0
    if ptype == "regression":
        pv, pt = est.predict(Xval), est.predict(Xte)
    else:
        pv, pt = est.predict_proba(Xval)[:, 1], est.predict_proba(Xte)[:, 1]
    return {
        "lr": lr,
        "val": rung0.metric_error(ptype, yval, pv),
        "test": rung0.metric_error(ptype, yte, pt),
        "trees": int(est.n_iters_),
        "time_s": dt,
    }


class _StopAfterOneTree:
    """CatBoost fit callback: returning False stops training after the iteration."""

    def after_iteration(self, info):
        return False


def resolve_cat_auto_lr(ptype, splits):
    """Resolved default learning_rate of a CatBoost fit at its defaults on this split.

    The fit is stopped after one tree by callback so `iterations` stays at the library
    default 1000 that the auto-lr heuristic keys on (a literal iterations=1 fit would
    resolve lr=0.5). eval_set is passed exactly as the ordered probe's default arm
    passed it, so the resolution context matches the committed CatBoost numbers.
    """
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool

    Xtr, ytr, Xval, yval, _, _ = splits
    cls = CatBoostRegressor if ptype == "regression" else CatBoostClassifier
    est = cls(random_seed=SEED, verbose=0, allow_writing_files=False)
    t0 = time.time()
    est.fit(Pool(Xtr, ytr), eval_set=Pool(Xval, yval), callbacks=[_StopAfterOneTree()])
    dt = time.time() - t0
    return float(est.get_all_params()["learning_rate"]), dt


def load_reference():
    ref = {}
    with open(REF_JSONL) as f:
        for line in f:
            r = json.loads(line)
            ref[r["dataset"]] = r
    return ref


def band_for(ptype, ref_bonsai):
    """Chance band per decision 55: ~2% relative for rmse, 0.001 absolute for 1-auc."""
    return 0.02 * ref_bonsai if ptype == "regression" else 0.001


def loo_fits(names, n_train, oracle_lr):
    """Per-fold (a, b) of ln(lr) = a + b*ln(n_train) on the other datasets + pooled fit."""
    x = np.log([n_train[d] for d in names])
    z = np.log([oracle_lr[d] for d in names])
    folds = {}
    for i, d in enumerate(names):
        mask = np.arange(len(names)) != i
        b, a = np.polyfit(x[mask], z[mask], 1)
        lr = float(np.exp(a + b * x[i]))
        folds[d] = {"a": float(a), "b": float(b), "lr": float(np.clip(lr, min(LR_GRID),
                                                                      max(LR_GRID)))}
    b, a = np.polyfit(x, z, 1)
    return folds, {"a": float(a), "b": float(b)}


def main():
    t_start = time.time()
    override = os.environ.get("PROBE_DATASETS", "").strip()
    if override:
        names = [d.strip() for d in override.split(",") if d.strip()]
    else:
        names = [g[0] for g in rung0.GAUGE_DATASETS] + [e[0] for e in rung0.EXTENSION_DATASETS]
    ref = load_reference()

    splits_by_ds, rows = {}, {}
    repro_fail = []
    for name in names:
        ptype = rung0.PTYPE[name]
        print(f"\n=== {name} ({ptype}) ===", flush=True)
        splits = rung0.load_dataset(name)
        splits_by_ds[name] = splits
        Xtr, ytr, _, yval, _, yte = splits
        r = ref[name]
        assert len(ytr) == r["n_train"], f"{name}: split drift, {len(ytr)} != {r['n_train']}"

        # ARM 1 + fidelity gate, before any new arm runs on this dataset.
        default = fit_bonsai(ptype, LR_DEFAULT, splits)
        delta = default["test"] - r["bonsai"]
        ok = abs(delta) <= REPRO_RTOL * max(abs(r["bonsai"]), 1e-12)
        print(f"  repro: fresh={default['test']:.10f} cached={r['bonsai']:.10f} "
              f"delta={delta:+.3e} {'PASS' if ok else '*** FAIL ***'}", flush=True)
        if not ok:
            repro_fail.append(name)

        # ARM 2.
        flat10 = fit_bonsai(ptype, LR_FLAT10, splits)

        # ARM 3: the sweep; select on VALIDATION only, report test.
        sweep = {}
        for lr in LR_GRID:
            sweep[f"{lr:g}"] = (
                default if lr == LR_DEFAULT else fit_bonsai(ptype, lr, splits)
            )
        chosen_key = min(sweep, key=lambda k: (sweep[k]["val"], sweep[k]["lr"]))
        oracle = sweep[chosen_key]

        # ARM 4: transplant CatBoost's resolved default lr.
        cat_lr, cat_resolve_t = resolve_cat_auto_lr(ptype, splits)
        cat_rule = fit_bonsai(ptype, cat_lr, splits)

        band = band_for(ptype, r["bonsai"])
        rows[name] = {
            "dataset": name,
            "source": "gauge" if name in rung0.GAUGE_NAMES else "ext",
            "problem_type": ptype,
            "metric": "rmse" if ptype == "regression" else "one_minus_auc",
            "n_train": len(ytr), "n_val": len(yval), "n_test": len(yte),
            "n_features": int(Xtr.shape[1]),
            "ref_bonsai": r["bonsai"],
            "ref_cat_default": r["cat_ordered_def"],
            "ref_catboost_lead": r["catboost_lead"],
            "band": band,
            "repro_delta": delta,
            "bonsai_default": default,
            "bonsai_lr_flat10": flat10,
            "bonsai_oracle": {**oracle, "chosen_lr": oracle["lr"]},
            "bonsai_cat_rule": {**cat_rule, "transplanted_lr": cat_lr,
                                "resolve_time_s": cat_resolve_t},
            "sweep": sweep,
        }
        print(f"  default={default['test']:.5f} flat10={flat10['test']:.5f} "
              f"oracle={oracle['test']:.5f}@lr{oracle['lr']:g} "
              f"cat_rule={cat_rule['test']:.5f}@lr{cat_lr:g} "
              f"(cat lead {r['catboost_lead']:+.5f}, band {band:.5f})", flush=True)

    # ARM 5: leave-one-out size rule on the oracle-chosen lrs.
    pooled = None
    if len(names) >= 4:
        n_train = {d: rows[d]["n_train"] for d in names}
        oracle_lr = {d: rows[d]["bonsai_oracle"]["chosen_lr"] for d in names}
        folds, pooled = loo_fits(names, n_train, oracle_lr)
        print(f"\npooled fit: ln(lr) = {pooled['a']:+.4f} {pooled['b']:+.4f}*ln(n_train)",
              flush=True)
        for name in names:
            f = folds[name]
            loo = fit_bonsai(rung0.PTYPE[name], f["lr"], splits_by_ds[name])
            rows[name]["bonsai_loo_rule"] = {**loo, "rule_lr": f["lr"], "a": f["a"],
                                             "b": f["b"]}
            print(f"  {name}: rule lr={f['lr']:.4f} (a={f['a']:+.3f}, b={f['b']:+.3f}) "
                  f"test={loo['test']:.5f}", flush=True)
    else:
        print("\n[smoke] fewer than 4 datasets: LOO arm and verdict skipped", flush=True)

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "w") as f:
        for name in names:
            f.write(json.dumps(rows[name]) + "\n")

    # ------------------------------------------------------------- verdict summary
    print("\n\n########## SUMMARY ##########")
    if repro_fail:
        print(f"*** REPRODUCTION FAILURES: {repro_fail} ***")
    else:
        print("reproduction gate: PASS on all datasets")
    arms = ["bonsai_lr_flat10", "bonsai_oracle", "bonsai_cat_rule"]
    if pooled is not None:
        arms.append("bonsai_loo_rule")
    print(f"\n{'dataset':32s} {'n_tr':>6s} {'band':>8s} {'lead':>9s} "
          f"{'default':>9s} " + " ".join(f"{a.split('bonsai_')[1]:>9s}" for a in arms))
    band_gain = {a: [] for a in arms}
    for name in names:
        r = rows[name]
        cells = []
        for a in arms:
            gain = r["bonsai_default"]["test"] - r[a]["test"]
            band_gain[a].append(gain / r["band"])
            cells.append(f"{r[a]['test']:9.5f}")
        print(f"{name:32s} {r['n_train']:6d} {r['band']:8.5f} {r['ref_catboost_lead']:+9.5f} "
              f"{r['bonsai_default']['test']:9.5f} " + " ".join(cells))
    print("\nmean gain over default, in chance-band units (>= 1.0 is beyond band):")
    for a in arms:
        print(f"  {a:18s} {np.mean(band_gain[a]):+8.3f}")

    leaders = [n for n in names
               if rows[n]["ref_catboost_lead"] > rows[n]["band"]]
    print(f"\nCatBoost-default leads beyond band on: {leaders or 'none'}")
    for n in leaders:
        r = rows[n]
        closure = (r["bonsai_default"]["test"] - r["bonsai_oracle"]["test"]) \
            / r["ref_catboost_lead"]
        print(f"  {n}: oracle closes {closure:+.1%} of the lead "
              f"(oracle lr {r['bonsai_oracle']['chosen_lr']:g})")

    if pooled is not None:
        chosen = np.array([rows[n]["bonsai_oracle"]["chosen_lr"] for n in names])
        trees = np.array([rows[n]["bonsai_oracle"]["trees"] for n in names], dtype=float)
        c = np.corrcoef(np.log(chosen), trees)[0, 1]
        print(f"\nES interaction: corr(ln chosen lr, trees at chosen lr) = {c:+.3f}")
        capped = [n for n in names
                  if rows[n]["sweep"]["0.01"]["trees"] >= MAX_ITERS]
        print(f"lr=0.01 hits the {MAX_ITERS}-iteration cap on: {capped or 'none'}")

    print(f"\nwrote {OUT_JSONL}  ({len(names)} rows)  wall {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
