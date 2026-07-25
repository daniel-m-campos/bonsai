# PROVENANCE NOTE: survey measurement for guide chapter 14 (selection methods
# compared on one budget ladder). Knobs mirror the matched shape of
# probe_feature_selection.py; splits come from probe_ordered_boosting_rung0.py
# (imported read-only in --export-splits mode) so every number is comparable
# to the committed probe baselines. The run mode needs only bonsai, numpy and
# scikit-learn, so it can execute on a bare node from exported npz splits.
#!/usr/bin/env python3
"""Feature-selection method survey: one ranking per method, one shared budget
ladder, one untouched holdout.

Modes:
  --export-splits DIR    local only: materialize (Xtr, ytr, Xval, yval, Xte,
                         yte, feature_names) per dataset as npz via the rung-0
                         loader (gauge venv required).
  --splits DIR --out F   run the survey from exported npz (bonsai + sklearn
                         only). PROBE_DATASETS narrows the pool for smoke.

Methods (each yields a full feature ranking, best first): corr, mutual_info,
gain, split, shap_train, shap_val, perm_val, rfe_gain, forward (small-p only).
Every ranking is evaluated by refitting at matched knobs on its top-k for each
ladder rung; error is measured on the untouched holdout.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
BONSAI_PYTHON = os.environ.get(
    "BONSAI_PYTHON", str(SCRIPTS.parent / "build-tabarena" / "python"))
sys.path.insert(0, BONSAI_PYTHON)

SEED = 42
LR = 0.05
DEPTH = 6
ES_ROUNDS = 50
ITERS = 1000
THREADS = os.cpu_count() or 8
MATCHED_PARAMS = {
    "tree.min_data_in_leaf": 20,
    "tree.lambda_l2": 1.0,
    "bin_mapper.max_bin": 255,
}

PERM_REPEATS = 5
FORWARD_K = 24
FORWARD_ITERS = 200
FORWARD_ES = 30
FORWARD_MAX_P = 100
BOOTSTRAP_REPS = 10
BOOTSTRAP_ITERS = 300
DIAG_K = 16

DATASETS = {
    "superconductivity": {"ladder": [64, 48, 32, 24, 16, 12, 8, 4],
                          "forward": True, "diagnostics": True},
    "QSAR-TID-11": {"ladder": [512, 256, 128, 64, 32],
                    "forward": False, "diagnostics": False},
}


def new_bonsai(n_iters=ITERS, es=ES_ROUNDS, n_threads=THREADS):
    import bonsai
    return bonsai.BonsaiRegressor(
        n_iters=n_iters, learning_rate=LR, max_depth=DEPTH, grower="depthwise",
        early_stopping_rounds=es, n_threads=n_threads, random_seed=SEED,
        params=dict(MATCHED_PARAMS))


def rmse(y, pred):
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def fit_eval(Xtr, ytr, Xval, yval, Xte, yte, cols=None):
    if cols is not None:
        Xtr, Xval, Xte = Xtr[:, cols], Xval[:, cols], Xte[:, cols]
    est = new_bonsai()
    est.fit(Xtr, ytr, eval_set=(Xval, yval))
    return rmse(yte, np.asarray(est.predict(Xte))), est


# ------------------------------------------------------------------ rankings
def rank_corr(D):
    t0 = time.perf_counter()
    X, y = D["Xtr"], D["ytr"]
    sd = X.std(axis=0)
    c = np.zeros(X.shape[1])
    ok = sd > 0
    Xc = X[:, ok] - X[:, ok].mean(axis=0)
    c[ok] = np.abs((Xc * (y - y.mean())[:, None]).mean(axis=0)
                   / (sd[ok] * y.std() + 1e-30))
    return np.argsort(c)[::-1], time.perf_counter() - t0


def rank_mutual_info(D):
    from sklearn.feature_selection import mutual_info_regression
    t0 = time.perf_counter()
    mi = mutual_info_regression(D["Xtr"], D["ytr"], random_state=SEED)
    return np.argsort(mi)[::-1], time.perf_counter() - t0


def rank_from_base_fit(D, base_est, base_wall):
    """gain / split / shap_train / shap_val / perm_val share one fit."""
    out = {}
    for kind in ("gain", "split"):
        t0 = time.perf_counter()
        imp = np.asarray(base_est.importance(kind))
        out[kind] = (np.argsort(imp)[::-1],
                     base_wall + time.perf_counter() - t0)
    for kind, X in (("shap_train", D["Xtr"]), ("shap_val", D["Xval"])):
        t0 = time.perf_counter()
        contribs = np.asarray(base_est.pred_contribs(X))
        imp = np.abs(contribs[:, :-1]).mean(axis=0)
        out[kind] = (np.argsort(imp)[::-1],
                     base_wall + time.perf_counter() - t0)
    t0 = time.perf_counter()
    Xval, yval = D["Xval"], D["yval"]
    base_err = rmse(yval, np.asarray(base_est.predict(Xval)))
    rng = np.random.default_rng(SEED)
    imp = np.zeros(Xval.shape[1])
    for j in range(Xval.shape[1]):
        drops = []
        for _ in range(PERM_REPEATS):
            Xp = Xval.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(rmse(yval, np.asarray(base_est.predict(Xp))) - base_err)
        imp[j] = float(np.mean(drops))
    out["perm_val"] = (np.argsort(imp)[::-1],
                       base_wall + time.perf_counter() - t0)
    return out


def rank_rfe_gain(D, ladder):
    """Backward elimination; gain recomputed each rung; drop order = ranking."""
    t0 = time.perf_counter()
    p = D["Xtr"].shape[1]
    rungs = [r for r in ladder if r < p]
    alive = np.arange(p)
    ranking_tail = []  # worst first, appended batch by batch
    for target in rungs:
        est = new_bonsai()
        est.fit(D["Xtr"][:, alive], D["ytr"],
                eval_set=(D["Xval"][:, alive], D["yval"]))
        gain = np.asarray(est.importance("gain"))
        order = np.argsort(gain)  # worst first
        n_drop = len(alive) - target
        dropped = alive[order[:n_drop]]
        ranking_tail.append(dropped[np.argsort(gain[order[:n_drop]])])
        alive = np.sort(alive[order[n_drop:]])
    est = new_bonsai()
    est.fit(D["Xtr"][:, alive], D["ytr"],
            eval_set=(D["Xval"][:, alive], D["yval"]))
    head = alive[np.argsort(np.asarray(est.importance("gain")))[::-1]]
    # Later-dropped batches rank above earlier-dropped ones; within a batch,
    # higher rung gain ranks first.
    tail = (np.concatenate([b[::-1] for b in reversed(ranking_tail)])
            if ranking_tail else np.array([], dtype=int))
    return np.concatenate([head, tail]).astype(int), time.perf_counter() - t0


_FWD = {}


def _forward_init(Xtr, ytr, Xval, yval, workers):
    _FWD.update(Xtr=Xtr, ytr=ytr, Xval=Xval, yval=yval,
                threads=max(1, (os.cpu_count() or 8) // workers))


def _forward_score(args):
    selected, cand = args
    cols = [*selected, cand]
    est = new_bonsai(n_iters=FORWARD_ITERS, es=FORWARD_ES,
                     n_threads=_FWD["threads"])
    est.fit(_FWD["Xtr"][:, cols], _FWD["ytr"],
            eval_set=(_FWD["Xval"][:, cols], _FWD["yval"]))
    return cand, rmse(_FWD["yval"], np.asarray(est.predict(_FWD["Xval"][:, cols])))


def rank_forward(D, workers):
    """Greedy forward selection with cheap candidate fits; add order = rank."""
    t0 = time.perf_counter()
    p = D["Xtr"].shape[1]
    selected, remaining = [], list(range(p))
    with ProcessPoolExecutor(
            max_workers=workers, initializer=_forward_init,
            initargs=(D["Xtr"], D["ytr"], D["Xval"], D["yval"], workers)) as ex:
        while len(selected) < min(FORWARD_K, p):
            scores = list(ex.map(_forward_score,
                                 [(selected, c) for c in remaining],
                                 chunksize=4))
            best, _ = min(scores, key=lambda s: s[1])
            selected.append(best)
            remaining.remove(best)
    rest = [c for c in np.argsort([0] * p) if c not in selected]
    return np.array(selected + rest, dtype=int), time.perf_counter() - t0


# ------------------------------------------------------------------ diagnostics
def bootstrap_stability(D):
    rng = np.random.default_rng(SEED)
    n = D["Xtr"].shape[0]
    freq = np.zeros(D["Xtr"].shape[1])
    for _ in range(BOOTSTRAP_REPS):
        idx = rng.integers(0, n, n)
        est = new_bonsai(n_iters=BOOTSTRAP_ITERS)
        est.fit(D["Xtr"][idx], D["ytr"][idx], eval_set=(D["Xval"], D["yval"]))
        top = np.argsort(np.asarray(est.importance("gain")))[::-1][:DIAG_K]
        freq[top] += 1
    return freq / BOOTSTRAP_REPS


# ------------------------------------------------------------------ modes
def export_splits(out_dir):
    import probe_ordered_boosting_rung0 as rung0
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in DATASETS:
        Xtr, ytr, Xval, yval, Xte, yte = rung0.load_dataset(name)
        names = np.array([f"f{i}" for i in range(Xtr.shape[1])])
        if name in rung0.GAUGE_NAMES:
            from tabarena.benchmark.task.openml.task_wrapper import OpenMLTaskWrapper
            w = OpenMLTaskWrapper.from_task_id(rung0.GAUGE_TASK[name])
            X, _ = w.get_X_y()
            names = np.array([str(c) for c in X.columns])
        np.savez_compressed(out / f"{name}.npz", Xtr=Xtr, ytr=ytr, Xval=Xval,
                            yval=yval, Xte=Xte, yte=yte, feature_names=names)
        print(f"exported {name}: train {Xtr.shape}, val {Xval.shape}, "
              f"test {Xte.shape}")


def run_survey(splits_dir, out_path, forward_workers):
    rows = []
    pool = [d.strip() for d in os.environ.get("PROBE_DATASETS", "").split(",")
            if d.strip()] or list(DATASETS)
    for name in pool:
        spec = DATASETS[name]
        z = np.load(Path(splits_dir) / f"{name}.npz", allow_pickle=False)
        D = {k: z[k] for k in ("Xtr", "ytr", "Xval", "yval", "Xte", "yte")}
        names = [str(s) for s in z["feature_names"]]
        p = D["Xtr"].shape[1]
        print(f"== {name} (p={p}) ==", flush=True)

        t0 = time.perf_counter()
        base_err, base_est = fit_eval(**D)
        base_wall = time.perf_counter() - t0
        rows.append({"row_type": "baseline", "dataset": name, "k": p,
                     "error": round(base_err, 5)})
        print(f"baseline all {p}: rmse {base_err:.5f}", flush=True)

        rankings = {"corr": rank_corr(D), "mutual_info": rank_mutual_info(D)}
        rankings.update(rank_from_base_fit(D, base_est, base_wall))
        rankings["rfe_gain"] = rank_rfe_gain(D, spec["ladder"])
        if spec["forward"] and p <= FORWARD_MAX_P:
            rankings["forward"] = rank_forward(D, forward_workers)

        for method, (order, wall) in rankings.items():
            rows.append({"row_type": "selection_meta", "dataset": name,
                         "method": method, "wall_s": round(wall, 2),
                         "top16": [names[i] for i in order[:DIAG_K]]})
            ladder = spec["ladder"]
            if method == "forward":
                ladder = [k for k in ladder if k <= FORWARD_K]
            for k in ladder:
                cols = np.sort(order[:k])
                err, _ = fit_eval(**D, cols=cols)
                rows.append({"row_type": "curve", "dataset": name,
                             "method": method, "k": int(k),
                             "error": round(err, 5)})
            curve = [r["error"] for r in rows if r["row_type"] == "curve"
                     and r["dataset"] == name and r["method"] == method]
            print(f"{method}: wall {wall:.1f}s, curve {curve}", flush=True)

        if spec["diagnostics"]:
            gain_top = set(int(i) for i in rankings["gain"][0][:DIAG_K])
            for method, (order, _) in rankings.items():
                s = set(int(i) for i in order[:DIAG_K])
                rows.append({"row_type": "overlap", "dataset": name,
                             "method": method, "k": DIAG_K,
                             "jaccard_vs_gain": round(
                                 len(s & gain_top) / len(s | gain_top), 3)})
            freq = bootstrap_stability(D)
            keep = np.argsort(freq)[::-1][:DIAG_K]
            rows.append({"row_type": "stability", "dataset": name,
                         "k": DIAG_K, "reps": BOOTSTRAP_REPS,
                         "mean_top_freq": round(float(freq[keep].mean()), 3),
                         "features": {names[i]: round(float(freq[i]), 2)
                                      for i in keep}})

    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-splits", metavar="DIR")
    ap.add_argument("--splits", metavar="DIR")
    ap.add_argument("--out", metavar="FILE")
    ap.add_argument("--forward-workers", type=int, default=4)
    a = ap.parse_args()
    if a.export_splits:
        export_splits(a.export_splits)
        return 0
    if not (a.splits and a.out):
        ap.error("run mode needs --splits and --out")
    run_survey(a.splits, a.out, a.forward_workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
