#!/usr/bin/env python3
# PROVENANCE NOTE: study probe for guide chapter 14. Question: at what n/p does
# noise-feature removal flip from free to a measured generalization win, and
# does a ranking fitted on a row-subsample select the same features as one
# fitted on all rows (the memory-constrained recipe)? Splits for crossover mode
# come from probe_selection_survey.py --export-splits; knobs are the survey's
# matched set, so full-size cells are directly comparable to the survey rows.
"""The n/p crossover: when does removing noise features start paying?

Modes:
  --mode crossover --splits DIR --out F
      Real datasets, row-subsampled. superconductivity is augmented with one
      shuffled copy of every column (ground-truth noise, oracle = the 81 real
      features); QSAR-TID-11 runs unaugmented at its native 1024 features.
      Arms per cell: baseline (all features), oracle (superconductivity only),
      gain top-k, shap_val top-k. Train and validation rows shrink together
      by the cell's factor; the holdout never shrinks and judges every arm.
  --mode scale --out F [--rows N --cols P --informative I]
      Synthetic at GPU scale with known informative features. Arms: baseline
      (all p), oracle (informative only), gain/shap_val top-k with the ranking
      fitted on ALL rows, and gain/shap_val top-k with the ranking fitted on a
      row SLICE (the memory-constrained recipe: rank cheap on a slice, spend
      the full row budget only on the selected columns).

Pre-registered predictions, on record before the first run:
  1. crossover, superconductivity: the oracle-vs-baseline gap is inside the
     full-size noise floor (2% relative) at n=11340 and beyond it by n=500;
     shap_val top-81 tracks the oracle within the floor at every n.
  2. crossover, QSAR: the k=256 tie against the baseline holds or becomes a
     win as n shrinks; it does not become a loss beyond the floor.
  3. scale: at n/p ~ 2000 selection buys no accuracy (all arms tie); the
     slice-ranked top-k matches the full-ranked top-k within the floor, so
     the full n x p matrix never needs to exist on one device.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

SEED = 42
LR = 0.05
DEPTH = 6
ITERS = 1000
ES_ROUNDS = 50
MIN_DATA = 20
LAMBDA = 1.0
MAX_BIN = 255
THREADS = 8

CROSSOVER = {
    "superconductivity": {"fractions": (1.0, 0.35, 0.13, 0.044), "draws": 5,
                          "augment": True, "k": 81},
    "QSAR-TID-11": {"fractions": (1.0, 0.5, 0.25), "draws": 5,
                    "augment": False, "k": 256},
}
SHAP_ROWS_CAP = 20000  # attribution rows needed for a stable ranking


def new_bonsai(grower="depthwise", n_iters=ITERS, es=ES_ROUNDS,
               n_threads=THREADS):
    import bonsai
    return bonsai.BonsaiRegressor(
        n_iters=n_iters, learning_rate=LR, max_depth=DEPTH, grower=grower,
        early_stopping_rounds=es, n_threads=n_threads, random_seed=SEED,
        params={"tree.min_data_in_leaf": MIN_DATA, "tree.lambda_l2": LAMBDA,
                "bin_mapper.max_bin": MAX_BIN})


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(p) - y) ** 2)))


def fit_eval(Xtr, ytr, Xval, yval, Xte, yte, cols, grower="depthwise"):
    cols = np.sort(np.asarray(cols))
    if cols.size == Xtr.shape[1]:  # avoid copying the full n x p matrix
        sub = (Xtr, Xval, Xte)
    else:
        sub = (Xtr[:, cols], Xval[:, cols], Xte[:, cols])
    t0 = time.perf_counter()
    est = new_bonsai(grower=grower)
    est.fit(sub[0], ytr, eval_set=(sub[1], yval))
    wall = time.perf_counter() - t0
    return est, rmse(yte, est.predict(sub[2])), wall, cols


def shap_ranking(est, Xval, cols, rng):
    """Mean |pred_contribs| on (a cap of) the validation rows, best first,
    mapped back to original column ids."""
    rows = np.arange(Xval.shape[0])
    if rows.size > SHAP_ROWS_CAP:
        rows = rng.choice(rows, SHAP_ROWS_CAP, replace=False)
    phi = np.asarray(est.pred_contribs(Xval[np.ix_(rows, cols)]))
    imp = np.abs(phi[:, :-1]).mean(axis=0)
    return cols[np.argsort(imp)[::-1]]


def gain_ranking(est, cols):
    gain = np.asarray(est.importance("gain"))
    return cols[np.argsort(gain)[::-1]]


def augment_noise(X, rng):
    """Append one shuffled copy of every column: marginals preserved, the
    X->y link destroyed. Each split shuffles independently."""
    noise = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
    return np.concatenate([X, noise], axis=1).astype(np.float32)


def run_crossover(splits_dir, out_path):
    results = []
    for name, spec in CROSSOVER.items():
        z = np.load(Path(splits_dir) / f"{name}.npz", allow_pickle=False)
        Xtr0, ytr0 = z["Xtr"].astype(np.float32), z["ytr"].astype(np.float32)
        Xval0, yval0 = z["Xval"].astype(np.float32), z["yval"].astype(np.float32)
        Xte, yte = z["Xte"].astype(np.float32), z["yte"].astype(np.float32)
        p_real = Xtr0.shape[1]
        k = spec["k"]
        for frac in spec["fractions"]:
            draws = 1 if frac == 1.0 else spec["draws"]
            for draw in range(draws):
                rng = np.random.default_rng(SEED + 1000 * draw)
                if frac == 1.0:
                    Xtr, ytr, Xval, yval = Xtr0, ytr0, Xval0, yval0
                else:
                    itr = rng.choice(len(ytr0), int(len(ytr0) * frac), replace=False)
                    ival = rng.choice(len(yval0), int(len(yval0) * frac), replace=False)
                    Xtr, ytr = Xtr0[itr], ytr0[itr]
                    Xval, yval = Xval0[ival], yval0[ival]
                if spec["augment"]:
                    Xtr = augment_noise(Xtr, rng)
                    Xval = augment_noise(Xval, rng)
                    Xte_a = augment_noise(Xte, rng)
                else:
                    Xte_a = Xte
                p = Xtr.shape[1]
                cell = {"dataset": name, "n_train": len(ytr), "p": p,
                        "frac": frac, "draw": draw}
                all_cols = np.arange(p)
                base, e_base, w_base, _ = fit_eval(Xtr, ytr, Xval, yval,
                                                  Xte_a, yte, all_cols)
                arms = {"baseline": (e_base, w_base, 0)}
                if spec["augment"]:
                    _, e, w, _ = fit_eval(Xtr, ytr, Xval, yval, Xte_a, yte,
                                          np.arange(p_real))
                    arms["oracle"] = (e, w, 0)
                for arm, ranking in (
                        ("gain_topk", gain_ranking(base, all_cols)),
                        ("shap_val_topk", shap_ranking(base, Xval, all_cols, rng))):
                    top = ranking[:k]
                    _, e, w, _ = fit_eval(Xtr, ytr, Xval, yval, Xte_a, yte, top)
                    noise_kept = int((top >= p_real).sum()) if spec["augment"] else -1
                    arms[arm] = (e, w, noise_kept)
                for arm, (e, w, nk) in arms.items():
                    results.append({**cell, "arm": arm, "error": round(e, 5),
                                    "fit_wall_s": round(w, 2),
                                    "noise_kept": nk})
                print(f"{name} n={len(ytr)} draw={draw}: " +
                      " ".join(f"{a}={v[0]:.4f}" for a, v in arms.items()),
                      flush=True)
    write_jsonl(out_path, results)


def make_synthetic(rows, cols, informative, seed, noise_sd=1.0):
    """Sparse-signal synthetic: y depends on `informative` columns through a
    Friedman-flavored mix of nonlinearities and pair interactions; every other
    column is pure noise. float32 throughout; generated in row chunks."""
    rng = np.random.default_rng(seed)
    X = np.empty((rows, cols), dtype=np.float32)
    chunk = max(1, 20_000_000 // cols)
    for lo in range(0, rows, chunk):
        hi = min(rows, lo + chunk)
        X[lo:hi] = rng.standard_normal((hi - lo, cols), dtype=np.float32)
    idx = np.arange(informative)
    w = rng.uniform(0.5, 2.0, informative).astype(np.float32)
    y = np.zeros(rows, dtype=np.float32)
    for j in idx:
        f = X[:, j]
        y += w[j] * (np.sin(f) if j % 3 == 0 else f if j % 3 == 1 else f * f * 0.3)
    for j in range(0, informative - 1, 4):
        y += 0.5 * X[:, j] * X[:, j + 1]
    y += noise_sd * rng.standard_normal(rows).astype(np.float32)
    return X, y


def run_scale(out_path, rows, cols, informative, grower, slice_rows):
    n_val, n_te = rows // 10, rows // 10
    n_tr = rows - n_val - n_te
    X, y = make_synthetic(rows, cols, informative, SEED)
    Xtr, ytr = X[:n_tr], y[:n_tr]
    Xval, yval = X[n_tr:n_tr + n_val], y[n_tr:n_tr + n_val]
    Xte, yte = X[n_tr + n_val:], y[n_tr + n_val:]
    rng = np.random.default_rng(SEED)
    k = informative
    all_cols = np.arange(cols)
    results = []
    cell = {"dataset": "synthetic", "n_train": n_tr, "p": cols,
            "informative": informative, "grower": grower,
            "slice_rows": slice_rows}

    def record(arm, e, w, top=None):
        nk = int((top >= informative).sum()) if top is not None else -1
        results.append({**cell, "arm": arm, "error": round(e, 5),
                        "fit_wall_s": round(w, 2), "noise_kept": nk})
        print(f"scale {arm}: rmse={e:.5f} wall={w:.1f}s noise_kept={nk}",
              flush=True)

    base, e, w, _ = fit_eval(Xtr, ytr, Xval, yval, Xte, yte, all_cols,
                             grower=grower)
    record("baseline", e, w)
    _, e, w, _ = fit_eval(Xtr, ytr, Xval, yval, Xte, yte,
                          np.arange(informative), grower=grower)
    record("oracle", e, w)
    for arm, ranking in (
            ("gain_topk_fullrank", gain_ranking(base, all_cols)),
            ("shap_topk_fullrank", shap_ranking(base, Xval, all_cols, rng))):
        top = ranking[:k]
        _, e, w, _ = fit_eval(Xtr, ytr, Xval, yval, Xte, yte, top,
                              grower=grower)
        record(arm, e, w, top)
    # The memory-constrained recipe: the ranking model never sees more than
    # slice_rows rows, so the full n x p matrix never has to fit anywhere.
    sl = rng.choice(n_tr, slice_rows, replace=False)
    sl_val = rng.choice(n_val, max(slice_rows // 5, 1), replace=False)
    t0 = time.perf_counter()
    ranker = new_bonsai(grower=grower)
    ranker.fit(Xtr[sl], ytr[sl], eval_set=(Xval[sl_val], yval[sl_val]))
    w_rank = time.perf_counter() - t0
    for arm, ranking in (
            ("gain_topk_slicerank", gain_ranking(ranker, all_cols)),
            ("shap_topk_slicerank",
             shap_ranking(ranker, Xval[sl_val], all_cols, rng))):
        top = ranking[:k]
        _, e, w, _ = fit_eval(Xtr, ytr, Xval, yval, Xte, yte, top,
                              grower=grower)
        record(arm, e, w + w_rank, top)
    write_jsonl(out_path, results)


def write_jsonl(out_path, results):
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {out_path} ({len(results)} rows)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("crossover", "scale", "hft"), required=True)
    ap.add_argument("--splits")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rows", type=int, default=4_000_000)
    ap.add_argument("--cols", type=int, default=2048)
    ap.add_argument("--informative", type=int, default=64)
    ap.add_argument("--grower", default="cuda_depthwise")
    ap.add_argument("--slice-rows", type=int, default=250_000)
    a = ap.parse_args()
    if a.mode == "crossover":
        run_crossover(a.splits, a.out)
    elif a.mode == "hft":
        run_hft(a.out)
    else:
        run_scale(a.out, a.rows, a.cols, a.informative, a.grower, a.slice_rows)




# ---------------------------------------------------------------- hft mode
# Question (Daniel's regime, 2026-07-27): where does a selection ranking stop
# being trustworthy on market-microstructure-shaped data, as a function of
# signal-to-noise ratio and effective sample size? The data violates the iid
# assumptions of the crossover mode three ways, each tunable:
#   - features are persistent AR(1) processes (rho 0.97), noise ones included;
#   - the target is signal + a FORWARD moving-average of width H, the overlap
#     structure of an H-row markout: adjacent rows share noise, so effective
#     n for the noise component is roughly n / H;
#   - signal weights can drift era to era (lam < 1), modeling alpha decay.
# Splits are temporal with an H-row embargo purge: train, then a future
# validation block (early stopping + any future-scoring arm), then a future
# holdout. k = the true signal count, so noise_kept is read against ground
# truth.
#
# Pre-registered predictions, on record before the first run:
#   h1. R2=0.05, H=1, no drift: rankings near-perfect (noise_kept ~ 0),
#       r2_share near 1.
#   h2. noise_kept rises with H at fixed R2 and with falling R2 at fixed H;
#       at the worst cell (R2=0.002, H=100) it exceeds 30% of top-32.
#   h3. Under drift (lam=0.7) the era-stability and future-scored rankings
#       beat raw gain on r2_share beyond the draw spread; without drift the
#       three tie (the survey's iid conclusion transfers).
#   h4. Selection never beats the all-features baseline on holdout r2 except
#       possibly at the lowest-SNR cells, where the baseline itself decays
#       toward zero r2 and the oracle keeps a small positive edge the
#       rankings mostly fail to capture.

HFT_GRID = {"r2": (0.05, 0.01, 0.002), "H": (1, 20, 100), "lam": (1.0, 0.7)}
HFT_N, HFT_P, HFT_SIG, HFT_ERAS, HFT_DRAWS = 200_000, 128, 32, 8, 2
HFT_RHO = 0.97


def make_hft(n, r2, H, lam, seed):
    rng = np.random.default_rng(seed)
    p, n_sig, eras = HFT_P, HFT_SIG, HFT_ERAS
    eps = rng.standard_normal((n, p)).astype(np.float32)
    X = np.empty((n, p), dtype=np.float32)
    X[0] = eps[0]
    c = np.float32(np.sqrt(1 - HFT_RHO * HFT_RHO))
    for t in range(1, n):
        X[t] = HFT_RHO * X[t - 1] + c * eps[t]
    base_w = (rng.choice((-1.0, 1.0), n_sig) / np.sqrt(1 + np.arange(n_sig)))
    W = [base_w]
    for _ in range(eras - 1):
        fresh = rng.choice((-1.0, 1.0), n_sig) / np.sqrt(1 + np.arange(n_sig))
        W.append(lam * W[-1] + np.sqrt(1 - lam * lam) * fresh)
    era_of = np.minimum(np.arange(n) * eras // n, eras - 1)
    G = np.empty((n, n_sig), dtype=np.float32)
    for j in range(n_sig):
        f = X[:, j]
        G[:, j] = f if j % 3 == 0 else np.tanh(f) if j % 3 == 1 else f * f - 1
    Wt = np.stack(W)[era_of]
    s = (G * Wt).sum(axis=1)
    eta = rng.standard_normal(n + H).astype(np.float32)
    u = np.convolve(eta, np.ones(H, dtype=np.float32), "valid")[:n] / np.sqrt(H)
    scale = np.sqrt(r2 / (1 - r2) * u.var() / s.var())
    y = (scale * s + u).astype(np.float32)
    return X, y


def rank_era_stability(Xtr, ytr, eras):
    """Per-era cheap fits; features ranked by mean gain-rank across eras."""
    n = len(ytr)
    ranks = np.zeros(Xtr.shape[1])
    for e in range(eras):
        lo, hi = n * e // eras, n * (e + 1) // eras
        est = new_bonsai(n_iters=200, es=30)
        cut = lo + int((hi - lo) * 0.85)
        est.fit(Xtr[lo:cut], ytr[lo:cut], eval_set=(Xtr[cut:hi], ytr[cut:hi]))
        gain = np.asarray(est.importance("gain"))
        order = np.argsort(np.argsort(gain)[::-1])
        ranks += order
    return np.argsort(ranks)


def r2_score(y, pred):
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    return float(1 - ((y - pred) ** 2).mean() / y.var())


def run_hft(out_path):
    results = []
    k, p = HFT_SIG, HFT_P
    for r2t in HFT_GRID["r2"]:
        for H in HFT_GRID["H"]:
            for lam in HFT_GRID["lam"]:
                for draw in range(HFT_DRAWS):
                    seed = SEED + 7919 * draw
                    X, y = make_hft(HFT_N, r2t, H, lam, seed)
                    n = len(y)
                    a, b = int(n * 0.7), int(n * 0.8)
                    Xtr, ytr = X[:a - H], y[:a - H]
                    Xval, yval = X[a:b - H], y[a:b - H]
                    Xte, yte = X[b:], y[b:]
                    rng = np.random.default_rng(seed)
                    cell = {"dataset": "hft", "r2_target": r2t, "H": H,
                            "lam": lam, "draw": draw}
                    all_cols = np.arange(p)

                    def run_arm(arm, cols):
                        est_, err, wall, cols_ = fit_eval(
                            Xtr, ytr, Xval, yval, Xte, yte, cols)
                        pred = est_.predict(Xte if len(cols_) == p
                                            else Xte[:, np.sort(cols_)])
                        r2h = r2_score(yte, pred)
                        nk = int((np.asarray(cols_) >= HFT_SIG).sum())
                        terc = [int(((np.asarray(cols_) >= t * k // 3)
                                     & (np.asarray(cols_) < (t + 1) * k // 3)).sum())
                                for t in range(3)]
                        results.append({**cell, "arm": arm, "r2_hold": round(r2h, 5),
                                        "noise_kept": nk if len(cols_) < p else -1,
                                        "sig_strong": terc[0], "sig_mid": terc[1],
                                        "sig_weak": terc[2],
                                        "fit_wall_s": round(wall, 1)})
                        return est_, r2h

                    base, _ = run_arm("baseline", all_cols)
                    run_arm("oracle", np.arange(k))
                    run_arm("gain_topk", gain_ranking(base, all_cols)[:k])
                    run_arm("shap_future_topk",
                            shap_ranking(base, Xval, all_cols, rng)[:k])
                    run_arm("era_stability_topk",
                            rank_era_stability(Xtr, ytr, HFT_ERAS)[:k])
                    print(f"hft r2={r2t} H={H} lam={lam} draw={draw} done",
                          flush=True)
    write_jsonl(out_path, results)


if __name__ == "__main__":
    main()
