# /// script
# requires-python = "==3.12.*"
# dependencies = ["numpy>=1.26", "xgboost>=3.3", "scikit-learn>=1.4"]
# ///
"""Feature-admission probe: does an expectile objective earn a core slot?

xgboost 3.3 shipped reg:expectileerror (2026-07-21). Expectile loss is
w(r) * r^2 with r = y - pred and w = alpha when r > 0 else 1 - alpha: a
smooth, real-hessian sibling of the pinball loss, aimed at asymmetric
over/under-prediction costs. bonsai already ships quantile (pinball); the
question is whether expectile unlocks anything the existing surface plus a
constant shift cannot.

Pre-registered predictions (scored in the tradeoff report, hits and misses):

- P1 (math): an xgboost custom objective with grad = 2*w*(pred - y),
  hess = 2*w matches native reg:expectileerror within 1% holdout expectile
  loss at matched knobs. Validates the gradient math both ways.
- P2 (benefit): on synthetic heteroscedastic data at alpha=0.9 the expectile
  fit beats MSE + validation-tuned constant shift by >5% holdout expectile
  loss; on california housing the gap is <2% (mild heteroscedasticity, a
  constant shift is nearly enough).
- P3 (existing surface): bonsai quantile at the same alpha, shift-tuned on
  validation and scored on expectile loss, stays >5% behind the true
  expectile fit (different functional, kinked loss).
- P4 (zero-core ceiling): bonsai MSE plus two IRLS sample_weight refits
  (weights from the previous fit's residual signs) closes more than half
  the mse_shift-to-expectile gap on synthetic.

Arms per (dataset, alpha in {0.75, 0.9, 0.95}), campaign-matched knobs
(200 iters, lr 0.05, depth 6, 255 bins, seed 42), 60/20/20 split:

  xgb_native      reg:expectileerror (the reference toggle)
  xgb_custom      custom-objective harness (the math check)
  xgb_mse_shift   reg:squarederror + constant shift fitted on validation
  bonsai_mse_shift    bonsai mse + the same shift treatment
  bonsai_quant_shift  bonsai quantile(alpha) + the same shift treatment
  bonsai_irls2        bonsai mse, then 2 sample_weight refits (IRLS)

Rows append to benchmarks/results/expectile-probe-2026-07.jsonl.

Run:  PYTHONPATH=build/python uv run --no-project scripts/probe_expectile.py
      (--smoke shrinks rows and iters for a wiring check)
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

SEED = 42
ALPHAS = (0.75, 0.9, 0.95)
KNOBS = {"iters": 200, "lr": 0.05, "depth": 6, "bins": 255}
OUT = pathlib.Path("benchmarks/results/expectile-probe-2026-07.jsonl")


# ---- expectile helpers ---------------------------------------------------------


def expectile_weights(resid: np.ndarray, alpha: float) -> np.ndarray:
    return np.where(resid > 0, alpha, 1.0 - alpha)


def expectile_loss(y: np.ndarray, pred: np.ndarray, alpha: float) -> float:
    r = y - pred
    return float(np.mean(expectile_weights(r, alpha) * r * r))


def scalar_expectile(y: np.ndarray, alpha: float, iters: int = 100) -> float:
    """The alpha-expectile of a sample, by the IRLS fixed point."""
    e = float(np.mean(y))
    for _ in range(iters):
        w = expectile_weights(y - e, alpha)
        nxt = float(np.average(y, weights=w))
        if abs(nxt - e) < 1e-10:
            break
        e = nxt
    return e


def best_shift(y_val: np.ndarray, pred_val: np.ndarray, alpha: float) -> float:
    """Constant added to predictions that minimizes validation expectile
    loss: the alpha-expectile of the validation residuals."""
    return scalar_expectile(y_val - pred_val, alpha)


# ---- data ----------------------------------------------------------------------


def make_hetero(n: int, p: int, rng: np.random.Generator):
    """Signal plus noise whose scale rides on x0: conditional expectiles
    genuinely vary with x, so no constant shift can match the target."""
    X = rng.standard_normal((n, p)).astype(np.float32)
    f = X[:, 0] + 0.5 * X[:, 1] * X[:, 2] + np.sin(2.0 * X[:, 3])
    scale = np.exp(0.8 * X[:, 0])
    y = f + scale * rng.standard_normal(n)
    return X, y.astype(np.float32)


def load_datasets(smoke: bool):
    rng = np.random.default_rng(SEED)
    n = 20_000 if smoke else 200_000
    yield "synthetic_hetero", *make_hetero(n, 20, rng)
    from sklearn.datasets import fetch_california_housing

    Xc, yc = fetch_california_housing(return_X_y=True)
    yield "california", Xc.astype(np.float32), yc.astype(np.float32)


def split(X, y):
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(y))
    n_tr, n_va = int(0.6 * len(y)), int(0.2 * len(y))
    tr, va, ho = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
    return (X[tr], y[tr]), (X[va], y[va]), (X[ho], y[ho])


# ---- arms ----------------------------------------------------------------------


def fit_xgb(Xtr, ytr, iters, objective, alpha=None, custom=None):
    import xgboost as xgb

    params = {"max_depth": KNOBS["depth"], "learning_rate": KNOBS["lr"],
              "max_bin": KNOBS["bins"], "tree_method": "hist",
              "seed": SEED, "nthread": 0}
    if custom is None:
        params["objective"] = objective
        if alpha is not None:
            params["expectile_alpha"] = alpha
    else:
        # custom objectives keep the default base_score=0.5; seed it with
        # the training expectile so the comparison is about gradients
        params["base_score"] = scalar_expectile(ytr, alpha)
    dtrain = xgb.QuantileDMatrix(Xtr, label=ytr, max_bin=KNOBS["bins"])
    return xgb.train(params, dtrain, num_boost_round=iters, obj=custom)


def xgb_predict(booster, X):
    import xgboost as xgb

    return booster.predict(xgb.DMatrix(X))


def make_custom_expectile(alpha: float):
    def obj(preds: np.ndarray, dtrain) -> tuple[np.ndarray, np.ndarray]:
        r = dtrain.get_label() - preds
        w = expectile_weights(r, alpha)
        return -2.0 * w * r, 2.0 * w

    return obj


def fit_bonsai(Xtr, ytr, iters, objective="mse", alpha=None, weights=None):
    from bonsai import BonsaiRegressor

    params = {"objective.quantile_alpha": alpha} if alpha is not None else None
    est = BonsaiRegressor(n_iters=iters, learning_rate=KNOBS["lr"],
                          max_depth=KNOBS["depth"], grower="depthwise",
                          max_bin=KNOBS["bins"], objective=objective,
                          random_seed=SEED, params=params)
    est.fit(Xtr, ytr, sample_weight=weights)
    return est


def run_arms(name, Xtr, ytr, Xva, yva, Xho, yho, alpha, iters):
    """Yields (arm, holdout predictions, fit seconds)."""
    t0 = time.perf_counter()
    native = fit_xgb(Xtr, ytr, iters, "reg:expectileerror", alpha=alpha)
    yield "xgb_native", xgb_predict(native, Xho), time.perf_counter() - t0

    t0 = time.perf_counter()
    custom = fit_xgb(Xtr, ytr, iters, None, alpha=alpha,
                     custom=make_custom_expectile(alpha))
    yield "xgb_custom", xgb_predict(custom, Xho), time.perf_counter() - t0

    t0 = time.perf_counter()
    mse = fit_xgb(Xtr, ytr, iters, "reg:squarederror")
    shift = best_shift(yva, xgb_predict(mse, Xva), alpha)
    yield "xgb_mse_shift", xgb_predict(mse, Xho) + shift, time.perf_counter() - t0

    t0 = time.perf_counter()
    bmse = fit_bonsai(Xtr, ytr, iters)
    shift = best_shift(yva, bmse.predict(Xva), alpha)
    yield "bonsai_mse_shift", bmse.predict(Xho) + shift, time.perf_counter() - t0

    t0 = time.perf_counter()
    bq = fit_bonsai(Xtr, ytr, iters, objective="quantile", alpha=alpha)
    shift = best_shift(yva, bq.predict(Xva), alpha)
    yield "bonsai_quant_shift", bq.predict(Xho) + shift, time.perf_counter() - t0

    # IRLS: refit mse with weights from the previous fit's residual signs.
    # Two refits, then no shift: the weights already encode the asymmetry.
    t0 = time.perf_counter()
    model = bmse
    for _ in range(2):
        w = expectile_weights(ytr - model.predict(Xtr), alpha)
        model = fit_bonsai(Xtr, ytr, iters, weights=w)
    yield "bonsai_irls2", model.predict(Xho), time.perf_counter() - t0


# ---- driver --------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    iters = 20 if args.smoke else KNOBS["iters"]

    import xgboost

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, X, y in load_datasets(args.smoke):
        (Xtr, ytr), (Xva, yva), (Xho, yho) = split(X, y)
        for alpha in ALPHAS:
            for arm, pred, secs in run_arms(name, Xtr, ytr, Xva, yva,
                                            Xho, yho, alpha, iters):
                r = yho - pred
                row = {
                    "probe": "expectile", "dataset": name, "alpha": alpha,
                    "arm": arm, "iters": iters,
                    "expectile_loss": round(expectile_loss(yho, pred, alpha), 6),
                    # identification gap: 0 at the true conditional expectile
                    "calib_gap": round(float(np.mean(
                        expectile_weights(r, alpha) * r)), 6),
                    "fit_s": round(secs, 2),
                    "xgboost": xgboost.__version__,
                }
                rows.append(row)
                with out.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(f"{name:18s} a={alpha:.2f} {arm:20s} "
                      f"eloss={row['expectile_loss']:.5f} "
                      f"calib={row['calib_gap']:+.5f} {secs:5.1f}s")

    # quick verdict lines against the pre-registered bars
    for name in {r["dataset"] for r in rows}:
        for alpha in ALPHAS:
            cell = {r["arm"]: r["expectile_loss"] for r in rows
                    if r["dataset"] == name and r["alpha"] == alpha}
            base = cell["xgb_native"]
            print(f"[{name} a={alpha}] native={base:.5f} " + " ".join(
                f"{a}={cell[a] / base:.3f}x" for a in
                ("xgb_custom", "xgb_mse_shift", "bonsai_mse_shift",
                 "bonsai_quant_shift", "bonsai_irls2")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
