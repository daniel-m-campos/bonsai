"""Multi-target regression probe: per-target trees against shared-structure trees.

The question is whether K-wide leaves (one tree per round whose split
criterion sums over the K targets) buy accuracy or time over one tree per
target per round. The reference libraries carry both shapes, so they are the
prototype: xgboost's ``multi_strategy`` toggles between
``one_output_per_tree`` and ``multi_output_tree`` on the same data, and
catboost's ``MultiRMSE`` is the shared-structure shape on symmetric trees
beside K separate ``RMSE`` fits. lightgbm and today's bonsai fit K models.

Pre-registered kill criteria, written before the first run:

- K-wide leaves are justified only if a shared-structure arm beats its own
  per-target sibling by more than 2% aRRMSE on a majority of datasets with no
  loss beyond 2% on any other.
- bonsai's K separate fits must sit inside the 2% band of xgboost's
  ``one_output_per_tree`` arm; that arm is what a K-output booster of
  per-target trees reproduces.

Knobs follow the quality campaign: 200 rounds, learning rate 0.05, depth 6,
255 bins, seed 42, a single 80/20 shuffled holdout. aRRMSE is the mean over
targets of the test RMSE divided by the RMSE of predicting the training mean.

Examples
--------
Run every dataset and arm, appending one JSON line per (dataset, arm):

    PYTHONPATH=build/python python3 scripts/probe_multitarget.py

Run one dataset and two arms:

    python3 scripts/probe_multitarget.py --datasets enb --arms xgb_per_tree,xgb_multi_tree
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "benchmarks" / "results" / "multitarget-tradeoff-2026-10.jsonl"
DATA_HOME = pathlib.Path(os.environ.get("BONSAI_PROBE_DATA", REPO_ROOT / "build" / "openml"))

N_ITERS: Final = 200
LEARNING_RATE: Final = 0.05
MAX_DEPTH: Final = 6
MAX_BIN: Final = 255
SEED: Final = 42
THREADS: Final = int(os.environ.get("BONSAI_PROBE_THREADS", "8"))
TEST_FRACTION: Final = 0.2

OPENML_IDS: Final = {
    "enb": 41478,
    "wq": 41491,
    "rf1": 41483,
    "rf2": 41484,
    "scm1d": 41485,
    "scm20d": 41486,
}
SYNTH: Final = "synth-horizons"
SYNTH_ROWS: Final = 200_000
SYNTH_FEATURES: Final = 64
SYNTH_HORIZONS: Final = 5


@dataclass
class Split:
    """One dataset's holdout split, targets as an (n, K) matrix."""

    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


@dataclass
class Row:
    """One (dataset, arm) measurement."""

    dataset: str
    arm: str
    n_train: int
    n_features: int
    n_targets: int
    arrmse: float
    rrmse: list[float]
    fit_s: float
    predict_s: float


# Public Functions =================================================================================


def main():
    """Run the probe over the chosen datasets and arms, printing a table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", default=",".join([*OPENML_IDS, SYNTH]))
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--out", type=pathlib.Path, default=RESULTS)
    args = parser.parse_args()

    rows: list[Row] = []
    for name in args.datasets.split(","):
        split = load_split(name)
        for arm in args.arms.split(","):
            row = measure(split, arm)
            rows.append(row)
            _append(args.out, row)
            print(_format(row), flush=True)
    print()
    print(table(rows))


def load_split(name: str) -> Split:
    """Fetch one dataset by name and hold out a shuffled fifth of it."""
    if name == SYNTH:
        X, y = _synth_horizons()
    else:
        X, y = _fetch_openml(name)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(X))
    n_test = int(len(X) * TEST_FRACTION)
    test, train = order[:n_test], order[n_test:]
    return Split(name, X[train], y[train], X[test], y[test])


def measure(split: Split, arm: str) -> Row:
    """Fit one arm on the split and score it."""
    fit, predict = ARMS[arm]()
    t0 = time.perf_counter()
    model = fit(split.X_train, split.y_train)
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    pred = np.asarray(predict(model, split.X_test), dtype=np.float64).reshape(split.y_test.shape)
    predict_s = time.perf_counter() - t1
    rrmse = relative_rmse(split.y_train, split.y_test, pred)
    return Row(
        dataset=split.name,
        arm=arm,
        n_train=len(split.X_train),
        n_features=int(split.X_train.shape[1]),
        n_targets=int(split.y_train.shape[1]),
        arrmse=float(np.mean(rrmse)),
        rrmse=[float(v) for v in rrmse],
        fit_s=fit_s,
        predict_s=predict_s,
    )


def relative_rmse(y_train: np.ndarray, y_test: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Per-target test RMSE over the RMSE of predicting the training mean."""
    base = np.sqrt(np.mean((y_test - y_train.mean(axis=0)) ** 2, axis=0))
    rmse = np.sqrt(np.mean((y_test - pred) ** 2, axis=0))
    return rmse / base


def table(rows: Sequence[Row]) -> str:
    """Markdown table of aRRMSE and fit seconds, datasets as columns."""
    datasets = list(dict.fromkeys(r.dataset for r in rows))
    arms = list(dict.fromkeys(r.arm for r in rows))
    by = {(r.dataset, r.arm): r for r in rows}
    head = "| arm | " + " | ".join(datasets) + " |"
    rule = "|---|" + "|".join("--:" for _ in datasets) + "|"
    lines = [head, rule]
    for arm in arms:
        cells = []
        for d in datasets:
            r = by.get((d, arm))
            cells.append("" if r is None else f"{r.arrmse:.4f} ({r.fit_s:.1f}s)")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# Arms =============================================================================================

FitFn = Callable[[np.ndarray, np.ndarray], Any]
PredictFn = Callable[[Any, np.ndarray], np.ndarray]


def xgb_arm(multi_strategy: str) -> tuple[FitFn, PredictFn]:
    """xgboost with 2-D targets under one ``multi_strategy``."""
    import xgboost as xgb

    def fit(X, y):
        model = xgb.XGBRegressor(
            n_estimators=N_ITERS,
            learning_rate=LEARNING_RATE,
            max_depth=MAX_DEPTH,
            max_bin=MAX_BIN,
            tree_method="hist",
            multi_strategy=multi_strategy,
            random_state=SEED,
            n_jobs=THREADS,
        )
        return model.fit(X, y)

    return fit, lambda model, X: model.predict(X)


def catboost_arm(loss: str) -> tuple[FitFn, PredictFn]:
    """catboost: ``MultiRMSE`` shares one symmetric tree per round, ``RMSE`` fits K models."""
    from catboost import CatBoostRegressor

    def one(X, y):
        model = CatBoostRegressor(
            loss_function=loss,
            iterations=N_ITERS,
            learning_rate=LEARNING_RATE,
            depth=MAX_DEPTH,
            border_count=MAX_BIN,
            random_seed=SEED,
            thread_count=THREADS,
            verbose=0,
            allow_writing_files=False,
        )
        return model.fit(X, y)

    if loss == "MultiRMSE":
        return one, lambda model, X: model.predict(X)
    return _per_target(one)


def lightgbm_arm() -> tuple[FitFn, PredictFn]:
    """lightgbm fits K models; it has no multi-output tree."""
    import lightgbm as lgb

    def one(X, y):
        model = lgb.LGBMRegressor(
            n_estimators=N_ITERS,
            learning_rate=LEARNING_RATE,
            max_depth=MAX_DEPTH,
            num_leaves=(1 << MAX_DEPTH) - 1,
            max_bin=MAX_BIN,
            random_state=SEED,
            n_jobs=THREADS,
            verbose=-1,
        )
        return model.fit(X, y)

    return _per_target(one)


def bonsai_arm() -> tuple[FitFn, PredictFn]:
    """bonsai today: K separate depthwise fits, the shape a K-output booster reproduces."""
    from bonsai import BonsaiRegressor

    def one(X, y):
        model = BonsaiRegressor(
            n_iters=N_ITERS,
            learning_rate=LEARNING_RATE,
            max_depth=MAX_DEPTH,
            grower="depthwise",
            max_bin=MAX_BIN,
            random_seed=SEED,
            n_threads=THREADS,
        )
        return model.fit(X, y)

    return _per_target(one)


ARMS: Final[dict[str, Callable[[], tuple[FitFn, PredictFn]]]] = {
    "xgb_per_tree": lambda: xgb_arm("one_output_per_tree"),
    "xgb_multi_tree": lambda: xgb_arm("multi_output_tree"),
    "cb_per_fit": lambda: catboost_arm("RMSE"),
    "cb_multirmse": lambda: catboost_arm("MultiRMSE"),
    "lgbm_per_fit": lightgbm_arm,
    "bonsai_per_fit": bonsai_arm,
}


# Private Functions ================================================================================


def _per_target(one: Callable[[np.ndarray, np.ndarray], Any]) -> tuple[FitFn, PredictFn]:
    """Lift a single-target fit into K fits, one per column of y."""

    def fit(X, y):
        return [one(X, y[:, k]) for k in range(y.shape[1])]

    def predict(models, X):
        return np.column_stack([m.predict(X) for m in models])

    return fit, predict


def _fetch_openml(name: str) -> tuple[np.ndarray, np.ndarray]:
    """One Mulan multi-target set from OpenML as float32 matrices."""
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(
        data_id=OPENML_IDS[name], data_home=str(DATA_HOME), as_frame=True, parser="auto"
    )
    X = bunch.data.to_numpy(dtype=np.float32)
    y = bunch.target.to_numpy(dtype=np.float32)
    return X, y.reshape(len(X), -1)


def _synth_horizons() -> tuple[np.ndarray, np.ndarray]:
    """Correlated horizon targets: one latent signal, noise growing with horizon.

    Mimics multi-horizon markup prediction: every horizon shares the same
    nonlinear signal in the features, each adds its own noise, and the
    signal-to-noise ratio is low (R2 near 0.1 at the first horizon).
    """
    rng = np.random.default_rng(SEED)
    X = rng.standard_normal((SYNTH_ROWS, SYNTH_FEATURES), dtype=np.float32)
    signal = (
        np.tanh(X[:, 0] * X[:, 1])
        + 0.5 * np.sin(2.0 * X[:, 2])
        + 0.3 * (X[:, 3] > 0.5) * X[:, 4]
        + 0.2 * X[:, 5:10].sum(axis=1)
    ).astype(np.float32)
    signal /= signal.std()
    horizons = np.arange(1, SYNTH_HORIZONS + 1, dtype=np.float32)
    scale = np.sqrt(horizons)
    noise = rng.standard_normal((SYNTH_ROWS, SYNTH_HORIZONS), dtype=np.float32).cumsum(axis=1)
    y = (signal[:, None] * scale * 0.35) + noise
    return X, y.astype(np.float32)


def _append(path: pathlib.Path, row: Row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(asdict(row)) + "\n")


def _format(row: Row) -> str:
    return (
        f"{row.dataset:>14} {row.arm:>15} K={row.n_targets:<2} n={row.n_train:<7} "
        f"aRRMSE={row.arrmse:.4f} fit={row.fit_s:7.1f}s predict={row.predict_s:.3f}s"
    )


if __name__ == "__main__":
    main()
