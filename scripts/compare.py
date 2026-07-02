#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=1.26",
#     "pandas>=2.2",
#     "scikit-learn>=1.4",
#     "xgboost>=2.0",
#     "lightgbm>=4.3",
#     "catboost>=1.2",
# ]
# ///
"""Compare bonsai against xgboost, lightgbm, catboost on a CSV dataset.

Reads hyperparameters from a bonsai TOML config so all four libraries see
the same settings. Writes:
    benchmarks/results/<stem>.json
    benchmarks/results/<stem>.md

Run via uv (after `make build` and `uv run scripts/fetch_toy.py`):
    uv run scripts/compare.py --config configs/california_housing.toml
"""

import argparse
import json
import pathlib
import subprocess
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import tomllib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

BONSAI_GROWERS = ("depthwise", "oblivious", "leafwise")
BONSAI_SAMPLERS = ("all_rows", "bernoulli")
BERNOULLI_P = 0.8


def load_toml(path: pathlib.Path) -> dict:
    return tomllib.loads(path.read_text())


@dataclass
class HP:
    n_iters: int
    learning_rate: float
    max_depth: int
    max_leaves: int
    min_data_in_leaf: int
    lambda_l2: float
    feature_fraction: float
    top_rate: float
    other_rate: float
    max_bin: int
    random_seed: int


def hp_from(cfg: dict) -> HP:
    tree = cfg.get("tree", {})
    booster = cfg.get("booster", {})
    bin_mapper = cfg.get("bin_mapper", {})
    sampler = cfg.get("sampler", {})
    return HP(
        n_iters=int(booster.get("n_iters", 100)),
        learning_rate=float(booster.get("learning_rate", 0.05)),
        max_depth=int(tree.get("max_depth", 6)),
        max_leaves=int(tree.get("max_leaves", 31)),
        min_data_in_leaf=int(tree.get("min_data_in_leaf", 20)),
        lambda_l2=float(tree.get("lambda_l2", 1.0)),
        feature_fraction=float(tree.get("feature_fraction", 1.0)),
        top_rate=float(sampler.get("top_rate", 0.2)),
        other_rate=float(sampler.get("other_rate", 0.1)),
        max_bin=int(bin_mapper.get("max_bin", 255)),
        random_seed=int(booster.get("random_seed", 42)),
    )


@dataclass
class Result:
    rmse: float
    fit_seconds: float
    predict_seconds: float


def rmse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def run_bonsai(config_path: pathlib.Path, hp: HP, grower: str, sampler: str,
               hp_overrides: list[str] | None = None) -> Result:
    binary = REPO_ROOT / "build" / "src" / "bonsai"
    stem = f"{grower}_{sampler}"
    model = REPO_ROOT / "build" / f"_compare_model_{stem}.msgpack"
    preds = REPO_ROOT / "build" / f"_compare_preds_{stem}.csv"
    overrides = [
        "--set", f"dispatch.grower_name={grower}",
        "--set", f"dispatch.sampler_name={sampler}",
    ]
    for ov in hp_overrides or []:
        overrides += ["--set", ov]
    if sampler == "bernoulli":
        overrides += ["--set", f"sampler.subsample={BERNOULLI_P}"]

    t0 = time.perf_counter()
    subprocess.run(
        [str(binary), "fit", "-c", str(config_path), *overrides,
         "--model", str(model)],
        check=True,
        capture_output=True,
    )
    fit_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    subprocess.run(
        [str(binary), "predict", "-c", str(config_path), *overrides,
         "--model", str(model), "--out", str(preds)],
        check=True,
        capture_output=True,
    )
    pred_s = time.perf_counter() - t1

    cfg = load_toml(config_path)
    test_path = REPO_ROOT / cfg["data"]["test"]
    test_df = pd.read_csv(test_path)
    y_test = test_df["label"].to_numpy()

    pred_df = pd.read_csv(preds)
    pred = pred_df["prediction"].to_numpy()

    return Result(rmse=rmse(pred, y_test), fit_seconds=fit_s, predict_seconds=pred_s)


def run_xgboost(train_df, test_df, hp: HP) -> Result:
    import xgboost as xgb

    feature_cols = [c for c in train_df.columns if c != "label"]
    dtrain = xgb.DMatrix(train_df[feature_cols], label=train_df["label"])
    dtest = xgb.DMatrix(test_df[feature_cols], label=test_df["label"])

    params = {
        "objective": "reg:squarederror",
        "learning_rate": hp.learning_rate,
        "max_depth": hp.max_depth,
        "max_leaves": hp.max_leaves,
        "min_child_weight": hp.min_data_in_leaf,
        "reg_lambda": hp.lambda_l2,
        "colsample_bytree": hp.feature_fraction,
        "max_bin": hp.max_bin,
        "tree_method": "hist",
        "seed": hp.random_seed,
    }
    t0 = time.perf_counter()
    booster = xgb.train(params, dtrain, num_boost_round=hp.n_iters)
    fit_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    pred = booster.predict(dtest)
    pred_s = time.perf_counter() - t1
    return Result(rmse=rmse(pred, test_df["label"].to_numpy()),
                  fit_seconds=fit_s, predict_seconds=pred_s)


def run_lightgbm(train_df, test_df, hp: HP, goss: bool = False) -> Result:
    import lightgbm as lgb

    feature_cols = [c for c in train_df.columns if c != "label"]
    dtrain = lgb.Dataset(train_df[feature_cols], label=train_df["label"])
    params = {
        "objective": "regression",
        "metric": "rmse",
        **({"data_sample_strategy": "goss",
            "top_rate": hp.top_rate,
            "other_rate": hp.other_rate} if goss else {}),
        "learning_rate": hp.learning_rate,
        "max_depth": hp.max_depth,
        "num_leaves": hp.max_leaves if hp.max_leaves > 0 else (1 << hp.max_depth) - 1,
        "min_data_in_leaf": hp.min_data_in_leaf,
        "lambda_l2": hp.lambda_l2,
        "feature_fraction": hp.feature_fraction,
        "max_bin": hp.max_bin,
        "verbose": -1,
        "seed": hp.random_seed,
    }
    t0 = time.perf_counter()
    model = lgb.train(params, dtrain, num_boost_round=hp.n_iters)
    fit_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    pred = model.predict(test_df[feature_cols])
    pred_s = time.perf_counter() - t1
    return Result(rmse=rmse(pred, test_df["label"].to_numpy()),
                  fit_seconds=fit_s, predict_seconds=pred_s)


def run_catboost(train_df, test_df, hp: HP) -> Result:
    from catboost import CatBoostRegressor

    feature_cols = [c for c in train_df.columns if c != "label"]
    model = CatBoostRegressor(
        iterations=hp.n_iters,
        learning_rate=hp.learning_rate,
        depth=hp.max_depth,
        l2_leaf_reg=hp.lambda_l2,
        rsm=hp.feature_fraction,
        random_seed=hp.random_seed,
        loss_function="RMSE",
        verbose=False,
    )
    t0 = time.perf_counter()
    model.fit(train_df[feature_cols], train_df["label"])
    fit_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    pred = model.predict(test_df[feature_cols])
    pred_s = time.perf_counter() - t1
    return Result(rmse=rmse(pred, test_df["label"].to_numpy()),
                  fit_seconds=fit_s, predict_seconds=pred_s)


def write_markdown(path: pathlib.Path, dataset: str, results: dict[str, Result]) -> None:
    width = max(len("library"), max(len(n) for n in results))
    rows = [
        f"| {'library':<{width}} | rmse   | fit_seconds | predict_seconds |",
        f"|{'-' * (width + 2)}|--------|-------------|-----------------|",
    ]
    for name, r in results.items():
        rows.append(
            f"| {name:<{width}} | {r.rmse:6.4f} | {r.fit_seconds:11.3f} | {r.predict_seconds:15.3f} |"
        )
    path.write_text(f"# {dataset} comparison\n\n" + "\n".join(rows) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=pathlib.Path)
    ap.add_argument("--name", default=None,
                    help="report stem (defaults to config stem)")
    ap.add_argument("--hp", action="append", default=[], metavar="SEC.KEY=VAL",
                    help="hyperparameter override applied to bonsai (--set) and "
                         "the mapped reference-library params, e.g. "
                         "tree.feature_fraction=0.8")
    ap.add_argument("--growers", default=",".join(BONSAI_GROWERS),
                    help="comma-separated bonsai growers to run")
    ap.add_argument("--samplers", default=",".join(BONSAI_SAMPLERS),
                    help="comma-separated bonsai samplers to run")
    args = ap.parse_args()

    cfg = load_toml(args.config)
    for ov in args.hp:
        key, _, raw = ov.partition("=")
        sec, _, name = key.partition(".")
        try:
            val: object = int(raw)
        except ValueError:
            try:
                val = float(raw)
            except ValueError:
                val = raw
        cfg.setdefault(sec, {})[name] = val
    hp = hp_from(cfg)

    train_path = REPO_ROOT / cfg["data"]["train"]
    test_path = REPO_ROOT / cfg["data"]["test"]
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    results: dict[str, Result] = {}

    for grower in args.growers.split(","):
        for sampler in args.samplers.split(","):
            label = f"bonsai ({grower}, {sampler})"
            print(f"{label} (n_iters={hp.n_iters})", flush=True)
            results[label] = run_bonsai(args.config, hp, grower, sampler, args.hp)

    print("xgboost", flush=True)
    results["xgboost"] = run_xgboost(train_df, test_df, hp)

    print("lightgbm", flush=True)
    results["lightgbm"] = run_lightgbm(train_df, test_df, hp)

    if "goss" in args.samplers.split(","):
        print("lightgbm (goss)", flush=True)
        results["lightgbm (goss)"] = run_lightgbm(train_df, test_df, hp, goss=True)

    print("catboost", flush=True)
    results["catboost"] = run_catboost(train_df, test_df, hp)

    out_dir = REPO_ROOT / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name or args.config.stem
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"

    json_path.write_text(
        json.dumps({k: vars(v) for k, v in results.items()}, indent=2)
    )
    write_markdown(md_path, stem, results)

    print(md_path.read_text())
    print(f"wrote {json_path}\nwrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
