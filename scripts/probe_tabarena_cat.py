#!/usr/bin/env python3
"""TabArena categorical reopener probe: price CatBoost's categorical machinery.

Feature-admission step 1 (measure the benefit at zero core cost). The prior
TabArena-Lite gauge left CatBoost ahead of bonsai_ts (OrderedTargetEncoder); the
escape hatch for native categorical splits reopens only if that lead is
CATEGORICAL. This probe decomposes the lead with CatBoost's own reference-library
toggle: run CatBoost twice at the matched TabArena default (CatBoost_c1_BAG_L1,
i.e. AutoGluon CatBoost defaults) on the same fold-0 splits as the gauge:

  (a) NATIVE   categorical columns declared to CatBoost (its ordered target
               statistics / one-hot machinery is live);
  (b) ABLATED  the identical columns ordinal-encoded to plain integer numerics,
               no categorical declaration (cat_features=[]).

The one lever between the arms is a single dtype cast, so (a) minus (b) per
dataset IS CatBoost's categorical machinery priced by CatBoost itself. bonsai,
bonsai_ts, catboost(cached), xgb and lgbm come from the cached per-task gauge
CSV; only the two CatBoost arms are computed here.

Partition (from the committed curated task metadata, num_cat_est = round(
num_features * percentage_cat_features / 100)):
  PURE-NUMERIC control : num_cat_est == 0 (the toggle is 0 by construction).
  CAT-HEAVY            : num_cat_est >= 3; capped at the 12 datasets with the
                         largest cached CatBoost lead over bonsai_ts.

Environment (the TabArena-Lite harness + venv are not vendored into this repo;
run inside the gauge venv):
  TABARENA_DIR    tabarena checkout whose `packages` are importable (default: the
                  gauge job dir). Needs autogluon + catboost + tabarena.
  CACHED_RESULTS  results_per_split.csv from the prior gauge (per-task errors).
  PROBE_OUT_DIR   where fresh CatBoost run artifacts land (default: TABARENA_DIR).
  OUT_JSONL       raw output rows (default: benchmarks/results/...jsonl).
  PROBE_DATASETS  optional comma-separated dataset override (smoke).

Verdict logic and the evidence tables live in
benchmarks/tabarena-cat-probe-2026-07.md.
"""

from __future__ import annotations

import gzip
import json
import os
import pickle
import sys
import time
from pathlib import Path

TABARENA_DIR = Path(
    os.environ.get("TABARENA_DIR", "/Users/danielmcampos/.claude/jobs/65493dc2/tmp/tabarena")
)
CACHED_RESULTS = Path(
    os.environ.get(
        "CACHED_RESULTS",
        str(TABARENA_DIR / "tmp_scripts/eval/bonsai_lite_lite/results_per_split.csv"),
    )
)
PROBE_OUT_DIR = Path(os.environ.get("PROBE_OUT_DIR", str(TABARENA_DIR / "tmp_scripts")))
REPO = Path(__file__).resolve().parent.parent
OUT_JSONL = Path(
    os.environ.get("OUT_JSONL", str(REPO / "benchmarks/results/tabarena-cat-probe-2026-07.jsonl"))
)

# The tabarena packages ship as a src layout under <checkout>/packages/*/src.
for pkg in sorted((TABARENA_DIR / "packages").glob("*/src")):
    sys.path.insert(0, str(pkg))

import pandas as pd  # noqa: E402
from autogluon.tabular.models.catboost.catboost_model import CatBoostModel  # noqa: E402

CACHED_NATIVE = "CAT (default)"
CACHED_BTS = "[New] BONSAI_TS (default)"
CACHED_BONSAI = "[New] BONSAI (default)"
CACHED_XGB = "XGB (default)"
CACHED_LGBM = "GBM (default)"


def _config_generator(cls):
    from tabarena.utils.config_utils import ConfigGenerator

    # manual_configs=[{}] == the CatBoost_c1 default config (no overrides).
    return ConfigGenerator(model_cls=cls, manual_configs=[{}], search_space={})


class CatBoostNativeModel(CatBoostModel):
    """CatBoost at its TabArena default, categorical columns declared natively."""

    ag_key = "CAT_NATIVE"
    ag_name = "CatNative"

    @classmethod
    def config_generator(cls):
        return _config_generator(cls)


class CatBoostAblatedModel(CatBoostModel):
    """Same config; strip the category dtype so cat_features resolves to empty.

    The base model declares cat_features via ``select_dtypes(include='category')``
    after ``_preprocess`` has already mapped each category to an integer code.
    Casting those code columns to a plain integer dtype leaves ordinal numerics
    and hides them from the categorical path: the single-lever ablation.
    """

    ag_key = "CAT_ABLATED"
    ag_name = "CatAblated"

    def _preprocess(self, X, **kwargs):
        X = super()._preprocess(X, **kwargs)
        cats = list(X.select_dtypes(include="category").columns)
        if cats:
            X = X.copy()
            for c in cats:
                X[c] = X[c].astype("int64")
        return X

    @classmethod
    def config_generator(cls):
        return _config_generator(cls)


def build_partition() -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return (merged table, cat_heavy names, pure_numeric names)."""
    from tabarena.benchmark.task.metadata.fetch_metadata import load_curated_task_metadata

    md = load_curated_task_metadata().copy()
    md["num_cat_est"] = (
        (md["num_features"] * md["percentage_cat_features"] / 100.0).round().astype(int)
    )

    res = pd.read_csv(CACHED_RESULTS)
    wanted = [CACHED_NATIVE, CACHED_BTS, CACHED_BONSAI, CACHED_XGB, CACHED_LGBM]
    piv = (
        res[res["method"].isin(wanted)]
        .pivot_table(
            index=["dataset", "metric", "problem_type"],
            columns="method",
            values="metric_error",
            aggfunc="first",
        )
        .reset_index()
    )
    m = piv.merge(
        md[["dataset_name", "num_features", "percentage_cat_features",
            "num_cat_est", "num_instances"]],
        left_on="dataset",
        right_on="dataset_name",
        how="left",
    )
    # Relative CatBoost lead over bonsai_ts (lower error is better).
    m["cat_lead_rel"] = (m[CACHED_BTS] - m[CACHED_NATIVE]) / m[CACHED_BTS].abs()

    pure = sorted(m.loc[m["num_cat_est"] == 0, "dataset"])
    heavy_all = m[m["num_cat_est"] >= 3].sort_values("cat_lead_rel", ascending=False)
    heavy = list(heavy_all["dataset"].head(12))
    return m, heavy, pure


def run_arms(dataset_names: list[str], run_name: str) -> Path:
    from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
    from tabarena.contexts import TabArenaContext

    experiments = TabArenaV0pt1ExperimentBundle(
        models=[
            (CatBoostNativeModel.config_generator(), 0),
            (CatBoostAblatedModel.config_generator(), 0),
        ],
    ).build_experiments()

    run_dir = PROBE_OUT_DIR / "experiments" / run_name
    # backend="native" keeps everything in-process (no Ray pickling of the model
    # classes) and runs synchronously, matching the gauge's fold-0 lite protocol.
    ctx = TabArenaContext(backend="native")
    ctx.build_and_run_jobs(
        experiments,
        expname=str(run_dir),
        subset="lite",
        build_kwargs={"dataset_names": dataset_names},
        new_result_prefix="[New] ",
        debug_mode=True,
    )
    return run_dir


def read_fresh(run_dir: Path, ag_name: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for pkl in (run_dir / "data").glob(f"{ag_name}*/*/*/results.pkl"):
        with gzip.open(pkl, "rb") as f:
            r = pickle.load(f)
        name = r["task_metadata"]["name"]
        out[name] = {
            "metric_error": float(r["metric_error"]),
            "metric": r["metric"],
            "problem_type": r["problem_type"],
            "time_train_s": float(r["time_train_s"]),
        }
    return out


def main() -> None:
    t0 = time.time()
    m, heavy, pure = build_partition()

    override = os.environ.get("PROBE_DATASETS", "").strip()
    if override:
        selected = [d.strip() for d in override.split(",") if d.strip()]
        run_name = "cat_probe_smoke"
    else:
        selected = heavy + pure
        run_name = "cat_probe"

    print(f"cat_heavy({len(heavy)}) = {heavy}")
    print(f"pure_numeric({len(pure)}) = {pure}")
    print(f"running CatBoost native+ablated on {len(selected)} datasets ...", flush=True)

    run_dir = run_arms(selected, run_name)
    native = read_fresh(run_dir, "CatNative")
    ablated = read_fresh(run_dir, "CatAblated")

    rows = []
    cached = m.set_index("dataset")
    for ds in selected:
        if ds not in native or ds not in ablated:
            print(f"  WARN missing fresh result for {ds}: "
                  f"native={ds in native} ablated={ds in ablated}")
            continue
        c = cached.loc[ds]
        subset = "cat_heavy" if ds in heavy else "pure_numeric"
        row = {
            "dataset": ds,
            "subset": subset,
            "metric": native[ds]["metric"],
            "problem_type": native[ds]["problem_type"],
            "num_features": int(c["num_features"]),
            "num_cat_est": int(c["num_cat_est"]),
            "num_instances": int(c["num_instances"]),
            "cat_native": native[ds]["metric_error"],
            "cat_ablated": ablated[ds]["metric_error"],
            "cat_cached": float(c[CACHED_NATIVE]),
            "bonsai_ts": float(c[CACHED_BTS]),
            "bonsai": float(c[CACHED_BONSAI]),
            "xgb": float(c[CACHED_XGB]),
            "lgbm": float(c[CACHED_LGBM]),
            "time_train_native_s": native[ds]["time_train_s"],
            "time_train_ablated_s": ablated[ds]["time_train_s"],
        }
        # Lower metric_error is better throughout.
        row["categorical_share"] = row["cat_native"] - row["cat_ablated"]  # <0 => cats help cat
        row["remaining_gap"] = row["cat_native"] - row["bonsai_ts"]  # <0 => cat beats bts
        # negative: catboost beats bonsai_ts even without its machinery
        row["noncat_share"] = row["cat_ablated"] - row["bonsai_ts"]
        rows.append(row)

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 240)
    show = [
        "dataset", "subset", "metric", "num_cat_est",
        "cat_native", "cat_ablated", "cat_cached", "bonsai_ts",
        "categorical_share", "remaining_gap", "noncat_share",
    ]
    print("\n=== per-dataset (lower metric_error is better) ===")
    print(df[show].to_string(index=False))

    for name, g in df.groupby("subset"):
        print(f"\n=== {name} aggregate (n={len(g)}) ===")
        print(f"  mean categorical_share (native-ablated): {g['categorical_share'].mean():+.5f}")
        print(f"  mean remaining_gap    (native-bts)     : {g['remaining_gap'].mean():+.5f}")
        print(f"  mean noncat_share     (ablated-bts)    : {g['noncat_share'].mean():+.5f}")
        fidelity = (g["cat_native"] - g["cat_cached"]).abs().max()
        print(f"  native vs cached max |delta|           : {fidelity:.5f}")

    print(f"\nwrote {OUT_JSONL}  ({len(rows)} rows)  wall {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
