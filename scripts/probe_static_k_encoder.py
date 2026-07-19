#!/usr/bin/env python3
"""Static-K categorical probe: price K-permutation-averaged ordered target stats.

Rung 1 of the categorical reopener (decision 80). Decision 80 priced CatBoost's
categorical machinery at a mean -0.0099 on the cat-heavy TabArena pool, 68% of its
remaining lead over bonsai_ts. CatBoost's ordered target statistics AVERAGE over
random permutations; bonsai's OrderedTargetEncoder (OTE) uses ONE ordering. This
probe asks how much of that categorical share plain K-permutation averaging
recovers, as preprocessing, at zero core cost, BEFORE anyone builds doc 17's
native engine feature.

The lever is built OUTSIDE the library, in this wrapper: for each of K row
permutations, fit the stock OTE (same config as the gauge's BonsaiTSModel, i.e.
cross=1 / keep_codes=True / prior_weight=10 / seed=0), inverse-permute its encoded
output back to row order, and average the K encodings per column. bonsai then
trains on the averaged TRAINING matrix; the validation/test matrix uses the full-
training-set statistics (OTE.transform), which are permutation-independent, so the
only thing K moves is the training-time causal encoding, exactly the CatBoost
ordered-TS averaging analogue.

Three arms on the 12 cat-heavy datasets, matched to the decision-80 protocol
(same datasets, same fold-0 lite split, same 8-fold bagged fits):
  bonsai_ts_k1  stock BonsaiTSModel, unmodified: reproduces cached bonsai_ts and
                is the K=1 rung of the curve (member 0 of every K-average is this
                same identity ordering, so the curve nests).
  bonsai_ts_k4  K=4 permutation-averaged.
  bonsai_ts_k8  K=8 permutation-averaged.

Recovery per dataset (lower metric_error is better):
  recovery = (k1_error - kK_error) / (k1_error - cat_native_error)
i.e. the fraction of the k1 -> CatBoost-native gap that K-averaging closes.

Reference metrics (cat_native, cat_ablated, bonsai_ts, bonsai) come from the
committed decision-80 pool: benchmarks/results/tabarena-cat-probe-2026-07.jsonl.
Only the three bonsai K-arms are computed here.

Environment (run inside the gauge venv; the harness is not vendored into the repo):
  TABARENA_DIR   tabarena checkout whose packages are importable (default: the
                 gauge job dir). Needs autogluon + tabarena.
  BONSAI_PY      build-tabarena/python to import the fitted bonsai (default: repo).
  POOL_JSONL     decision-80 pool rows (default: repo committed jsonl).
  PROBE_OUT_DIR  where fresh run artifacts land (default: TABARENA_DIR/tmp_scripts).
  OUT_JSONL      raw output rows (default: benchmarks/results/static-k-...jsonl).
  PROBE_DATASETS optional comma-separated dataset override (smoke).
  PROBE_KS       optional comma-separated K list override (default 1,4,8).

Method, tables and verdict: benchmarks/static-k-encoder-probe-2026-07.md.
"""

from __future__ import annotations

import gzip
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
TABARENA_DIR = Path(
    os.environ.get("TABARENA_DIR", "/Users/danielmcampos/.claude/jobs/65493dc2/tmp/tabarena")
)
BONSAI_PY = Path(os.environ.get("BONSAI_PY", str(REPO / "build-tabarena/python")))
POOL_JSONL = Path(
    os.environ.get("POOL_JSONL", str(REPO / "benchmarks/results/tabarena-cat-probe-2026-07.jsonl"))
)
PROBE_OUT_DIR = Path(os.environ.get("PROBE_OUT_DIR", str(TABARENA_DIR / "tmp_scripts")))
OUT_JSONL = Path(
    os.environ.get("OUT_JSONL", str(REPO / "benchmarks/results/static-k-encoder-probe-2026-07.jsonl"))
)

# bonsai's build (imported inside the model _fit) and the gauge's tmp_scripts (for
# the stock BonsaiTSModel we subclass). Set before importing bonsai_model.
sys.path.insert(0, str(BONSAI_PY))
sys.path.insert(0, str(TABARENA_DIR / "tmp_scripts"))
# The tabarena packages ship as a src layout under <checkout>/packages/*/src.
for pkg in sorted((TABARENA_DIR / "packages").glob("*/src")):
    sys.path.insert(0, str(pkg))

from bonsai_model import BonsaiModel, BonsaiTSModel  # noqa: E402

# ---------------------------------------------------------------------------
# The three arms. K=1 is the stock wrapper, unmodified. K>=2 replaces the single
# fit_transform with a K-permutation average, changing ONLY the training matrix's
# causal columns; everything else (config, seeds, folds, transform path) is stock.
# ---------------------------------------------------------------------------


class _BonsaiTSKMixin(BonsaiTSModel):
    """K-permutation-averaged ordered TS. Subclasses set K."""

    K = 1

    def _k_avg_fit_transform(self, X_arr, y_arr):
        from bonsai.encoding import OrderedTargetEncoder

        cats = self._cat_positions
        # Member 0 is the identity ordering: OTE at the stock config. This is
        # byte-identical to what K=1 (the stock wrapper) produces, so the K curve
        # nests (K1 subset K4 subset K8) and arm 1 is literally its first rung.
        base = OrderedTargetEncoder(cats)
        acc = base.fit_transform(X_arr, y_arr).astype(np.float64)
        n = len(y_arr)
        for k in range(1, self.K):
            perm = np.random.default_rng(k).permutation(n)
            enc = OrderedTargetEncoder(cats)
            Xk = enc.fit_transform(X_arr[perm], y_arr[perm]).astype(np.float64)
            inv = np.empty_like(Xk)
            inv[perm] = Xk  # inverse-permute rows back to original order
            acc += inv
        # base carries the full-training-set stats used by transform() on val/test;
        # those are permutation-independent, so val/test encoding is K-independent.
        self._ts = base
        return (acc / self.K).astype(np.float32)

    def _fit(self, X, y, X_val=None, y_val=None, time_limit=None, num_cpus=1, **kwargs):
        use_ts = self.problem_type in ("binary", "regression")
        if not use_ts:
            # Multiclass: plain ordinal codes, no TS, no averaging (stock path).
            return super()._fit(
                X, y, X_val=X_val, y_val=y_val, time_limit=time_limit, num_cpus=num_cpus, **kwargs
            )

        import bonsai

        params = self._get_model_params()
        n_iters = params.pop("n_iters", 1000)
        es_rounds = params.pop("early_stopping_rounds", 50)
        if time_limit is not None and time_limit < 30:
            n_iters = min(n_iters, 200)

        X_arr = self.preprocess(X, is_train=True)  # sets self._cat_positions
        y_arr = np.asarray(y)
        if self._cat_positions:
            X_arr = self._k_avg_fit_transform(X_arr, y_arr)

        eval_set = None
        if X_val is not None:
            Xv = self.preprocess(X_val)
            if self._ts is not None:
                Xv = self._ts.transform(Xv)
            eval_set = (Xv, np.asarray(y_val))

        model_cls = (
            bonsai.BonsaiRegressor if self.problem_type == "regression" else bonsai.BonsaiClassifier
        )
        est = model_cls(
            n_iters=n_iters,
            early_stopping_rounds=es_rounds if X_val is not None else 0,
            n_threads=num_cpus,
            **params,
        )
        est.fit(X_arr, y_arr, eval_set=eval_set)
        self.model = est

    @classmethod
    def config_generator(cls):
        from tabarena.utils.config_utils import ConfigGenerator

        return ConfigGenerator(model_cls=cls, manual_configs=[{}], search_space={})


class BonsaiTSK1Model(BonsaiTSModel):
    """Arm 1 / control: the stock wrapper, unmodified. Reproduces cached bonsai_ts."""

    ag_key = "BONSAI_TS_K1"
    ag_name = "BonsaiTSK1"

    @classmethod
    def config_generator(cls):
        from tabarena.utils.config_utils import ConfigGenerator

        return ConfigGenerator(model_cls=cls, manual_configs=[{}], search_space={})


class BonsaiTSK4Model(_BonsaiTSKMixin):
    ag_key = "BONSAI_TS_K4"
    ag_name = "BonsaiTSK4"
    K = 4


class BonsaiTSK8Model(_BonsaiTSKMixin):
    ag_key = "BONSAI_TS_K8"
    ag_name = "BonsaiTSK8"
    K = 8


ARMS = {1: BonsaiTSK1Model, 4: BonsaiTSK4Model, 8: BonsaiTSK8Model}


def load_pool() -> dict[str, dict]:
    """The 12 cat-heavy reference rows from the committed decision-80 pool."""
    pool: dict[str, dict] = {}
    with open(POOL_JSONL) as f:
        for line in f:
            r = json.loads(line)
            if r.get("subset") == "cat_heavy":
                pool[r["dataset"]] = r
    return pool


def run_arms(dataset_names: list[str], ks: list[int], run_name: str) -> Path:
    from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
    from tabarena.contexts import TabArenaContext

    experiments = TabArenaV0pt1ExperimentBundle(
        models=[(ARMS[k].config_generator(), 0) for k in ks],
    ).build_experiments()

    run_dir = PROBE_OUT_DIR / "experiments" / run_name
    # backend="native": in-process, synchronous, matching the gauge fold-0 lite
    # protocol and the decision-80 probe.
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
    pool = load_pool()

    override = os.environ.get("PROBE_DATASETS", "").strip()
    if override:
        selected = [d.strip() for d in override.split(",") if d.strip()]
        run_name = "static_k_smoke"
    else:
        selected = list(pool.keys())
        run_name = "static_k"

    ks = [int(x) for x in os.environ.get("PROBE_KS", "1,4,8").split(",")]

    print(f"cat_heavy pool ({len(pool)}): {list(pool.keys())}")
    print(f"running K={ks} on {len(selected)} datasets: {selected}", flush=True)

    run_dir = run_arms(selected, ks, run_name)
    fresh = {k: read_fresh(run_dir, ARMS[k].ag_name) for k in ks}

    rows = []
    for ds in selected:
        ref = pool[ds]
        if any(ds not in fresh[k] for k in ks):
            miss = [k for k in ks if ds not in fresh[k]]
            print(f"  WARN missing fresh result for {ds}: K={miss}")
            continue
        row = {
            "dataset": ds,
            "subset": "cat_heavy",
            "metric": ref["metric"],
            "problem_type": ref["problem_type"],
            "num_features": ref["num_features"],
            "num_cat_est": ref["num_cat_est"],
            "num_instances": ref["num_instances"],
            # reference metrics from the decision-80 committed pool
            "cat_native": ref["cat_native"],
            "cat_ablated": ref["cat_ablated"],
            "bonsai_ts_cached": ref["bonsai_ts"],
            "bonsai_cached": ref["bonsai"],
            "categorical_share_d80": ref["categorical_share"],
            "remaining_gap_d80": ref["remaining_gap"],
        }
        for k in ks:
            row[f"ts_k{k}"] = fresh[k][ds]["metric_error"]
            row[f"time_train_k{k}_s"] = fresh[k][ds]["time_train_s"]
        k1 = row["ts_k1"]
        # reproduction of the cached bonsai_ts by the fresh K=1 arm
        row["repro_delta"] = k1 - row["bonsai_ts_cached"]
        denom = k1 - row["cat_native"]  # k1 -> CatBoost-native gap (>0: cat is better)
        row["cat_gap_k1"] = denom
        for k in ks:
            if k == 1:
                continue
            improve = k1 - row[f"ts_k{k}"]  # >0: K-avg is better (lower error)
            row[f"improve_k{k}"] = improve
            row[f"recovery_k{k}"] = (improve / denom) if abs(denom) > 1e-12 else float("nan")
        rows.append(row)

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSONL, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    _report(rows, ks)
    print(f"\nwrote {OUT_JSONL}  ({len(rows)} rows)  wall {time.time()-t0:.1f}s")


def _report(rows: list[dict], ks: list[int]) -> None:
    print("\n=== reproduction check: fresh K=1 vs cached bonsai_ts (band +-0.002) ===")
    worst = 0.0
    for r in sorted(rows, key=lambda r: -abs(r["repro_delta"])):
        flag = "  <-- OUT OF BAND" if abs(r["repro_delta"]) > 0.002 else ""
        worst = max(worst, abs(r["repro_delta"]))
        print(f"  {r['dataset']:34s} k1={r['ts_k1']:.5f}  cached={r['bonsai_ts_cached']:.5f}  "
              f"delta={r['repro_delta']:+.5f}{flag}")
    print(f"  max |repro delta| = {worst:.5f}  ({'PASS' if worst <= 0.005 else 'INVESTIGATE'})")

    binary = [r for r in rows if r["problem_type"] == "binary"]
    for k in ks:
        if k == 1:
            continue
        print(f"\n=== recovery at K={k} (fraction of k1->cat_native gap closed) ===")
        recs = [r[f"recovery_k{k}"] for r in rows]
        recs_bin = [r[f"recovery_k{k}"] for r in binary]
        for r in sorted(rows, key=lambda r: -r.get(f"improve_k{k}", 0.0)):
            imp = r[f"improve_k{k}"]
            band = "noise" if abs(imp) <= 0.001 else ("HELPS" if imp > 0 else "HURTS")
            print(f"  {r['dataset']:34s} k1={r['ts_k1']:.5f} k{k}={r[f'ts_k{k}']:.5f}  "
                  f"improve={imp:+.5f}  recovery={r[f'recovery_k{k}']:+.3f}  [{band}]")
        print(f"  POOL mean recovery (n=12)      : {np.mean(recs):+.3f}")
        print(f"  BINARY mean recovery (n={len(binary)})     : {np.mean(recs_bin):+.3f}")
        print(f"  POOL mean improve (metric)     : {np.mean([r[f'improve_k{k}'] for r in rows]):+.5f}")
        print(f"  BINARY mean improve (metric)   : "
              f"{np.mean([r[f'improve_k{k}'] for r in binary]):+.5f}")


if __name__ == "__main__":
    main()
