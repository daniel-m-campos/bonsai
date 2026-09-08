"""Grinsztajn tabular benchmark: bonsai's external quality-division standings
suite (decision 68). Four OpenML suites from "Why do tree-based models still
outperform deep learning on tabular data?" (2022): 297/298 numerical
regression/classification, 299/304 categorical regression/classification,
55 tasks total.

Protocol: the paper's "medium" setting (train truncated to 10k rows), test
capped at 50k, categorical features as ordinal codes for every library, the
CAMPAIGN knob set (bonsai.bench.params), 3 seeds per task, primary metric per
task (r2 regression / AUC binary). Deviations from the paper: fixed random
splits instead of its resampling protocol, and no per-model tuning (matched
knobs is the point). xgboost's min_child_weight follows the campaign mapping;
see decision 68's correction for the bracketing caveat.

Resumable: rows already in the output jsonl are skipped. `--device cuda`
sweeps the device plane into its own file: bonsai's three CUDA growers
and the three references on their GPU builds, so the device standings
rank like for like, and scripts/check_standings.py reads each arm
against its CPU partner and holds bonsai's inside a band.

`--regime leaf-capped` is the head to head at lightgbm's own regime: the
campaign's 63 leaves under a depth cap a 63-leaf tree cannot reach, so the
leaf count is the only cap. Only the learners a leaf count alone can cap
run there, bonsai's leafwise grower and lightgbm, into their own file.

    python -m bonsai.bench.grinsztajn out.jsonl
    python -m bonsai.bench.grinsztajn out.jsonl --report
    python -m bonsai.bench.grinsztajn --device cuda out-gpu.jsonl
    python -m bonsai.bench.grinsztajn --device cuda --regime leaf-capped out-duel.jsonl

Needs the [bench] extra (xgboost, lightgbm, catboost, scikit-learn, pandas,
openml).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import time
import warnings

import numpy as np

from bonsai.bench import metrics, params, runlog, runners
from bonsai.bench import variants as vr

SUITES = {297: "num_reg", 298: "num_clf", 299: "cat_reg", 304: "cat_clf"}
SEEDS = (0, 1, 2)
TRAIN_CAP, TEST_CAP = 10_000, 50_000
# This suite's historical short names, registered as aliases in the registry.
VARIANTS = vr.GRINSZTAJN
DEVICE_VARIANTS = {vr.Device.CPU: VARIANTS,
                   vr.Device.CUDA: vr.GRINSZTAJN_CUDA}
# The campaign knobs with the leaf count they imply spelled out, so a
# regime can hold the leaves and move the depth.
C = dict(params.CAMPAIGN,
         leaves=params.num_leaves_campaign(params.CAMPAIGN["depth"]))


@dataclasses.dataclass(frozen=True)
class Regime:
    """One matched knob set and the arms that take it on each device."""

    name: str
    knobs: dict
    arms: dict


CAMPAIGN = Regime("campaign", C, DEVICE_VARIANTS)
# lightgbm's CUDA learner grows num_leaves leaves at any depth (it applies
# no max_depth), so the campaign table cannot rank it. This regime meets it
# there: the same 63 leaves under a depth cap a 63-leaf tree cannot reach,
# the one parameter change that makes bonsai's leafwise grower fit the
# same shape of tree, and lightgbm's max_depth=-1 fits the same trees as
# this cap. Only the learners a leaf count alone can cap run here.
LEAF_CAPPED = Regime(
    "leaf-capped", dict(C, depth=params.leaf_bound_depth(C["leaves"])),
    {vr.Device.CPU: vr.GRINSZTAJN_LEAF_CAPPED,
     vr.Device.CUDA: vr.GRINSZTAJN_LEAF_CAPPED_CUDA})
REGIMES = {r.name: r for r in (CAMPAIGN, LEAF_CAPPED)}


def fit_predict(variant, Xtr, ytr, Xte, kind, knobs=C):
    """Fit one arm at the given knobs and predict the test split.

    The arm's library and device come from the variant registry, the same
    lookup the perf runners' dispatch table is keyed by, so a name this
    suite committed (its short aliases) and a canonical one both land on
    the same fit, and a `_cuda` reference arm places its library on the
    GPU the way the perf runners do. The fits themselves stay this suite's
    own: sklearn-style estimators at the regime's knobs, untimed,
    predicting rather than scoring.
    """
    v = vr.resolve(variant)
    return _FITS[v.lib](v, Xtr, ytr, Xte, kind, knobs)


def load_task(task):
    """(X, y, kind, name) for an OpenML task, floats only.

    Non-numeric columns become category codes. For classification the
    POSITIVE class is the first pandas category (a fixed convention: AUC is
    symmetric under label flip only up to 1-x, so every arm must agree).
    """
    import pandas as pd
    ds = task.get_dataset()
    X, y, _, _ = ds.get_data(target=task.target_name, dataset_format="dataframe")
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category").cat.codes
    Xn = X.to_numpy(np.float32)
    if pd.api.types.is_numeric_dtype(y):
        yn, kind = y.to_numpy(np.float32), "r2"
    else:
        yn, kind = (y == y.cat.categories[0]).to_numpy(np.float32), "auc"
    return Xn, yn, kind, ds.name


def run(out_path, variants=VARIANTS, regime=CAMPAIGN):
    """Sweep every (suite, task, seed, variant); resume by completed keys."""
    import openml
    out = pathlib.Path(out_path)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["suite"], r["dataset"], r["variant"], r["seed"]))
    host = runlog.detect_host()
    knobs = dict(regime.knobs, regime=regime.name, train_cap=TRAIN_CAP)
    for sid, sname in SUITES.items():
        suite = openml.study.get_suite(sid)
        for tid in suite.tasks:
            try:
                X, y, kind, name = load_task(openml.tasks.get_task(
                    tid, download_splits=False))
            except Exception as e:
                print(f"skip task {tid}: {e!r}", flush=True)
                continue
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                idx = rng.permutation(len(X))
                n_tr = min(TRAIN_CAP, int(len(X) * 0.8))
                tr = idx[:n_tr]
                te = idx[n_tr:n_tr + TEST_CAP]
                for v in variants:
                    if (sname, name, v, seed) in done:
                        continue
                    t0 = time.perf_counter()
                    try:
                        pred = fit_predict(v, X[tr], y[tr], X[te], kind,
                                           regime.knobs)
                        fn = metrics.auc if kind == "auc" else metrics.r2
                        value, status = fn(y[te], pred), "ok"
                    except Exception as e:
                        value, status = None, f"error: {e!r}"[:200]
                    runlog.emit_row(
                        out, division="quality", suite=sname, knobs=knobs,
                        host=host, dataset=name, task=kind, variant=v,
                        seed=seed, kind=kind, metric=kind, value=value,
                        status=status, n_train=len(tr),
                        n_features=int(X.shape[1]),
                        fit_s=round(time.perf_counter() - t0, 2))
                    shown = value if value is None else round(value, 4)
                    print(f"{sname:8s} {name[:28]:28s} {v:11s} s{seed} "
                          f"{shown}", flush=True)


def report(out_path):
    """Library standings from a results file: mean rank, wins, per suite."""
    import pandas as pd
    rows = [json.loads(x)
            for x in pathlib.Path(out_path).read_text().splitlines()]
    ok = [dict(r, value=_value(r)) for r in rows if r["status"] == "ok"]
    df = pd.DataFrame(ok)
    mean = (df.groupby(["suite", "dataset", "variant"])["value"].mean()
            .reset_index())
    mean["lib"] = mean["variant"].map(
        lambda v: vr.Lib.BONSAI if v.startswith(vr.Lib.BONSAI) else v)
    lib = (mean.groupby(["suite", "dataset", "lib"])["value"].max()
           .reset_index())
    ranks = []
    for _k, g in lib.groupby(["suite", "dataset"]):
        g = g.copy()
        g["rank"] = g["value"].rank(ascending=False)
        ranks.append(g)
    rk = pd.concat(ranks)
    print("== library mean rank (best variant per lib; lower is better) ==")
    print(rk.groupby("lib")["rank"].mean().sort_values().round(3).to_string())
    print("\n== outright wins ==")
    print(rk[rk["rank"] == 1.0].groupby("lib").size()
          .sort_values(ascending=False).to_string())
    print("\n== per suite library mean rank ==")
    print(rk.groupby(["suite", "lib"])["rank"].mean().round(3).unstack()
          .to_string())
    print("\ndatasets per suite:\n"
          + rk.groupby(["suite"])["dataset"].nunique().to_string())


def main(argv=None):
    """`grinsztajn out.jsonl` runs; `--report` prints; `--device` and
    `--regime` pick the arms and their knobs."""
    args = parse_args(argv)
    if args.report:
        report(args.out)
    else:
        regime = REGIMES[args.regime]
        run(args.out, regime.arms[args.device], regime)


def parse_args(argv=None):
    """The CLI: an output path, `--report`, the device whose arms run, and
    the regime they run at."""
    ap = argparse.ArgumentParser(prog="python -m bonsai.bench.grinsztajn")
    ap.add_argument("out", help="results jsonl; existing rows are resumed")
    ap.add_argument("--report", action="store_true",
                    help="print the library standings from `out` instead")
    ap.add_argument("--device", choices=sorted(DEVICE_VARIANTS),
                    default=vr.Device.CPU,
                    help="cpu sweeps every library on the host; cuda "
                         "sweeps every library on the GPU")
    ap.add_argument("--regime", choices=sorted(REGIMES),
                    default=CAMPAIGN.name,
                    help="campaign sweeps every library at depth 6; "
                         "leaf-capped sweeps bonsai leafwise and lightgbm "
                         "at 63 leaves with no binding depth cap")
    return ap.parse_args(argv)


# Private Functions ================================================================================

def _fit_bonsai(v, Xtr, ytr, Xte, kind, k):
    import bonsai
    grower = v.name.removeprefix("bonsai_")
    obj = "logloss" if kind == "auc" else "mse"
    m = bonsai.BonsaiRegressor(
        objective=obj, grower=grower, n_iters=k["iters"],
        learning_rate=k["lr"], max_depth=k["depth"], max_leaves=k["leaves"],
        random_seed=k["seed"], n_threads=8,
        params=params.BONSAI_CAMPAIGN_PARAMS).fit(Xtr, ytr)
    return np.asarray(m.predict(Xte))


def _fit_xgb(v, Xtr, ytr, Xte, kind, k):
    """xgboost at the regime's knobs; a cuda arm is checked after the fit
    because xgboost 3.3 drops an unserved cuda request to CPU silently."""
    import xgboost as xgb
    cls = xgb.XGBClassifier if kind == "auc" else xgb.XGBRegressor
    core = params.xgb_core(
        learning_rate=k["lr"], max_depth=k["depth"],
        min_data_in_leaf=k["min_data_in_leaf"], lambda_l2=k["lambda_l2"],
        max_bin=k["bins"], seed=k["seed"])
    core["random_state"] = core.pop("seed")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = cls(n_estimators=k["iters"], n_jobs=8, device=v.device,
                **core).fit(Xtr, ytr)
    if v.device == vr.Device.CUDA:
        runners.assert_xgb_trained_on_device(m.get_booster(), caught)
    return _predicted(m, Xte, kind)


def _fit_lgbm(v, Xtr, ytr, Xte, kind, k):
    import lightgbm as lgb
    obj = "binary" if kind == "auc" else "regression"
    p = {**params.lgbm_core(
             learning_rate=k["lr"], max_depth=k["depth"],
             num_leaves=k["leaves"], min_data_in_leaf=k["min_data_in_leaf"],
             lambda_l2=k["lambda_l2"], max_bin=k["bins"], seed=k["seed"]),
         "objective": obj, "num_iterations": k["iters"],
         "deterministic": True, "num_threads": 8, "device_type": v.device}
    m = lgb.train(p, lgb.Dataset(Xtr, label=ytr))
    return m.predict(Xte)


def _fit_catboost(v, Xtr, ytr, Xte, kind, k):
    import catboost as cb
    cls = cb.CatBoostClassifier if kind == "auc" else cb.CatBoostRegressor
    m = cls(**params.catboost_core(
                learning_rate=k["lr"], max_depth=k["depth"],
                lambda_l2=k["lambda_l2"], max_bin=k["bins"],
                seed=k["seed"], device=v.device),
            iterations=k["iters"], verbose=False, thread_count=8,
            allow_writing_files=False,
            task_type=("GPU" if v.device == vr.Device.CUDA else "CPU"),
            devices="0").fit(Xtr, ytr)
    return _predicted(m, Xte, kind)


_FITS = {vr.Lib.BONSAI: _fit_bonsai, vr.Lib.XGB: _fit_xgb,
         vr.Lib.LGBM: _fit_lgbm, vr.Lib.CATBOOST: _fit_catboost}


def _predicted(model, Xte, kind):
    """A binary task scores P(positive class); a regression scores the value.

    The positive class is column 1 because load_task encodes it as 1.0.
    """
    if kind == "auc":
        return model.predict_proba(Xte)[:, 1]
    return model.predict(Xte)


def _value(row) -> float | None:
    """Metric value across schema generations: v1 rows carry `value`, the
    pre-schema rows carried the number in `metric`."""
    v = row.get("value")
    if v is None and not isinstance(row.get("metric"), str):
        v = row.get("metric")
    return v


if __name__ == "__main__":
    main()
