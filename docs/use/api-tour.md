# The API in one read

bonsai's entire API follows from three facts. Everything else is detail.

1. **There are two layers over one engine.** Scikit-learn-shaped estimators (`BonsaiRegressor`, `BonsaiClassifier`) for pipelines and quick work, and an explicit layer (`train`, `Dataset`, `Model`) when you want full control. Both call the same C++ training path the CLI uses.
2. **There is one configuration system.** Every knob is a dotted key like `tree.max_depth` or `dispatch.grower_name`. The same keys work as `params` pairs in Python, as `--set` overrides on the CLI, and as sections in a TOML file. `bonsai.default_config_toml()` prints all of them with defaults.
3. **There is one model format.** `.msgpack` files round-trip everywhere: a model trained in Python predicts from the CLI, and vice versa. A model trained on CPU is byte-identical across runs, thread counts, and CPU architectures, so that file is a reproducible artifact rather than a snapshot of one machine. GPU training does not carry that guarantee: device histograms accumulate under atomics, so the same fit writes different bytes each run, and what the device plane offers instead is a measured run-to-run spread ([the contract](../design/determinism.md)).

## Install

One command from PyPI; [Install](install.md) has the full story (wheel matrix, GPU support, docker, extras) and [Building from source](building.md) covers everything past a wheel:

```
pip install bonsai-gbt
```

## Which API when

Three call shapes over one engine. The question that picks between them is how many fits come out of one ingest.

**Estimators, for scikit-learn interop.** `BonsaiRegressor` and `BonsaiClassifier` are what a `Pipeline`, a `GridSearchCV`, a `cross_val_score`, or a `clone` expects to be handed. Reach for them when something else in your stack holds the model.

**`Dataset` plus `train`, for repeat fits.** `bonsai.Dataset(X, y, device=...)` runs the binning pass once at construction and every later `train(pairs, ds)` skips it, bit-identical to training from the arrays. This is the shape for hyperparameter search, multi-seed ensembling, production refits on a warm dataset, and any serving path where one ingest feeds many fits. On GPU it is also where the matrix uploads once instead of once per fit.

**Fused `train(pairs, X, y)`, for one-shots.** One fit, one array, no object to keep. The binning happens inside the call and goes away with it.

The estimator layer is the sklearn-shaped surface, not the primary one. Everything it does, the explicit layer does with less ceremony; what it adds is the contract other libraries call.

## The estimator layer

If you know scikit-learn, you know this layer:

```{.python .run}
import bonsai
import numpy as np

rng = np.random.default_rng(0)
X = rng.random((600, 8), dtype=np.float32)
y = (X[:, 0] * 2 + rng.normal(0, 0.2, 600)).astype(np.float32)
X_train, y_train = X[:400], y[:400]
X_valid, y_valid = X[400:], y[400:]
X_test, w = X[400:], np.ones(400, dtype=np.float32)

model = bonsai.BonsaiRegressor(
    n_iters=200, learning_rate=0.05, grower="leafwise",
    early_stopping_rounds=20,
)
model.fit(X_train, y_train, eval_set=(X_valid, y_valid), sample_weight=w)
pred = model.predict(X_test)
model.save("model.msgpack")
```

`BonsaiClassifier` handles binary and multiclass the sklearn way: arbitrary label values in, `classes_` out, `predict_proba` returning `(n, K)` probabilities.

Both estimators duck-type the full sklearn contract (`get_params`, `clone`, `Pipeline`, `GridSearchCV`, `cross_val_score`, pickling) without importing sklearn at runtime. Constructor arguments are bonsai's own names and nothing else, so the signature always tells you which vocabulary you are speaking; a parameter dict written for another library goes through `bonsai.interop` first.

Anything without a first-class kwarg goes through `params`, using the dotted keys:

```python
bonsai.BonsaiRegressor(params={"tree.lambda_l1": 0.5, "sampler.subsample": 0.8})
```

## Bringing a config from another library

`bonsai.interop` translates a parameter dict written for XGBoost, LightGBM, or CatBoost into the pairs `train` and `params` take. It is the only place in the repo that knows those names, so the benchmark harness, the estimators, and these docs cannot drift apart:

```{.python .run}
import bonsai

lgbm_config = {
    "objective": "regression",
    "num_leaves": 63,
    "max_depth": -1,
    "learning_rate": 0.05,
    "min_data_in_leaf": 20,
    "lambda_l2": 1.0,
    "verbose": -1,
}
for key, value in sorted(bonsai.interop.from_lightgbm(lgbm_config)):
    print(f"{key} = {value}")
```

Two lines of that output are the whole reason the module exists. `max_depth=-1` is LightGBM's uncapped growth and bonsai has no `-1` sentinel, so it arrives as `tree.max_depth = 255`, a depth no histogram tree reaches; the budget that actually binds a leaf-wise fit is `num_leaves`, which becomes `tree.max_leaves`. And `verbose` is logging rather than a knob, so it is dropped instead of translated into something that looks like one.

An unrecognized parameter raises by default, with every one of them named in the message, because a quietly discarded regularizer is a quietly different model. `strict=False` drops them instead, which is what you want while porting a large dict one knob at a time.

`to_xgboost`, `to_lightgbm`, and `to_catboost` run the other direction, turning a bonsai config into a reference library's dict. That is how the benchmark harness builds matched arms.

Translation is not equivalence, and each mapping documents where its two sides part company: XGBoost's `min_child_weight` counts hessian mass rather than rows, and CatBoost's `border_count` counts splits where `max_bin` counts bins. `help(bonsai.interop)` prints the caveats; read them before trusting a translated config to reproduce a number.

## The explicit layer

`train` takes the dotted keys directly and returns a `Model`:

```python
model = bonsai.train(
    [("dispatch.grower_name", "levelwise"), ("booster.n_iters", "200")],
    X, y,
    eval_set=(Xv, yv),
)
```

For hyperparameter searches and cross-validation, bin once and train many times. A `Dataset` runs the binning pass at construction and every subsequent `train` call skips it, bit-identical to training from the arrays:

```python
ds = bonsai.Dataset(X, y, max_bin=255)
for params in grid:
    m = bonsai.train(params, ds, eval_set=(Xv, yv))
```

Bin settings are sealed into the `Dataset`; a `bin_mapper.*` override in `params` or a config file raises instead of silently diverging.

The binning pass runs on the host by default. For GPU work, say so at construction: `bonsai.Dataset(X, y, device="cuda", device_id=0)` bins on the device and leaves the matrix resident there. A `cuda_*` fit then costs what the fused `train(params, X, y)` call costs, and a sweep uploads the matrix once instead of once per fit. Without the hint the `Dataset` bins on the CPU and every GPU fit uploads the result. `device="cuda"` raises when the build carries no CUDA backend or no device is visible: it is an explicit request, not an engine inference. `ds.device` reports where the bins ended up. Handing a device-binned `Dataset` to a CPU grower is fine: host columns materialize once, on first use, bit-identical to the host fill. A `parallel.device_id` at `train` time that disagrees with the `Dataset`'s raises instead of migrating the matrix. A `Dataset` cannot be pickled either way; rebuild it from `X` and `y` in the target process.

If the data already lives on the GPU, hand it over as it is: `bonsai.train(params, cp.asarray(X), cp.asarray(y))`. Any CUDA array supporting DLPack (a cupy array, a torch CUDA tensor, a jax array; a numba device array through `cupy.asarray`) is accepted wherever `X`, `y`, or `sample_weight` is accepted, by `train` and by `Dataset` alike, and `X` is binned on the device in place: no copy to the host and no copy back. The bins are identical to the ones the same array would produce through numpy, so the model is byte-identical either way. `y` and the weights are downloaded once, because bonsai keeps labels and weights on the host whatever the features do. `device` then defaults to where `X` already is, so the hint is only needed for host arrays; passing `device="cpu"` with a device-resident `X` raises rather than quietly copying it back. The array must be C-contiguous float32 (missing values are NaN, as everywhere else), and stream ordering is the producer's under DLPack: it synchronizes at export, and bonsai reads on the default stream. An array on a device that `parallel.device_id` disagrees with raises instead of migrating. A CPU grower fed device-resident input still works: the bins are built on the device and the host copy materializes once, on first use.

Prediction stays a host call: `predict` takes numpy arrays, whichever way the model was fit.

`n_threads` sizes the binning pass the way `parallel.n_threads` sizes a fit, and defaults to the same auto setting.

`Model` carries the full prediction surface: `predict` (with `num_iteration` for truncated ensembles), `predict_proba`, `staged_predict`, `predict_leaf`, `pred_contribs` (exact TreeSHAP), `feature_importance("gain")`/`feature_importance("split")`, `dump`, and `save`.

## The CLI

The same engine, the same keys, the same models:

```
bonsai fit     -c config.toml --set tree.max_depth=8 --model out.msgpack
bonsai predict -c config.toml --model out.msgpack --out preds.csv
bonsai eval    -c config.toml --model out.msgpack
bonsai info    # every (objective, grower, sampler) combo this binary dispatches
bonsai params  # the default config as TOML
```

`-c` supplies a TOML base; `--set` overrides it, exactly like `config=` and `params` in Python. `--dump-config` prints the resolved result and exits, which is the fastest way to check what a run will actually use.

## GPU training

Pass `grower="cuda_leafwise"`, `"cuda_depthwise"`, or `"cuda_levelwise"` (or the dotted key `dispatch.grower_name`). `bonsai.cuda_available()` reports whether this build and machine can train on GPU; models trained on GPU predict everywhere, including CPU-only installs.

On linux x86_64 the release wheel trains on GPU out of the box: any NVIDIA driver R525+, no CUDA toolkit needed, 2.3MB total (the wheel carries its own statically linked CUDA runtime). Other platforms need a source build with `BONSAI_CUDA=ON`. Every release's CUDA wheel is validated on real GPU hardware before it ships ([issue #99](https://github.com/daniel-m-campos/bonsai/issues/99)).

## Objectives

`mse` (default), `mae`, `huber`, `quantile`, `poisson`, `logloss`, `softmax`. The estimators pick classification objectives automatically from your labels; the explicit layer sets `dispatch.objective_name`. Ranking objectives are a measured, scoped gap, not an accident of omission.

## Reproducing the benchmarks

The harness behind every published table ships in the package. `pip install bonsai-gbt[bench]` adds the reference libraries, then `python -m bonsai.bench.grinsztajn out.jsonl` runs the external standings suite and `--report` renders the standings from the jsonl; [Running the benchmarks](benchmarks.md) walks every suite. The building blocks are importable directly:

```{.python .run}
import bonsai
from bonsai.bench import metrics, params, synth

X_train, y_train, X_test, y_test = synth.gen_data(
    10_000, 20, seed=42, n_test=1_000, informative=20)
model = bonsai.BonsaiRegressor(n_iters=8).fit(X_train, y_train)
print(round(metrics.r2(y_test, model.predict(X_test)), 3),
      params.CAMPAIGN["iters"])
```

The protocol (divisions, metrics, timing modes) is the [benchmark charter](https://daniel-m-campos.github.io/bonsai/method/benchmark-protocol/).

## What to read next

The [guide](../guide/README.md) explains what every knob actually does, mechanism first: growers in [chapter 4](../guide/4-growing-trees.md), sampling in [chapter 5](../guide/5-sampling.md), regularization and constraints in [chapter 6](../guide/6-regularization-and-constraints.md), early stopping and DART in [chapter 7](../guide/7-early-stopping-and-dart.md).

This page is the surface you call. To extend the engine, read [Concepts to types](../design/api-tour-concepts.md). It is this page's mirror image: the concepts you satisfy to add an objective, a grower, or a compute backend.
