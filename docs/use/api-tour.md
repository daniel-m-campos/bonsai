# The API in one read

bonsai's entire API follows from three facts. Everything else is detail.

1. **There are two layers over one engine.** Scikit-learn-shaped estimators (`BonsaiRegressor`, `BonsaiClassifier`) for pipelines and quick work, and an explicit layer (`train`, `Dataset`, `Model`) when you want full control. Both call the same C++ training path the CLI uses.
2. **There is one configuration system.** Every knob is a dotted key like `tree.max_depth` or `dispatch.grower_name`. The same keys work as `params` pairs in Python, as `--set` overrides on the CLI, and as sections in a TOML file. `bonsai.default_config_toml()` prints all of them with defaults.
3. **There is one model format.** `.msgpack` files round-trip everywhere: a model trained in Python predicts from the CLI, and vice versa. Models are byte-identical across CPU architectures and thread counts, so a saved model is a reproducible artifact, not a snapshot of one machine.

## Install

Grab the wheel for your platform from the [latest release](https://github.com/daniel-m-campos/bonsai/releases/latest) (Linux x86_64/aarch64, macOS arm64; Python 3.9 to 3.13; no toolchain needed):

```
pip install <wheel url>
```

Building from source instead needs LLVM 20+ and CMake; see the [README](https://github.com/daniel-m-campos/bonsai/blob/main/README.md#build).

## The estimator layer

If you know scikit-learn, you know this layer:

```python
import bonsai

model = bonsai.BonsaiRegressor(
    n_iters=200, learning_rate=0.05, grower="leafwise",
    early_stopping_rounds=20,
)
model.fit(X_train, y_train, eval_set=(X_valid, y_valid), sample_weight=w)
pred = model.predict(X_test)
model.save("model.msgpack")
```

`BonsaiClassifier` handles binary and multiclass the sklearn way: arbitrary label values in, `classes_` out, `predict_proba` returning `(n, K)` probabilities.

Both estimators duck-type the full sklearn contract (`get_params`, `clone`, `Pipeline`, `GridSearchCV`, `cross_val_score`, pickling) without importing sklearn at runtime, and both accept the familiar constructor aliases (`n_estimators`, `num_leaves`, `random_state`, `reg_lambda`, `max_bin`, ...).

Anything without a first-class kwarg goes through `params`, using the dotted keys:

```python
bonsai.BonsaiRegressor(params={"tree.lambda_l1": 0.5, "sampler.subsample": 0.8})
```

## The explicit layer

`train` takes the dotted keys directly and returns a `Model`:

```python
model = bonsai.train(
    [("dispatch.grower_name", "oblivious"), ("booster.n_iters", "200")],
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

Pass `grower="cuda_depthwise"` or `"cuda_oblivious"` (or the dotted key `dispatch.grower_name`). `bonsai.cuda_available()` reports whether this build and machine can train on GPU; models trained on GPU predict everywhere, including CPU-only installs.

On linux x86_64 the release wheel trains on GPU out of the box: any NVIDIA driver R525+, no CUDA toolkit needed, 2.3MB total (the wheel carries its own statically linked CUDA runtime). Other platforms need a source build with `BONSAI_CUDA=ON`. Every release's CUDA wheel is validated on real GPU hardware before it ships ([issue #99](https://github.com/daniel-m-campos/bonsai/issues/99)).

## Objectives

`mse` (default), `mae`, `huber`, `quantile`, `poisson`, `logloss`, `softmax`. The estimators pick classification objectives automatically from your labels; the explicit layer sets `dispatch.objective_name`. Ranking objectives are a measured, scoped gap, not an accident of omission.

## Reproducing the benchmarks

The harness behind every published table ships in the package. `pip install bonsai-gbt[bench]` adds the reference libraries, then `python -m bonsai.bench.grinsztajn out.jsonl --report` re-runs the external standings suite. The building blocks are importable directly:

```python
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
