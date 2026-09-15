# Quickstart

bonsai's entire API follows from three facts. Everything else is detail, and the detail is on [the reference](reference.md).

1. **There are two layers over one engine.** Scikit-learn-shaped estimators (`BonsaiRegressor`, `BonsaiClassifier`) for pipelines and quick work, and an explicit layer (`train`, `Dataset`, `Model`) when you want full control. Both call the same C++ training path the CLI uses.
2. **There is one configuration system.** Every knob is a dotted key like `tree.max_depth` or `dispatch.grower_name`. The same keys work as `params` in Python (a `Params` or a dotted-key dict), as `--set` overrides on the CLI, and as sections in a TOML file. [Parameters](parameters.md) lists all of them with defaults.
3. **There is one model format.** `.msgpack` files round-trip everywhere: a model trained in Python predicts from the CLI, and vice versa. A model trained on CPU is byte-identical across runs, thread counts, and CPU architectures, so that file is a reproducible artifact rather than a snapshot of one machine. GPU training carries the run-to-run half of that: a device fit is byte-identical to itself on one device and one build, since histogram cells are integer fixed point, but not across devices, and it matches the host fit to 1e-4 rather than byte for byte ([the contract](../learn/determinism-as-a-contract.md)).

## Which API when

Three call shapes over one engine. The question that picks between them is how many fits come out of one ingest.

**Estimators, for scikit-learn interop.** `BonsaiRegressor` and `BonsaiClassifier` are what a `Pipeline`, a `GridSearchCV`, a `cross_val_score`, or a `clone` expects to be handed. Reach for them when something else in your stack holds the model.

**`Dataset` plus `train`, for repeat fits.** `bonsai.Dataset(X, y, device=...)` runs the binning pass once at construction and every later `train(params, ds)` skips it, bit-identical to training from the arrays. This is the shape for hyperparameter search, multi-seed ensembling, production refits on a warm dataset, and any serving path where one ingest feeds many fits. On GPU it is also where the matrix uploads once instead of once per fit.

**Fused `train(params, X, y)`, for one-shots.** One fit, one array, no object to keep. The binning happens inside the call and goes away with it.

## The estimator layer

If you know scikit-learn, you know this layer:

```python {.run}
import bonsai
import numpy as np

rng = np.random.default_rng(0)
X = rng.random((600, 8), dtype=np.float32)
y = (X[:, 0] * 2 + rng.normal(0, 0.2, 600)).astype(np.float32)
X_train, y_train = X[:400], y[:400]
X_valid, y_valid = X[400:], y[400:]
X_test, w = X[400:], np.ones(400, dtype=np.float32)

model = bonsai.BonsaiRegressor(
    n_iters=200,
    learning_rate=0.05,
    grower="leafwise",
    early_stopping_rounds=20,
)
model.fit(X_train, y_train, eval_set=(X_valid, y_valid), sample_weight=w)
pred = model.predict(X_test)
model.save("model.msgpack")
```

`BonsaiClassifier` handles binary and multiclass the sklearn way: arbitrary label values in, `classes_` out, `predict_proba` returning `(n, K)` probabilities. Both estimators duck-type the full sklearn contract (`get_params`, `clone`, `Pipeline`, `GridSearchCV`, `cross_val_score`, pickling) without importing sklearn at runtime.

Constructor arguments are bonsai's own names and nothing else; a parameter dict written for XGBoost, LightGBM, or CatBoost goes through [`bonsai.interop`](reference.md#bonsai.interop) first. Anything without a first-class kwarg goes through `params`, as a `bonsai.Params` or a dict of the dotted keys:

```python
bonsai.BonsaiRegressor(params={"tree.lambda_l1": 0.5, "sampler.subsample": 0.8})
bonsai.BonsaiRegressor(params=bonsai.Params(tree=bonsai.params.Tree(lambda_l1=0.5)))
```

## The explicit layer

`train` takes the overrides in either of two shapes and returns a `Model`. The typed shape is `bonsai.Params`, one generated dataclass per config section, where an unset field means "leave the library default"; a plain dict of dotted keys renders to identical overrides. `Params | dict` merges the way dicts do (right side wins), which is the sweep idiom:

```python
from bonsai.params import Params, Booster, Dispatch

BASE = Params(dispatch=Dispatch(grower_name="levelwise"), booster=Booster(n_iters=200))


def objective(trial):
    depth = trial.suggest_int("tree.max_depth", 3, 12)
    m = bonsai.train(BASE | {"tree.max_depth": depth}, ds, eval_set=valid)
    return m.eval_history[-1]
```

For hyperparameter searches and cross-validation, bin once and train many times. A `Dataset` runs the binning pass at construction and every subsequent `train` call skips it; `reference=` bins a validation set with the training set's own cut points so every fit routes it in bin space from its first round:

```python
ds = bonsai.Dataset(X, y, max_bin=255)
valid = bonsai.Dataset(Xv, yv, reference=ds)
for params in grid:
    m = bonsai.train(params, ds, eval_set=valid)
```

`bonsai.Dataset(X, y, device="cuda")` bins on the device and leaves the matrix resident there; a CUDA array (cupy, torch, jax, anything DLPack) is accepted wherever `X` is and binned in place. Every `Model` method that takes `X` (`predict`, `predict_proba`, `staged_predict`, `predict_leaf`, `pred_contribs`) takes a `Dataset` too. What each of those accepts and refuses is on [the reference](reference.md#bonsai.Dataset).

## The CLI

The same engine, the same keys, the same models:

```
bonsai fit     -c config.toml --set tree.max_depth=8 --model out.msgpack
bonsai predict -c config.toml --model out.msgpack --out preds.csv
bonsai eval    -c config.toml --model out.msgpack
bonsai info    # every (objective, grower, sampler) combo this binary dispatches
bonsai params  # the default config as TOML
```

`-c` supplies a TOML base; `--set` overrides it, exactly like `Params.from_toml(path) | overrides` in Python. `--dump-config` prints the resolved result and exits. The binary is a source-build artifact: [Building from source](building.md).

## GPU training and objectives

Pass `grower="cuda_leafwise"`, `"cuda_depthwise"`, or `"cuda_levelwise"` (or the dotted key `dispatch.grower_name`). `bonsai.cuda_available()` reports whether this build and machine can train on GPU; models trained on GPU predict everywhere, including CPU-only installs. The linux x86_64 wheel carries the CUDA backend, so nothing but a driver is needed ([Install](install.md#gpu-support-in-the-linux-x86_64-wheel)).

Objectives: `mse` (default), `mae`, `huber`, `quantile`, `poisson`, `logloss`, `softmax`. The estimators pick classification objectives from your labels; the explicit layer sets `dispatch.objective_name`. What every knob does, mechanism first, is [the guide](../guide/README.md).
