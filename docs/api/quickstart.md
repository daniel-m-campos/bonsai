# Quickstart

bonsai is a gradient-boosted tree library with scikit-learn-style estimators, an explicit `train`/`Dataset`/`Model` layer, and a CLI, all over one C++ engine. Three facts shape how you use it: two layers, one configuration system, one model format. [The reference](reference.md) lists every function and argument.

1. **There are two layers over one engine.** Scikit-learn-shaped estimators (`BonsaiRegressor`, `BonsaiClassifier`) for pipelines and quick work, and an explicit layer (`train`, `Dataset`, `Model`) when you want full control. Both call the same C++ training path the CLI uses.
2. **There is one configuration system.** Every knob is a dotted key like `tree.max_depth` or `dispatch.grower_name`. The same keys work as `params` in Python (a `Params` or a dotted-key dict), as `--set` overrides on the CLI, and as sections in a TOML file. [Parameters](parameters.md) lists all of them with defaults.
3. **There is one model format.** `.msgpack` files round-trip everywhere: a model trained in Python predicts from the CLI, and vice versa. A model trained on CPU is byte-identical across runs and across CPU architectures at a fixed thread count (a different thread count can change the bytes), so that file is a reproducible artifact rather than a snapshot of one machine. A GPU fit is byte-identical across runs on the same device and build. Its histogram cells are integer fixed point (sums of integers, so the order of addition does not change the result). Identical bytes are not promised across devices, and a GPU fit's predictions match the CPU fit's to within 1e-4, not byte for byte ([the contract](../learn/determinism-as-a-contract.md)).

## Which API when

Choose the API by who holds the model and how many fits you run on the same data: estimators when another library (scikit-learn) holds the model, `Dataset` plus `train` for many fits on one dataset, and fused `train` for a single fit.

**Estimators, for scikit-learn interop.** `BonsaiRegressor` and `BonsaiClassifier` are what a `Pipeline`, a `GridSearchCV`, a `cross_val_score`, or a `clone` expects to be handed. Reach for them when something else in your stack holds the model.

**`Dataset` plus `train`, for repeat fits.** `bonsai.Dataset(X, y, device=...)` bins your features once at construction (it buckets each feature's values into histogram bins, up to 255 per feature by default, so splits are evaluated over bins instead of raw values). Every later `train(params, ds)` skips that step and gives a model bit-identical to training from the arrays. Use it for hyperparameter search, multi-seed ensembles, repeated refits on data you already hold in memory, and any serving path where one dataset feeds many fits. On GPU it is also where the matrix uploads once instead of once per fit.

**Fused `train(params, X, y)`, for one-shots.** `train(params, X, y)` fits once from arrays and keeps no `Dataset`. It bins the features inside the call and discards the bins when the call returns, so ten fits on the same data bin ten times.

## The estimator layer

This layer follows scikit-learn's estimator interface (`fit`, `predict`, `get_params`, `clone`), with bonsai's own parameter names and an `eval_set` argument on `fit`:

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

A 2-D `y` of shape `(n_rows, n_targets)` fits one tree per target per round and `predict` returns `(n_rows, n_targets)`, on both estimator and `Dataset` ([chapter 16](../guide/16-multi-target.md)). `BonsaiClassifier` handles binary and multiclass the scikit-learn way: arbitrary label values in, `classes_` out, `predict_proba` returning `(n, K)` probabilities. Both estimators implement `get_params`, `set_params`, `clone` support, pickling, and the interfaces that `Pipeline`, `GridSearchCV`, and `cross_val_score` use. scikit-learn is not a dependency: bonsai does not subclass its classes, and `import bonsai` never needs it.

Constructor arguments are bonsai's own names and nothing else; a parameter dict written for XGBoost, LightGBM, or CatBoost goes through [`bonsai.interop`](reference.md#bonsai.interop) first. Anything without a first-class kwarg goes through `params`, as a `bonsai.Params` or a dict of the dotted keys:

```python
bonsai.BonsaiRegressor(params={"tree.lambda_l1": 0.5, "sampler.subsample": 0.8})
bonsai.BonsaiRegressor(params=bonsai.Params(tree=bonsai.params.Tree(lambda_l1=0.5)))
```

## The explicit layer

`train` takes the overrides in either of two shapes and returns a `Model`. The typed shape is `bonsai.Params`, one generated dataclass per config section, where an unset field means "leave the library default"; a plain dict of dotted keys renders to identical overrides. `Params | dict` merges like two dicts: the right side wins. Use it to change one setting per trial in a hyperparameter search:

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

`bonsai.Dataset(X, y, device="cuda")` bins the matrix on the GPU and keeps it in GPU memory, so later fits do not upload it again. You can pass a CUDA array from cupy, torch, jax, or any library that supports DLPack (a shared format for exchanging GPU arrays) wherever `X` is expected, and bonsai bins it on the GPU without copying it to the host. Every `Model` method that takes `X` (`predict`, `predict_proba`, `staged_predict`, `predict_leaf`, `pred_contribs`) takes a `Dataset` too. What each of those accepts and refuses is on [the reference](reference.md#bonsai.Dataset).

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

Pass `grower="cuda_leafwise"`, `"cuda_depthwise"`, or `"cuda_levelwise"` (or the dotted key `dispatch.grower_name`). `bonsai.cuda_available()` reports whether this build and machine can train on GPU; models trained on GPU predict everywhere, including CPU-only installs. Only the Linux x86_64 wheel includes the CUDA backend; wheels for other platforms train on CPU. On that wheel you need only an NVIDIA driver ([Install](install.md#gpu-support-in-the-linux-x86_64-wheel)).

Objectives: `mse` (default), `mae`, `huber`, `quantile`, `poisson`, `logloss`, `softmax`. The estimators pick classification objectives from your labels; the explicit layer sets `dispatch.objective_name`. What every knob does, mechanism first, is [the guide](../guide/README.md).
