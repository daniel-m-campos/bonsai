# Switching from XGBoost

bonsai's estimators speak bonsai's parameter names. There is no alias union in the constructor, because a signature that accepts three libraries' spellings at once cannot tell you which one you are using, and that ambiguity is what this page used to paper over.

What replaces it is explicit: `bonsai.interop` translates an XGBoost parameter dict into bonsai keys, in one place, with every non-equivalence written down at the mapping.

## Porting a config

Hand your XGBoost parameters to `from_xgboost` and train on what comes back:

```{.python .run}
import bonsai
import numpy as np

rng = np.random.default_rng(0)
X = rng.random((600, 8), dtype=np.float32)
y = (X[:, 0] * 2 + np.sin(X[:, 1] * 6) + rng.normal(0, 0.2, 600)).astype(np.float32)
X_train, y_train, X_valid, y_valid = X[:400], y[:400], X[400:], y[400:]

xgb_config = {
    "n_estimators": 200,
    "learning_rate": 0.1,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1.0,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "random_state": 0,
    "early_stopping_rounds": 20,
    "tree_method": "hist",
}
params = bonsai.interop.from_xgboost(xgb_config)
model = bonsai.train(params, X_train, y_train, eval_set=(X_valid, y_valid))
print("rounds kept:", model.n_iters)
print("top feature:", int(np.argmax(model.feature_importance("gain"))))
```

The same translation configures an estimator, if you want the sklearn contract around the fit:

```python
translated = bonsai.interop.from_xgboost({"n_estimators": 200, "reg_lambda": 1.0})
est = bonsai.BonsaiRegressor(params=translated.to_dict())
```

Writing the config in bonsai's own names from the start is one line shorter and needs no translation at all:

```python
est = bonsai.BonsaiRegressor(
    n_iters=200, learning_rate=0.1, max_depth=5, subsample=0.8,
    objective="mse", random_seed=0, early_stopping_rounds=20,
    params={"tree.feature_fraction": 0.8, "tree.min_child_hess": 1.0},
)
```

## What the mapping does

| XGBoost spelling | bonsai key |
| --- | --- |
| `n_estimators`, `learning_rate`, `max_depth`, `max_leaves` | `booster.n_iters`, `booster.learning_rate`, `tree.max_depth`, `tree.max_leaves` |
| `seed` / `random_state`, `n_jobs` / `nthread` | `booster.random_seed`, `parallel.n_threads` |
| `reg_lambda`, `reg_alpha`, `gamma` | `tree.lambda_l2`, `tree.lambda_l1`, `tree.min_gain_to_split` |
| `min_child_weight` | `tree.min_child_hess`, the same minimum hessian mass per child, same default of 1.0 |
| `subsample` | `sampler.subsample`, and the sampler switches to `bernoulli` |
| `colsample_bytree`, `max_bin` | `tree.feature_fraction`, `bin_mapper.max_bin` |
| `objective` | `dispatch.objective_name`: `reg:squarederror` is `mse`, `reg:absoluteerror` is `mae`, `reg:quantileerror` is `quantile`, `reg:pseudohubererror` is `huber`, `count:poisson` is `poisson`, `binary:logistic` is `logloss`, `multi:softprob` and `multi:softmax` are `softmax` |
| `grow_policy` | `dispatch.grower_name`: `depthwise` is `depthwise`, `lossguide` is `leafwise` |
| `early_stopping_rounds`, `rate_drop`, `quantile_alpha` | `booster.early_stopping_rounds`, `booster.dart_drop_rate`, `objective.quantile_alpha` |
| `tree_method`, `device`, `verbosity` | dropped: bonsai is always histogram-based, its device is the grower name, and logging is not a knob |

Anything else raises, naming every parameter that did not translate. `from_xgboost(config, strict=False)` drops them instead.

## What is deliberately different

Honest differences rather than familiar spellings over changed behavior.

- The estimator constructor takes bonsai names only. `n_estimators`, `num_leaves`, `random_state`, `n_jobs`, `reg_lambda`, `reg_alpha`, `min_child_samples`, `colsample_bytree`, `min_child_weight`, and `gamma` are gone; use `interop` or the dotted keys in `params`.
- `min_child_weight` is a hessian floor on both sides, so under squared error it is a row count and under logloss it demands far more rows than its face value. Matching it across libraries matches the knob, not the constraint.
- `reg:pseudohubererror` maps to bonsai's `huber`, which is the exact Huber loss rather than the pseudo-Huber approximation; the `huber_delta` knob is `params={"objective.huber_delta": ...}`.
- XGBoost keeps every tree it grew when early stopping fires and leaves truncation to `iteration_range`. bonsai returns the best-iteration model, so a translated config predicts from a shorter ensemble than XGBoost would.
- `eval_set` is one `(X, y)` tuple, or one `bonsai.Dataset(Xv, yv, reference=train_dataset)`, which is the `QuantileDMatrix` idea: the validation rows are binned once against the training cuts and every fit in a sweep reuses them. The list-of-tuples form is rejected rather than quietly reduced to its last entry, because bonsai tracks a single validation set.
- `evals_result()` returns `{"valid": {objective_name: [...]}}`, not `{"validation_0": {metric: [...]}}`. The outer key names the eval set the way the CLI labels its metric column, and the inner key is the objective under its bonsai name and in its own units, so `best_score` and the curve read `mse` rather than a square-rooted `rmse`.
- Prediction truncates with `num_iteration=n`, the same argument `Model.predict` takes; there is no `iteration_range`. A prefix always starts at the head, because a boosted sum has no meaning without it.
- Saving is `save(path)` and loading is `BonsaiRegressor.from_file(path)`. `save_model` and `load_model` are gone, and there is no in-place loader: bind the returned estimator.
- `apply` stays, because it is scikit-learn's own name for per-tree leaf indices (`GradientBoostingRegressor.apply`) and this layer exists for sklearn speakers. `predict_leaf` is the native spelling of the same call.
- `early_stopping_rounds` lives in the constructor, matching XGBoost 2.x and later; there is no `fit(early_stopping_rounds=...)`.
- Categorical features go through [`OrderedTargetEncoder`](../guide/13-categorical-features.md) rather than an `enable_categorical` flag; the measurement behind that choice is decision 58.
- The native layer (`train`, `Dataset`, `Model`) is bonsai's own explicit API, not a `DMatrix` clone; callbacks, dask, and spark integrations are out of scope.
- Loading a saved classifier restores encoded `0..K-1` class ids, because the native format stores only the booster; pickle the estimator to preserve original labels.

`from_lightgbm` and `from_catboost` are the same shape for the other two libraries, with their own caveats recorded at their own tables. Every knob that has no mapping at all is still reachable as a dotted config key; [Parameters](parameters.md) lists them all.
