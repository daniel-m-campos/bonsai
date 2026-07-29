# Switching from XGBoost

The estimator layer is built so that the common XGBoost script runs with the class name swapped and nothing else. `XGBRegressor` becomes `BonsaiRegressor`, `XGBClassifier` becomes `BonsaiClassifier`, and the constructor arguments, `fit` shapes, and post-fit attributes you are likely using keep their spellings.

A canonical XGBoost script, unchanged except for the class name:

```{.python .run}
import bonsai
import numpy as np

rng = np.random.default_rng(0)
X = rng.random((600, 8), dtype=np.float32)
y = (X[:, 0] * 2 + np.sin(X[:, 1] * 6) + rng.normal(0, 0.2, 600)).astype(np.float32)
X_train, y_train, X_valid, y_valid = X[:400], y[:400], X[400:], y[400:]

est = bonsai.BonsaiRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1.0,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=0,
    early_stopping_rounds=20,
)
est.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

print("best_iteration:", est.best_iteration)
print("best rmse:", round(est.best_score, 4))
curve = est.evals_result()["validation_0"]["rmse"]
print("eval curve, first and best:", round(curve[0], 4), round(min(curve), 4))
print("top feature:", int(est.feature_importances_.argmax()))
```

## What maps automatically

| XGBoost spelling | bonsai meaning |
| --- | --- |
| `n_estimators`, `learning_rate`, `max_depth`, `max_leaves` | same knobs (`n_estimators` wins over `n_iters` when both are set) |
| `random_state`, `n_jobs` | `booster.random_seed`, `parallel.n_threads` |
| `reg_lambda`, `reg_alpha`, `gamma` | `tree.lambda_l2`, `tree.lambda_l1`, `tree.min_gain_to_split` |
| `min_child_weight` | `tree.min_child_hess`: the same minimum hessian mass per child, same default of 1.0 (a row count under squared error) |
| `subsample` | row sampling; switches the sampler to `bernoulli` when `sampler` is at its default |
| `colsample_bytree`, `max_bin`, `min_child_samples` | `tree.feature_fraction`, `bin_mapper.max_bin`, `tree.min_data_in_leaf` |
| `device="cuda"` | the CUDA grower matching your grower choice (`cuda_depthwise` by default, `cuda_oblivious` for `grower="oblivious"`) |
| `objective="reg:squarederror"` etc. | `mse`, `reg:absoluteerror` is `mae`, `reg:quantileerror` (+ `quantile_alpha`) is `quantile`, `count:poisson` is `poisson`; classifiers accept `binary:logistic` / `multi:softprob` / `multi:softmax` and derive the real objective from the class count |
| `eval_set=[(X, y)]` | the list form is native; with several entries the last one drives the eval history and early stopping, XGBoost's own convention |
| `evals_result()`, `best_iteration`, `best_score` | same shapes; the squared-error eval is presented as rmse (the exact root, so early stopping saw the same ordering) |
| `save_model` / `load_model` / `apply` / `iteration_range` | `save` / `from_file` / `predict_leaf` / `num_iteration` under their XGBoost names |

## What is deliberately different

Honest differences rather than missing spellings, so a silent behavior change never hides behind a familiar name.

- `early_stopping_rounds` lives in the constructor, which matches XGBoost 2.x and later; there is no `fit(early_stopping_rounds=...)`.
- `reg:pseudohubererror` maps to bonsai's `huber`, which is the exact Huber loss, not the pseudo-Huber approximation; the `huber_delta` knob is `params={"objective.huber_delta": ...}`.
- `iteration_range` must start at 0: a boosted sum has no meaning without its head.
- `verbose` is accepted and ignored; bonsai prints one line on early stop and nothing per round.
- Categorical features go through [`OrderedTargetEncoder`](../guide/13-categorical-features.md) rather than an `enable_categorical` flag; the measurement behind that choice is decision 58.
- The native layer (`train`, `Dataset`, `Model`) is bonsai's own explicit API, not a `DMatrix` clone; callbacks, dask, and spark integrations are out of scope.
- Loading a saved classifier restores encoded `0..K-1` class ids (XGBoost's `load_model` convention); pickle the estimator to preserve original labels.

Every knob that has no alias is reachable as a dotted config key through `params=`; [Parameters](parameters.md) lists them all.
