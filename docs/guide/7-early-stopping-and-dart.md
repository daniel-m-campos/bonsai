# 7. Early stopping & DART

Early stopping ends training at the best round on a validation set, and DART drops random trees each round to regularize. You cannot combine them: DART with an eval set or early stopping is refused (the Python surface raises an error).

## Early stopping

### The idea

Hand-tuning `n_iters` is error-prone: too few rounds underfit, and too many overfit and add training time. Watch a held-out validation set instead; when its loss hasn't improved for $k$ consecutive rounds ("patience"), stop, and keep the model as of the best round. A side benefit: you can raise the learning rate, because the validation set decides which round to keep.

### In bonsai

The loop lives in `train_with_progress` ([`src/cli/pipeline.cpp`](../../src/cli/pipeline.cpp)). bonsai scores the validation set incrementally, so each round costs $O(\text{rows})$. Re-predicting the whole ensemble every round would cost $O(\text{rows} \times \text{trees})$, which is quadratic over the fit. `ITrainableBooster::seed_validation_scores` fills the buffer as of the rounds already in the model, and `accumulate_last_round` adds just the newest round's tree or trees ([`include/bonsai/booster.hpp`](../../include/bonsai/booster.hpp)). Seeding at zero rounds gives base scores only. A warm start (continuing a fit from an existing model) seeds at the rounds the model already holds. The loss comes from `ITrainableBooster::validation_loss`, which scores with the booster's own configured objective, so any registered objective works. On stop, `truncate(scorer->base() + stopper.best_iter() + 1)` drops the trailing trees. Here `base()` is the round count the warm start brought in. The saved model *is* the best iteration, so it holds only the trees up to the stop point (LightGBM and CatBoost do the same; XGBoost keeps every tree and leaves truncation to prediction).

Config: `booster.early_stopping_rounds` + a `data.valid` CSV (Python: `eval_set=(Xv, yv)`).

### Try it

```python {.run}
import numpy as np
import bonsai

rng = np.random.default_rng(0)
X = rng.normal(size=(6000, 12)).astype(np.float32)
y = (X[:, 0] * 2.0 + X[:, 3] + rng.normal(0, 0.2, 6000)).astype(np.float32)
Xtr, ytr = X[:5000], y[:5000]
Xv, yv = X[5000:], y[5000:]

m = bonsai.BonsaiRegressor(
    n_iters=400, learning_rate=0.15, grower="leafwise", early_stopping_rounds=20
).fit(Xtr, ytr, eval_set=(Xv, yv))
print("stopped at iteration:", m.n_iters_)
print("valid R2:", round(m.score(Xv, yv), 4))
```

With every library early-stopping on the same 90/10 split, all five runs (four libraries, with bonsai run under both growers) reach RMSE 8.96 to 9.00 ([the feature-gap record §3](https://github.com/daniel-m-campos/bonsai/blob/main/benchmarks/feature-gap-2026-07.md)). At a fixed 200 iterations the same five spanned 8.99 to 9.14, so early stopping narrows the spread from 0.15 to 0.04 RMSE. bonsai leafwise lands between XGBoost and LightGBM either way (8.9977 vs 8.9900 and 9.0016 with early stopping; 9.0871 vs 9.1389 and 9.0826 at a fixed 200).

### Reusing the validation set

Each round walks the validation rows. For a 128-feature set and a depth-8 tree, the walk reads a quarter of the bytes when those rows are binned (converted to bin indexes using the training set's cut points) instead of raw. Binning them costs about what 86 rounds of the raw walk cost at that shape, so a fit starts raw and switches to the binned rows after 86 rounds: a short fit never pays for binning, and a long one recovers the cost. Nobody has to guess how long the fit will be, which matters here because early stopping is what decides it. A wider validation matrix needs more rounds before binning pays. With 64 features or more per level of `max_depth` (64 bytes per cache line is why), the raw walk already reads only the cache lines it needs, so the fit never switches. A `max_bin` above 255 stays raw, and a device grower bins on round 0. A hyperparameter search still loses time: every fit repeats the binning on the same validation rows, and runs its first rounds raw before it does. Bin them once instead, against the training set's own cut points, and hand the same object to every fit:

```python {.run}
import numpy as np
import bonsai

rng = np.random.default_rng(0)
X = rng.normal(size=(6000, 12)).astype(np.float32)
y = (X[:, 0] * 2.0 + X[:, 3] + rng.normal(0, 0.2, 6000)).astype(np.float32)
train = bonsai.Dataset(X[:5000], y[:5000])
valid = bonsai.Dataset(X[5000:], y[5000:], reference=train)

for depth in (4, 6, 8):
    m = bonsai.train(
        {
            "tree.max_depth": depth,
            "booster.n_iters": 400,
            "booster.learning_rate": 0.15,
            "booster.early_stopping_rounds": 20,
        },
        train,
        eval_set=valid,
    )
    print(f"depth {depth}: stopped at {m.n_iters}, valid mse {min(m.eval_history):.4f}")
```

Pass `reference=train` to bind the validation rows to the training cut points. A split's stored threshold names a bin under those cuts, so no other binning can route the rows through the fit. A `Dataset` binned any other way is refused with that spelling in the message rather than scored against the wrong splits. The estimators take the same object: `fit(X, y, eval_set=valid)`, where `BonsaiClassifier` wants the labels already encoded to `0..K-1`. This is what LightGBM's `Dataset(reference=...)`, XGBoost's `QuantileDMatrix`, and CatBoost's `Pool` do at construction, and it is why a per-fit bin pass never shows up in their sweeps.

## DART

### The idea

In plain boosting, the first trees dominate the prediction and later trees only correct the errors those first trees leave, which the DART paper calls over-specialization. DART (Dropouts meet Multiple Additive Regression Trees) borrows dropout from neural networks: each round, temporarily *drop* a random subset of existing trees, fit the new tree against the reduced model's gradients, then rescale so the ensemble's expected output is preserved.

### The math

Drop $k$ trees; fit the new tree; then normalize (XGBoost's `normalize_type="tree"`, with learning rate $\eta$):

```math
\text{new tree} \times \frac{1}{k + \eta} \qquad\qquad
\text{dropped trees} \times \frac{k}{k + \eta}
```

The paper's original factors are $\tfrac{1}{k+1}$ and $\tfrac{k}{k+1}$, correct for *unshrunk* trees. With a learning rate, they shrink the new tree by a factor of about $1/\eta$ (20x at $\eta = 0.05$), which raised bonsai's RMSE to 0.88 (see the $k+1$ trap below).

### In bonsai

All in `Booster::update_one_iter` ([`include/bonsai/booster.hpp`](../../include/bonsai/booster.hpp)): draw the dropout set from the booster rng (seed-deterministic), subtract the dropped trees' training contributions from `scores_`, compute gradients, grow, rescale via `scale_leaves` (a member of both `DenseTree` and `ObliviousTree`, called from `Booster::apply_dart_round`), add everything back.

bonsai recovers a dropped tree's per-row training contribution **without caching predictions**, because caching 200 trees over 500k rows would store 100 million values. It routes rows through the tree *in bin space*: `detail::accumulate_train_contribution` maps each stored float threshold back to its bin with one `lower_bound` over the mapper cuts, exact because thresholds are cut values (the same invariant chapter 5's fix uses).

Config: `booster.dart_drop_rate` (XGBoost `rate_drop`, LightGBM `drop_rate`). Incompatible with an eval set by construction, and so with early stopping: DART rescales *earlier* trees each round, which invalidates incrementally accumulated valid scores, so `fit` with an `eval_set` and `dart_drop_rate > 0` throws even when `early_stopping_rounds` is 0.

### Try it

```python {.run}
import numpy as np
import bonsai

rng = np.random.default_rng(0)
X = rng.normal(size=(4000, 8)).astype(np.float32)
y = (X[:, 0] * 2.0 + X[:, 1] + rng.normal(0, 0.1, 4000)).astype(np.float32)

plain = bonsai.BonsaiRegressor(n_iters=100, grower="depthwise").fit(X, y)
dart = bonsai.BonsaiRegressor(
    n_iters=100, grower="depthwise", params={"booster.dart_drop_rate": 0.1}
).fit(X, y)
print("plain R2:", round(plain.score(X, y), 4))
print("dart  R2:", round(dart.score(X, y), 4))
```

Measured ([the feature-gap record §8](https://github.com/daniel-m-campos/bonsai/blob/main/benchmarks/feature-gap-2026-07.md)): DART regularizes: everyone lands *above* their plain-GBDT RMSE at a fixed 200-round budget, and bonsai's implementation degrades least (0.593 vs XGBoost 0.626, LightGBM 0.702).

## Gotchas & war stories

- **The $k+1$ trap.** bonsai first implemented the DART paper's $1/(k+1)$ normalization literally. RMSE: 0.88 (*worse than every reference*) because with $\eta = 0.05$ each new tree landed $\sim 20$x too small to matter. Switching to $k+\eta$ (what XGBoost actually does) moved bonsai to best in the DART field in one line. Read the paper *and* the reference source.
- **Early stopping evaluates in raw-score space**: the objective's own loss, before any link inverse. Monotonic transformations don't change the argmin, so this is fine *and* cheaper.
- **Patience interacts with learning rate.** High $\eta$ + small patience stops on noise; the benchmark's $\eta = 0.15$, patience $= 20$ is a reasonable default pairing.
