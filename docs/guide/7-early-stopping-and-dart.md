# 7. Early stopping & DART

Two booster-level features that reshape *the ensemble* rather than any
single tree.

## Early stopping

### The idea

`n_iters` is the worst hyperparameter to hand-tune: too few underfits, too
many overfits *and* wastes time. Watch a held-out validation set instead;
when its loss hasn't improved for $k$ consecutive rounds ("patience"),
stop, and keep the model as of the best round. A side benefit: you can run
a higher learning rate and let the valid set decide where to get off.

### In bonsai

The loop lives in `train_with_progress`
([`src/cli/pipeline.cpp`](../../src/cli/pipeline.cpp)). The interesting
part is keeping the per-round valid evaluation $O(\text{rows})$ instead of $O(\text{rows} \times \text{trees})$:
re-predicting the whole ensemble every round is quadratic in total. bonsai accumulates valid scores *incrementally*:
`IBooster::seed_validation_scores` fills the buffer as of the rounds already in the
model, and `accumulate_last_round` adds just the newest round's tree or trees
([`include/bonsai/booster.hpp`](../../include/bonsai/booster.hpp)).
Seeding at zero rounds gives base scores only, which is also the warm-start
seam. The loss comes from `IBooster::validation_loss`, which scores with the
booster's own configured objective, so any registered objective works. On stop,
`truncate(es_base + best_iter + 1)` drops the trailing trees, where `es_base`
is the round count a warm start brought in: the saved model *is* the best
iteration, like the references.

Config: `booster.early_stopping_rounds` + a `data.valid` CSV (Python:
`eval_set=(Xv, yv)`).

### Try it

```{.python .run}
import numpy as np
import bonsai

rng = np.random.default_rng(0)
X = rng.normal(size=(6000, 12)).astype(np.float32)
y = (X[:, 0] * 2.0 + X[:, 3] + rng.normal(0, 0.2, 6000)).astype(np.float32)
Xtr, ytr = X[:5000], y[:5000]
Xv, yv = X[5000:], y[5000:]

m = bonsai.BonsaiRegressor(
    n_iters=400, learning_rate=0.15, grower="leafwise",
    early_stopping_rounds=20).fit(Xtr, ytr, eval_set=(Xv, yv))
print("stopped at iteration:", m.n_iters_)
print("valid R2:", round(m.score(Xv, yv), 4))
```

Measured (feature_gap §3): with everyone stopping on the same 90/10 split,
all five libraries converge to RMSE 8.96–9.00, and bonsai leafwise lands
*between* XGBoost and LightGBM, erasing the gap a fixed 200-iteration
budget showed.

### Reusing the validation set

Each round walks the validation rows, and the walk reads a quarter of the
bytes when those rows are binned instead of raw. bonsai bins them inside the
fit, but only when the rounds it will run pay for the pass, because that
pass costs about what 90 rounds of the raw walk cost. A hyperparameter
search is where that arithmetic goes wrong: every fit pays it again for the
same rows. Bin them once instead, against the training set's own cut points,
and hand the same object to every fit:

```{.python .run}
import numpy as np
import bonsai

rng = np.random.default_rng(0)
X = rng.normal(size=(6000, 12)).astype(np.float32)
y = (X[:, 0] * 2.0 + X[:, 3] + rng.normal(0, 0.2, 6000)).astype(np.float32)
train = bonsai.Dataset(X[:5000], y[:5000])
valid = bonsai.Dataset(X[5000:], y[5000:], reference=train)

for depth in ("4", "6", "8"):
    m = bonsai.train(
        [("tree.max_depth", depth), ("booster.n_iters", "400"),
         ("booster.learning_rate", "0.15"),
         ("booster.early_stopping_rounds", "20")],
        train, eval_set=valid)
    print(f"depth {depth}: stopped at {m.n_iters}, valid mse "
          f"{min(m.eval_history):.4f}")
```

`reference=train` is the whole feature: it binds the validation rows to the
training cuts, which is the only binning a fit can route them through, since
a split's stored threshold names a bin under those cuts and no others. A
`Dataset` binned any other way is refused with that spelling in the message
rather than scored against the wrong splits. The estimators take the same
object: `fit(X, y, eval_set=valid)`, where `BonsaiClassifier` wants the
labels already encoded to `0..K-1`. This is what LightGBM's
`Dataset(reference=...)`, XGBoost's `QuantileDMatrix`, and CatBoost's `Pool`
do at construction, and it is why a per-fit bin pass never shows up in their
sweeps.

## DART

### The idea

In plain boosting, early trees are load-bearing forever and late trees fix
leftovers ("over-specialization"). DART (Dropouts meet Multiple Additive
Regression Trees) borrows dropout: each round, temporarily *drop* a random
subset of existing trees, fit the new tree against the reduced model's
gradients, then rescale so the ensemble's expected output is preserved.

### The math

Drop $k$ trees; fit the new tree; then normalize (XGBoost's
`normalize_type="tree"`, with learning rate $\eta$):

```math
\text{new tree} \times \frac{1}{k + \eta} \qquad\qquad
\text{dropped trees} \times \frac{k}{k + \eta}
```

The paper's original factors are $\tfrac{1}{k+1}$ and $\tfrac{k}{k+1}$,
correct for *unshrunk* trees. Combined with a learning rate they starve
the new tree by a factor of $\sim 1/\eta$. This is not hypothetical (see
below).

### In bonsai

All in `Booster::update_one_iter`
([`include/bonsai/booster.hpp`](../../include/bonsai/booster.hpp)): draw
the dropout set from the booster rng (seed-deterministic), subtract the
dropped trees' training contributions from `scores_`, compute gradients,
grow, rescale via `Tree::scale_leaves`, add everything back.

The neat trick is recovering a dropped tree's per-row training
contribution **without caching predictions** (200 trees × 500k rows of
cache is a non-starter): route rows through the tree *in bin space*:
`internal::accumulate_train_contribution` maps each stored float threshold
back to its bin with one `lower_bound` over the mapper cuts, exact because
thresholds are cut values (the same invariant chapter 5's fix uses).

Config: `booster.dart_drop_rate` (XGBoost `rate_drop`, LightGBM
`drop_rate`). Incompatible with early stopping by construction: DART
rescales *earlier* trees each round, which invalidates incrementally
accumulated valid scores, so the combination throws.

### Try it

```{.python .run}
import numpy as np
import bonsai

rng = np.random.default_rng(0)
X = rng.normal(size=(4000, 8)).astype(np.float32)
y = (X[:, 0] * 2.0 + X[:, 1] + rng.normal(0, 0.1, 4000)).astype(np.float32)

plain = bonsai.BonsaiRegressor(n_iters=100, grower="depthwise").fit(X, y)
dart = bonsai.BonsaiRegressor(
    n_iters=100, grower="depthwise",
    params={"booster.dart_drop_rate": 0.1}).fit(X, y)
print("plain R2:", round(plain.score(X, y), 4))
print("dart  R2:", round(dart.score(X, y), 4))
```

Measured (feature_gap §8): DART regularizes: everyone lands *above* their
plain-GBDT RMSE at a fixed 200-round budget, and bonsai's implementation
degrades least (0.593 vs XGBoost 0.626, LightGBM 0.702).

## Gotchas & war stories

- **The $k+1$ trap.** bonsai first implemented the DART paper's
  $1/(k+1)$ normalization literally. RMSE: 0.88 (*worse than every
  reference*) because with $\eta = 0.05$ each new tree landed $\sim 20$x
  too small to matter. Switching to $k+\eta$ (what XGBoost actually does)
  moved bonsai to best in
  the DART field in one line. Read the paper *and* the reference source.
- **Early stopping evaluates in raw-score space**: the objective's own
  loss, before any link inverse. Monotonic transformations don't change
  the argmin, so this is fine *and* cheaper.
- **Patience interacts with learning rate.** High $\eta$ + small patience
  stops on noise; the benchmark's $\eta = 0.15$, patience $= 20$ is a
  reasonable default pairing.
