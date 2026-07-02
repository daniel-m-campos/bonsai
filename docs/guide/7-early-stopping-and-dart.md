# 7 — Early stopping & DART

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
re-predicting the whole ensemble every round is quadratic in total. bonsai accumulates valid scores *incrementally* —
`IBooster::score_base()` seeds the buffer with the init score, and
`accumulate_last_tree` adds just the newest tree's (shrinkage-scaled)
contribution each round
([`include/bonsai/booster.hpp`](../../include/bonsai/booster.hpp)).
The loss comes from `eval_objective_by_name`, so any registered objective
works. On stop, `truncate(best_iter + 1)` drops the trailing trees — the
saved model *is* the best iteration, like the references.

Config: `booster.early_stopping_rounds` + a `data.valid` CSV (Python:
`eval_set=(Xv, yv)`).

### Try it

```bash
uv run scripts/compare.py --config configs/year_prediction_msd.toml \
    --hp booster.early_stopping_rounds=20 --hp booster.n_iters=400 \
    --hp booster.learning_rate=0.15 --growers leafwise --samplers all_rows
```

Measured (feature_gap §3): with everyone stopping on the same 90/10 split,
all five libraries converge to RMSE 8.96–9.00 — and bonsai leafwise lands
*between* xgboost and lightgbm, erasing the gap a fixed 200-iteration
budget showed.

## DART

### The idea

In plain boosting, early trees are load-bearing forever and late trees fix
leftovers ("over-specialization"). DART (Dropouts meet Multiple Additive
Regression Trees) borrows dropout: each round, temporarily *drop* a random
subset of existing trees, fit the new tree against the reduced model's
gradients, then rescale so the ensemble's expected output is preserved.

### The math

Drop $k$ trees; fit the new tree; then normalize (xgboost's
`normalize_type="tree"`, with learning rate $\eta$):

```math
\text{new tree} \times \frac{1}{k + \eta} \qquad\qquad
\text{dropped trees} \times \frac{k}{k + \eta}
```

The paper's original factors are $\tfrac{1}{k+1}$ and $\tfrac{k}{k+1}$ —
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
cache is a non-starter): route rows through the tree *in bin space* —
`internal::accumulate_train_contribution` maps each stored float threshold
back to its bin with one `lower_bound` over the mapper cuts, exact because
thresholds are cut values (the same invariant chapter 5's fix uses).

Config: `booster.dart_drop_rate` (xgboost `rate_drop`, lightgbm
`drop_rate`). Incompatible with early stopping by construction — DART
rescales *earlier* trees each round, which invalidates incrementally
accumulated valid scores — so the combination throws.

### Try it

```bash
uv run scripts/compare.py --config configs/california_housing.toml \
    --hp booster.dart_drop_rate=0.1 --growers depthwise,leafwise --samplers all_rows
```

Measured (feature_gap §8): DART regularizes — everyone lands *above* their
plain-GBDT RMSE at a fixed 200-round budget — and bonsai's implementation
degrades least (0.593 vs xgboost 0.626, lightgbm 0.702).

## Gotchas & war stories

- **The $k+1$ trap.** bonsai first implemented the DART paper's
  $1/(k+1)$ normalization literally. RMSE: 0.88 — *worse than every
  reference* — because with $\eta = 0.05$ each new tree landed $\sim 20$x
  too small to matter. Switching to $k+\eta$ (what xgboost actually does)
  moved bonsai to best in
  the DART field in one line. Read the paper *and* the reference source.
- **Early stopping evaluates in raw-score space** — the objective's own
  loss, before any link inverse. Monotonic transformations don't change
  the argmin, so this is fine *and* cheaper.
- **Patience interacts with learning rate.** High $\eta$ + small patience
  stops on noise; the benchmark's $\eta = 0.15$, patience $= 20$ is a
  reasonable default pairing.
