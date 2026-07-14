# 3. Finding splits

## The idea

A node holds a set of rows. A split sends each row left or right by
comparing one feature to one threshold. The best split is the one that
lets the two children correct the loss better than the parent could alone,
and with histograms, evaluating *every* candidate threshold of a feature
costs one scan over its ~255 cells.

## The math

From chapter 1, a leaf with sums $(G, H)$ achieves (up to a constant) loss
reduction proportional to its **score**:

```math
\mathrm{score}(G, H) = \frac{G^2}{H + \lambda}
```

A split's **gain** is what the children score over the parent:

```math
\text{gain} = \mathrm{score}(G_L, H_L) +
\mathrm{score}(G_R, H_R) - \mathrm{score}(G, H)
```

Intuition: $G^2$ rewards leaves whose gradients *agree* (a large sum of
same-signed residuals is a mistake a leaf value can fix); dividing by
$H + \lambda$ discounts thin evidence. A split earns gain by separating rows
with positive residuals from rows with negative ones.

The scan: walk cells left to right keeping a running $(G_L, H_L)$; the
right side is the node total minus the prefix. Each of the ~255 positions
is scored in O(1), each feature independently, and the best (feature, bin)
wins.

**Missing values** get their own reserved bin, excluded from the prefix
walk. For every candidate, bonsai scores *both* routings (NaNs-left and
NaNs-right) and keeps the better (`default_left` on the split). That is
xgboost's "learned default direction": the data decides where missing
belongs, per split.

## In bonsai

All of it is [`src/split.cpp`](../../src/split.cpp):

- `update_best_for_feature_for_node`: the per-feature scan. Read it
  top to bottom: hoist the node score and the real (non-missing) totals,
  then for each cut cell accumulate the prefix and score both
  `default_left` routings via `split_sums_at`, the single source of truth
  for missing-routing arithmetic.
- Candidate acceptance folds in the regularizers, each one line:
  `min_child_hess` (reject thin children), `min_gain_to_split`, the L1
  soft threshold inside `score(G, H, α, λ)`
  ([`split.hpp`](../../include/bonsai/split.hpp)), monotone rejection and
  interaction-constraint masking ([chapter 6](6-regularization-and-constraints.md)).
- `HistogramNodeSplitFinder::find`: the feature loop, parallel, with the
  per-feature bests merged **serially in feature order** afterward so ties
  break identically to a serial scan (lowest feature id wins). There is
  also a *level* finder used by the oblivious grower which scores one
  shared split summed across a whole frontier
  (`update_best_for_feature_for_level`).
- Applying the winner: `split_node` in
  [`src/grower.cpp`](../../src/grower.cpp): a stable two-pass scatter of
  the parent's rows (bin-compare against the split bin, missing routed by
  `default_left`), then the subtraction trick from chapter 2.

The float threshold stored in the tree is `cuts[bin]`, so bin-space
routing during training and float-space routing at predict agree exactly:
an invariant several later features lean on (out-of-bag routing, DART).

## Try it

```bash
# Starve splits of evidence and watch trees shrink:
bonsai fit -c configs/california_housing.toml \
    --set tree.min_child_hess=200 --set booster.log_intervals=10
```

```python
# The learned NaN direction in action: predict with a missing feature.
import numpy as np, bonsai
m = bonsai.BonsaiRegressor(n_iters=50).fit(X, y)
x = X[:1].copy(); x[0, 0] = np.nan
m.predict(x)   # finite — NaN followed each split's default_left
```

## Gotchas & war stories

- **Ties need a law.** Two features with bit-equal gain must resolve the
  same way every run, or "deterministic training" is a lie. bonsai's rule:
  strictly-greater comparisons everywhere + merge in ascending feature
  order. The parallel finder was shaped around preserving exactly this.
- **`gain > best.gain` with `best.gain = 0`** means "no positive-gain
  split" and "no split" are the same condition: a node with nothing to
  say becomes a leaf even if depth remains.
- **Score both NaN routings even on NaN-free data**: cheap, and it makes
  fit-time behavior independent of whether missing values appear later at
  predict time.
