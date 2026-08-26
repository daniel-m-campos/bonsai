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
XGBoost's "learned default direction": the data decides where missing
belongs, per split.

### A worked example

Six rows at one node, one feature with 4 bins (0 to 2 real, 3 missing):

| row | bin | $g$ | $h$ |
|-----|-----|------|-----|
| 0 | 1 | +0.3 | 1.0 |
| 1 | 0 | -0.5 | 1.0 |
| 2 | 2 | +0.1 | 1.0 |
| 3 | 0 | -0.4 | 1.0 |
| 4 | 3 | +0.2 | 1.0 |
| 5 | 1 | +0.6 | 1.0 |

The build pass produces one cell per bin:

| bin | sum_grad | sum_hess | |
|-----|----------|----------|---|
| 0 | -0.9 | 2.0 | |
| 1 | +0.9 | 2.0 | |
| 2 | +0.1 | 1.0 | |
| 3 | +0.2 | 1.0 | missing |

Totals over the real bins: $G = +0.1$, $H = 5.0$. The scan sweeps the two interior cuts, putting bins $0..b-1$ on the left:

- cut at 0|1: $G_L = -0.9,\ H_L = 2.0$; right side by subtraction, $G_R = +1.0,\ H_R = 3.0$
- cut at 1|2: $G_L = 0.0,\ H_L = 4.0$; $G_R = +0.1,\ H_R = 1.0$

Score each with the gain formula above and keep the larger. The missing bin's $(+0.2, 1.0)$ never enters the prefix; the splitter folds it into one side per routing and keeps the better score. That is the whole inner game; everything else in this chapter is making it fast, parallel, and deterministic.

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
  also a *level* finder used by the levelwise grower which scores one
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

Raise the evidence a split needs, then watch a missing feature still land
on a finite leaf:

```{.python .run}
import numpy as np
import bonsai

rng = np.random.default_rng(0)
X = rng.normal(size=(3000, 6)).astype(np.float32)
y = (X[:, 0] + X[:, 1] * X[:, 2] + rng.normal(0, 0.1, 3000)).astype(np.float32)

# min_child_hess is the split's evidence floor; raise it and trees stay shallow.
loose = bonsai.BonsaiRegressor(n_iters=50).fit(X, y)
strict = bonsai.BonsaiRegressor(n_iters=50, params={"tree.min_child_hess": 500}).fit(X, y)
print("loose  R2:", round(loose.score(X, y), 4))
print("strict R2:", round(strict.score(X, y), 4))

# The learned NaN direction: a missing feature follows each split's default_left.
x = X[:1].copy()
x[0, 0] = np.nan
print("NaN prediction:", float(np.asarray(loose.predict(x))[0]))
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
