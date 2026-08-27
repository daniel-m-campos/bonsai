# 6. Regularization & constraints

## The idea

Boosting will happily memorize the training set; every knob here limits
*how* it's allowed to fit. Three families:

1. **Shrink the leaves**: L2 (`lambda_l2`) and L1 (`lambda_l1`) penalties
   on leaf values.
2. **Starve the splits**: `min_child_hess`, `min_data_in_leaf`,
   `min_gain_to_split`, `max_depth`/`max_leaves`, and per-tree feature
   subsampling (`feature_fraction`).
3. **Constrain the shape**: monotone constraints (predictions must be
   non-decreasing/-increasing in a feature) and interaction constraints
   (features may only combine within declared groups). These encode domain
   knowledge rather than fight variance.

## The math

**L2** appears as the $+\lambda$ in every score and leaf value: it damps
leaves supported by little evidence. **L1** soft-thresholds the gradient
sum:

```math
T(G, \alpha) = \begin{cases} G - \alpha & G > \alpha \\ G + \alpha & G < -\alpha \\ 0 & \text{otherwise} \end{cases}
\qquad w^{\ast} = -\frac{T(G, \alpha)}{H + \lambda}, \quad
\mathrm{score} = \frac{T(G, \alpha)^2}{H + \lambda}
```

so leaves with $|G| \le \alpha$ are exactly zero (sparsity), and
everything else shrinks toward zero by $\alpha$.

**Monotone ($+1$ on feature $j$)**: every split on $j$ must have
$w_L \le w_R$, *and* that must keep holding as descendants refine, so
after a constrained split, the left subtree's values are capped and the
right subtree's floored at the midpoint $(w_L + w_R)/2$. Bounds inherit
downward; leaf values clamp into them. Any value in $[w_L, w_R]$ would
keep the whole tree monotone in $x_j$; the midpoint just splits the
remaining headroom evenly, so neither subtree starts already pinned to
its bound.

**Interaction groups**: a node may split on feature $f$ only if some
declared group contains $f$ together with *every* feature already used on
the path from the root (features outside all groups may keep splitting
alone). Prediction becomes a sum of functions over the groups only.

## In bonsai

- **L1/L2**: `l1_thresholded`, the four-argument `score`, and
  `bounded_leaf_weight` in [`include/bonsai/split.hpp`](../../include/bonsai/split.hpp).
  The same three functions serve the finders and the growers, so gain and
  leaf values can't disagree about the penalty.
- **feature_fraction**: `sample_features` in
  [`src/grower.cpp`](../../src/grower.cpp): a per-tree sorted draw from a
  grower-owned rng (`tree.feature_seed`); unselected features get
  zero-binned placeholder histograms the finders skip.
- **Monotone**: two touch points on the node-splitting growers (depthwise, leafwise). Rejection: in the candidate loop of [`src/split.cpp`](../../src/split.cpp), skip when $\text{mc} \cdot (w_R - w_L) < 0$, using *bounded* child weights. Propagation: `propagate_monotone_bounds` in `src/grower.cpp` fences children at the midpoint via `SplitInput::lo/hi`; `finalize_as_leaf` clamps into them. Levelwise takes a different route entirely, described below. Config: `tree.monotone_constraints = [1, 0, -1, ...]` (or `--set tree.monotone_constraints=1,0,-1`). From Python the key also takes a mapping keyed by feature name, `{"age": 1, "debt": -1}`. The python layer resolves it against the training data's feature names and hands the core the same positional list. Features the mapping leaves out are free (0). A name the data does not carry raises, listing the offenders.
- **Interaction**: `SplitInput::allowed/path` carry the permitted set
  down the tree; `allowed_features` / `propagate_interaction_state`
  (`src/grower.cpp`) recompute it per split; the finder masks excluded
  features. Config: `tree.interaction_constraints = ["0,1", "2,3"]`
  (CLI: `0+1,2+3`).

The levelwise grower rejects both constraint types at construction rather
than silently ignoring them.

## Try it

```{.python .run}
import numpy as np
import bonsai

rng = np.random.default_rng(0)
X = rng.normal(size=(4000, 8)).astype(np.float32)
y = (X[:, 0] * 2.0 + X[:, 1] + rng.normal(0, 0.1, 4000)).astype(np.float32)

# Monotone: prediction non-decreasing in feature 0.
mono = bonsai.BonsaiRegressor(
    n_iters=60, grower="depthwise",
    params={"tree.monotone_constraints": "1,0,0,0,0,0,0,0"}).fit(X, y)

# The same constraint by name. A plain array is named f0..fN; a Dataset built
# with feature_names=, a fit on a frame with string columns, or fit(X, y,
# feature_names=...) carries the names you gave it.
named = bonsai.BonsaiRegressor(
    n_iters=60, grower="depthwise",
    params={"tree.monotone_constraints": {"f0": 1}}).fit(X, y)

# Interaction: features 0-3 may not mix with 4-7 on any path.
inter = bonsai.BonsaiRegressor(
    n_iters=60, grower="depthwise",
    params={"tree.interaction_constraints": "0+1+2+3,4+5+6+7"}).fit(X, y)

print("monotone    R2:", round(mono.score(X, y), 4))
print("by name     R2:", round(named.score(X, y), 4))
print("interaction R2:", round(inter.score(X, y), 4))
```

The unit tests are readable specs: `[grower][monotone]` asserts a
non-monotone dataset yields a provably monotone prediction curve;
`[grower][interaction]` walks every root-to-leaf path and asserts groups
never mix ([tests/unit/test_grower.cpp](../../tests/unit/test_grower.cpp)).

## Levelwise gets there a different way

The scheme above needs a per-node bound to inherit, which a levelwise (oblivious) tree has nowhere to put: every node at a level shares one split, so there is no per-node state to fence. For a long time that is why levelwise rejected the constraint outright.

The shared split turns out to be the thing that makes it easy. A levelwise leaf index *is* the bit vector of level outcomes, so the leaves form a perfect lattice: bits from constrained features induce a partial order over leaves, and bits from free features cut the leaves into groups that are independent of each other. Ordering the leaves within a group and running weighted isotonic regression along that order gives the nearest leaf table that satisfies every constraint. The structure of the tree never changes; only the values at the bottom move.

That is [`src/monotone.cpp`](../../src/monotone.cpp), called once per tree between building the leaf table and stamping rows, with the leaves' hessians as weights, because weighted isotonic regression on the Newton step is the constrained minimiser of the same second-order objective the splits were chosen under. It costs nothing on the GPU plane either: the leaf table is built on the host in both planes, so the projection lands before the table is uploaded.

Per-tree is enough for the whole model. Monotone functions are closed under addition and multiplication by a positive number, and a boosted model is `init + lr * sum(trees)` with `lr > 0`, so every tree monotone implies the ensemble monotone. The end-to-end violation is exactly zero, not merely small.

Two honest caveats. With two or more constrained features the leaves form a partial order rather than a chain, and the projection runs along one linear extension of it: every constraint holds, but the result is not quite the L2-nearest monotone table (a single constrained feature is exact). And interaction constraints are still rejected on levelwise, because they constrain which features may share a path rather than how leaf values are ordered, and no projection can express that.

CatBoost supports monotone constraints on its `SymmetricTree` policy and *only* there, which is the mirror image of where bonsai stood before this: the same lattice argument, reached from the other side.

## Gotchas & war stories

- **Rejecting the split isn't enough for monotonicity.** A split on an
  *unconstrained* feature can still create descendants that later violate
  the constrained feature's ordering; that's what the inherited
  `lo/hi` bounds prevent. Basic-mode implementations that skip
  propagation produce trees that are monotone split-by-split and
  non-monotone end-to-end.
- **Leaf renewal can undo a monotone constraint.** MAE, Huber and Quantile replace every leaf value with a robust statistic of its residuals *after* the tree is grown, and that statistic has no reason to respect an ordering the growth-time machinery established. Levelwise handles it by projecting a second time, over the renewed table, with row counts as weights, since a renewed value is not a Newton step. The node-splitting growers have no equivalent second pass and do still violate under those three objectives, by around 1e-2 on a feature whose true relationship is monotone.
- **L1's scale is data-scale.** `lambda_l1=100` was needed on
  YearPredictionMSD (leaf gradient sums in the thousands) to move RMSE at
  all; the same value on California Housing would zero half the leaves.
- **A feature outside every interaction group isn't banned**: it can
  still split, just never *with* anything else on the path.
