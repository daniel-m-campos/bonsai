# 6. Regularization & constraints

## The idea

Boosting fits the training set ever more closely unless you limit it. The parameters on this page limit that fit, in three groups:

1. **Shrink the leaves**: L2 (`lambda_l2`) and L1 (`lambda_l1`) penalties on leaf values.
2. **Limit the splits**: six parameters. `min_data_in_leaf` (default 20) lets a node split only if it holds at least twice that many rows. `min_child_hess` (default 1.0) rejects a split when either child has too little hessian (the second derivative of the loss, which counts rows under squared error). `min_gain_to_split` (default 0.0) rejects splits that reduce the loss by less than that amount. `max_depth` (default 6) and `max_leaves` (default 31, leafwise growth only) cap tree size. `feature_fraction` (default 1.0) trains each tree on a random subset of the features. Each one gives up some training fit for lower variance.
3. **Constrain the shape**: monotone constraints (predictions must be non-decreasing/-increasing in a feature) and interaction constraints (features may only combine within declared groups). These encode domain knowledge rather than fight variance.

## The math

**L2** appears as the $+\lambda$ in every score and leaf value: it damps leaves supported by little evidence. **L1** soft-thresholds the gradient sum:

```math
T(G, \alpha) = \begin{cases} G - \alpha & G > \alpha \\ G + \alpha & G < -\alpha \\ 0 & \text{otherwise} \end{cases}
\qquad w^{\ast} = -\frac{T(G, \alpha)}{H + \lambda}, \quad
\mathrm{score} = \frac{T(G, \alpha)^2}{H + \lambda}
```

so leaves with $|G| \le \alpha$ are exactly zero (sparsity), and everything else shrinks toward zero by $\alpha$.

**Monotone ($+1$ on feature $j$)**: every split on $j$ must have $w_L \le w_R$, *and* that must keep holding as descendants refine, so after a constrained split, the left subtree's values are capped and the right subtree's floored at the midpoint $(w_L + w_R)/2$. Bounds inherit downward; leaf values clamp into them. Any value in $[w_L, w_R]$ would keep the whole tree monotone in $x_j$; the midpoint just splits the remaining headroom evenly, so neither subtree starts already pinned to its bound.

**Interaction groups**: a node may split on feature $f$ only if some declared group contains $f$ together with *every* feature already used on the path from the root (features outside all groups may keep splitting alone). Prediction becomes a sum of functions over the groups only.

## In bonsai

- **L1/L2**: `l1_thresholded`, the four-argument `score`, and `bounded_leaf_weight` in [`include/bonsai/split.hpp`](../../include/bonsai/split.hpp). The same three functions serve the finders and the growers, so gain and leaf values can't disagree about the penalty.
- **feature_fraction** (default 1.0): each tree trains on a random subset of the features, which decorrelates trees. A strong feature can be missing from some trees. `feature_sample` in [`include/bonsai/grower.hpp`](../../include/bonsai/grower.hpp) draws the subset for each tree. The draw is seeded from `booster.random_seed` together with `tree.feature_seed`. The finders skip the features a tree did not draw.
- **Monotone**: two touch points on the node-splitting growers (depthwise, leafwise). Levelwise (one shared split per level) takes a different route, described below.
    - Rejection: the candidate loop in [`src/split.cpp`](../../src/split.cpp) skips a split when $\text{mc} \cdot (w_R - w_L) < 0$, using *bounded* child weights.
    - Propagation: `propagate_monotone_bounds` in `src/grower_impl.hpp` sets the midpoint of the two child weights as a bound through `SplitInput::lo/hi`. For mc = +1 the left child gets an upper bound and the right child a lower bound. For mc = -1 the two bounds swap sides. `finalize_as_leaf` clamps leaf values into the bounds.
    - Config: `tree.monotone_constraints = [1, 0, -1, ...]` (or `--set tree.monotone_constraints=1,0,-1`).
    - Python: the key also takes a mapping by feature name, `{"age": 1, "debt": -1}`. The Python layer resolves it against the training data's feature names and passes the core the same positional list. Features the mapping omits are free (0). A name the data does not carry raises, listing the offenders.
- **Interaction**: `SplitInput::allowed/path` carry the permitted set down the tree; `allowed_features` / `propagate_interaction_state` (`src/grower_impl.hpp`) recompute it per split; the finder masks excluded features. Config: `tree.interaction_constraints = ["0,1", "2,3"]` (CLI: `0+1,2+3`).

The levelwise grower rejects interaction constraints at construction instead of ignoring them. It honours monotone constraints by projecting each finished leaf table, described [below](#levelwise-gets-there-a-different-way).

## Try it

```python {.run}
import numpy as np
import bonsai

rng = np.random.default_rng(0)
X = rng.normal(size=(4000, 8)).astype(np.float32)
y = (X[:, 0] * 2.0 + X[:, 1] + rng.normal(0, 0.1, 4000)).astype(np.float32)

# Monotone: prediction non-decreasing in feature 0.
mono = bonsai.BonsaiRegressor(
    n_iters=60, grower="depthwise", params={"tree.monotone_constraints": "1,0,0,0,0,0,0,0"}
).fit(X, y)

# The same constraint by name. A plain array is named f0..fN; a Dataset built
# with feature_names=, a fit on a frame with string columns, or fit(X, y,
# feature_names=...) carries the names you gave it.
named = bonsai.BonsaiRegressor(
    n_iters=60, grower="depthwise", params={"tree.monotone_constraints": {"f0": 1}}
).fit(X, y)

# Interaction: features 0-3 may not mix with 4-7 on any path.
inter = bonsai.BonsaiRegressor(
    n_iters=60, grower="depthwise", params={"tree.interaction_constraints": "0+1+2+3,4+5+6+7"}
).fit(X, y)

print("monotone    R2:", round(mono.score(X, y), 4))
print("by name     R2:", round(named.score(X, y), 4))
print("interaction R2:", round(inter.score(X, y), 4))
```

The unit tests are readable specs: `[grower][monotone]` asserts a non-monotone dataset yields a provably monotone prediction curve; `[grower][interaction]` walks every root-to-leaf path and asserts groups never mix ([tests/unit/test_grower.cpp](../../tests/unit/test_grower.cpp)).

## Levelwise gets there a different way

Levelwise grows a monotone model by projecting each tree's leaf table onto the nearest ordered table, since a levelwise (oblivious) tree, one split shared by every node at a level, has no per-node bound to inherit. The end-to-end violation is exactly zero (2.0.0 and later). Interaction constraints are still rejected.

Because every node at a level shares one split, a leaf's index is the list of its level outcomes (one bit per level), so the leaves form a lattice (a set where any two elements have a defined meet and join). Bits from constrained features put some leaves in a partial order (leaf A must not exceed leaf B), and bits from free features cut the leaves into groups that are independent of each other. Within a group, weighted isotonic regression (fitting the closest sequence that never moves against the constraint direction, weighting each leaf by its hessian) gives a leaf table that satisfies every constraint. The structure of the tree never changes; only the values at the bottom move.

That is [`src/monotone.cpp`](../../src/monotone.cpp), called once per tree between building the leaf table and stamping rows, with the leaves' hessians as weights, because weighted isotonic regression on the Newton step is the constrained minimiser of the same second-order objective the splits were chosen under. It adds no GPU work: the leaf table is built on the host for both the CPU and the GPU grower, so the projection runs before the table is uploaded to the device.

Per-tree is enough for the whole model. Monotone functions are closed under addition and multiplication by a positive number, and a boosted model is `init + lr * sum(trees)` with `lr > 0`, so every tree monotone implies the ensemble monotone. The end-to-end violation is exactly zero, not merely small.

Levelwise monotone constraints have two limits. First, on data that resists the constraint, the projection costs more accuracy than the depthwise veto: California Housing with +1 on MedInc loses 3.38% projected against 1.73% vetoed (200 rounds, lr 0.05, depth 6, each grower against its own unconstrained baseline; [decision 115](../decisions.md#115-levelwise-honours-monotone-constraints-by-projecting-its-leaves-adopted)). On data that agrees with the constraint it costs less, -0.06% against 1.90% on a synthetic whose truth is monotone. Second, with two or more constrained features the leaves form a partial order rather than a chain, and the projection runs along one linear extension of it: every constraint holds, but the result is not quite the L2-nearest monotone table (a single constrained feature is exact). Interaction constraints are still rejected on levelwise, because they constrain which features may share a path rather than how leaf values are ordered, and no projection can express that.

CatBoost supports monotone constraints on its `SymmetricTree` grow policy and, per decision 115, not on its non-symmetric ones. bonsai was the other way round until 2.0.0: depthwise and leafwise honoured the constraint and levelwise, which builds symmetric (oblivious) trees, refused it ([decision 115](../decisions.md#115-levelwise-honours-monotone-constraints-by-projecting-its-leaves-adopted)).

## Gotchas & war stories

- **Rejecting the split isn't enough for monotonicity.** A split on an *unconstrained* feature can still create descendants that later violate the constrained feature's ordering; that's what the inherited `lo/hi` bounds prevent. A grower that vetoes violating splits but passes no bounds down produces trees that are monotone at each split and non-monotone end to end.
- **Leaf renewal would undo a monotone constraint without a second step.** MAE, Huber and Quantile replace every leaf value with a robust statistic of its residuals *after* the tree is grown, and that statistic has no reason to respect an ordering the growth-time machinery established. Levelwise handles it by projecting a second time, over the renewed table, with row counts as weights, since a renewed value is not a Newton step. The depthwise and leafwise growers hand each leaf's monotone bounds out with the tree and renewal clamps the renewed value back into them, so their violation is exactly zero as well.
- **Set `lambda_l1` relative to your leaf gradient sums.** The penalty subtracts a fixed amount from each sum, so a value that matters on one dataset can do nothing on another. On YearPredictionMSD, where sums run into the thousands, `lambda_l1=100` moved RMSE by under 0.1% for every library that has the knob ([the feature-gap record](https://github.com/daniel-m-campos/bonsai/blob/main/benchmarks/feature-gap-2026-07.md), section 4).
- **A feature outside every interaction group isn't banned**: it can still split, just never *with* anything else on the path.
