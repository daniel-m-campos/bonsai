# 6 — Regularization & constraints

## The idea

Boosting will happily memorize the training set; every knob here limits
*how* it's allowed to fit. Three families:

1. **Shrink the leaves** — L2 (`lambda_l2`) and L1 (`lambda_l1`) penalties
   on leaf values.
2. **Starve the splits** — `min_child_hess`, `min_data_in_leaf`,
   `min_gain_to_split`, `max_depth`/`max_leaves`, and per-tree feature
   subsampling (`feature_fraction`).
3. **Constrain the shape** — monotone constraints (predictions must be
   non-decreasing/-increasing in a feature) and interaction constraints
   (features may only combine within declared groups). These encode domain
   knowledge rather than fight variance.

## The math

**L2** appears as the $+\lambda$ in every score and leaf value — it damps
leaves supported by little evidence. **L1** soft-thresholds the gradient
sum:

```math
T(G, \alpha) = \begin{cases} G - \alpha & G > \alpha \\ G + \alpha & G < -\alpha \\ 0 & \text{otherwise} \end{cases}
\qquad w^{\ast} = -\frac{T(G, \alpha)}{H + \lambda}, \quad
\operatorname{score} = \frac{T(G, \alpha)^2}{H + \lambda}
```

so leaves with $|G| \le \alpha$ are exactly zero (sparsity), and
everything else shrinks toward zero by $\alpha$.

**Monotone ($+1$ on feature $j$)**: every split on $j$ must have
$w_L \le w_R$, *and* that must keep holding as descendants refine — so
after a constrained split, the left subtree's values are capped and the
right subtree's floored at the midpoint $(w_L + w_R)/2$. Bounds inherit
downward; leaf values clamp into them.

**Interaction groups**: a node may split on feature $f$ only if some
declared group contains $f$ together with *every* feature already used on
the path from the root (features outside all groups may keep splitting
alone). Prediction becomes a sum of functions over the groups only.

## In bonsai

- **L1/L2** — `l1_thresholded`, the four-argument `score`, and
  `bounded_leaf_weight` in [`include/bonsai/split.hpp`](../../include/bonsai/split.hpp).
  The same three functions serve the finders and the growers, so gain and
  leaf values can't disagree about the penalty.
- **feature_fraction** — `sample_features` in
  [`src/grower.cpp`](../../src/grower.cpp): a per-tree sorted draw from a
  grower-owned rng (`tree.feature_seed`); unselected features get
  zero-binned placeholder histograms the finders skip.
- **Monotone** — two touch points. Rejection: in the candidate loop of
  [`src/split.cpp`](../../src/split.cpp), skip when
  $\text{mc} \cdot (w_R - w_L) < 0$, using *bounded* child weights. Propagation: `propagate_monotone_bounds`
  in `src/grower.cpp` fences children at the midpoint via
  `SplitInput::lo/hi`; `finalize_as_leaf` clamps into them. Config:
  `tree.monotone_constraints = [1, 0, -1, ...]` (or `--set
  tree.monotone_constraints=1,0,-1`).
- **Interaction** — `SplitInput::allowed/path` carry the permitted set
  down the tree; `allowed_features` / `propagate_interaction_state`
  (`src/grower.cpp`) recompute it per split; the finder masks excluded
  features. Config: `tree.interaction_constraints = ["0,1", "2,3"]`
  (CLI: `0+1,2+3`).

The oblivious grower rejects both constraint types at construction rather
than silently ignoring them.

## Try it

```bash
# Monotone: house value non-decreasing in median income —
# every library pays the same ~2% RMSE for the guarantee (feature_gap §6):
uv run scripts/compare.py --config configs/california_housing.toml \
    --hp tree.monotone_constraints=1 --growers depthwise,leafwise --samplers all_rows

# Interaction: economics {0-3} may not mix with geography {4-7} — ~9% RMSE,
# identically across bonsai/xgboost/lightgbm (feature_gap §7):
uv run scripts/compare.py --config configs/california_housing.toml \
    --hp tree.interaction_constraints=0+1+2+3,4+5+6+7 \
    --growers depthwise,leafwise --samplers all_rows
```

The unit tests are readable specs: `[grower][monotone]` asserts a
non-monotone dataset yields a provably monotone prediction curve;
`[grower][interaction]` walks every root-to-leaf path and asserts groups
never mix ([tests/unit/test_grower.cpp](../../tests/unit/test_grower.cpp)).

## Gotchas & war stories

- **Rejecting the split isn't enough for monotonicity.** A split on an
  *unconstrained* feature can still create descendants that later violate
  the constrained feature's ordering — that's what the inherited
  `lo/hi` bounds prevent. Basic-mode implementations that skip
  propagation produce trees that are monotone split-by-split and
  non-monotone end-to-end.
- **L1's scale is data-scale.** `lambda_l1=100` was needed on
  YearPredictionMSD (leaf gradient sums in the thousands) to move RMSE at
  all; the same value on California Housing would zero half the leaves.
- **A feature outside every interaction group isn't banned** — it can
  still split, just never *with* anything else on the path.
