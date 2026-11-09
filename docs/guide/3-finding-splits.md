# 3. Finding splits

## The idea

A node holds a set of rows. A split sends each row left or right by comparing one feature to one threshold. The best split is the one with the highest gain: the children's combined score minus the parent's (defined below). A histogram stores one cell per bin, holding the summed gradients and hessians of the rows in that bin, with about 255 bins per feature. One scan over those cells scores every candidate threshold of the feature.

## The math

From chapter 1, a leaf with sums $(G, H)$ achieves (up to a constant) loss reduction proportional to its **score**:

```math
\mathrm{score}(G, H) = \frac{G^2}{H + \lambda}
```

A split's **gain** is what the children score over the parent:

```math
\text{gain} = \mathrm{score}(G_L, H_L) +
\mathrm{score}(G_R, H_R) - \mathrm{score}(G, H)
```

Intuition: $G^2$ rewards leaves whose gradients *agree* (a large sum of same-signed residuals is a mistake a leaf value can fix). Dividing by $H + \lambda$ turns the squared sum into a per-hessian average, where $H$ is the leaf's total hessian (it counts rows under squared error). The L2 penalty $\lambda$ stops a leaf with almost no hessian from scoring without bound. A split earns gain by separating rows with positive residuals from rows with negative ones.

The scan: walk cells left to right keeping a running $(G_L, H_L)$; the right side is the node total minus the prefix. Each of the ~255 positions is scored in O(1), each feature independently, and the best (feature, bin) wins.

**Missing values** get their own reserved bin, excluded from the prefix walk. When the node holds missing values for the feature, bonsai scores *both* routings (NaNs-left and NaNs-right) for each candidate and keeps the better (`default_left` on the split); with no missing rows it scores NaNs-left only, since both routings tie. That is XGBoost's "learned default direction": the data decides where missing belongs, per split.

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

Score each with the gain formula above and keep the larger. The missing bin's $(+0.2, 1.0)$ never enters the prefix; the splitter folds it into one side per routing and keeps the better score. This scan is the whole split search for one feature. The rest of the chapter covers how bonsai runs it across features in parallel with vector instructions while keeping results identical across runs.

## In bonsai

The split search is in [`src/split.cpp`](../../src/split.cpp):

- `update_best_for_feature_for_node`: the per-feature scan. Read it top to bottom: hoist the node score and the real (non-missing) totals from the feature's own totals, then for each cut cell accumulate the prefix and let `offer_cut` score the `default_left` routings via `split_sums_at` (both when the node has missing rows, one when it has none), the single source of truth for missing-routing arithmetic. The same loop sums the next scalar feature's cells, so that feature's totals are ready when its scan starts.
- `scan_lanes`: the same scan over eight features at once. A one-feature scan is limited by its two divisions per cell, which the scalar divider handles one at a time. `scan_features` therefore groups eight plain features into one `LaneGroup`. A plain feature has histogram storage for this node, is allowed, and has no monotone constraint. The group walks its cut cells together using vector instructions, one feature per lane (a lane is one slot of a vector register): a first pass stores the running left sums at every cut, which also yields each feature's totals, and a second pass scores the cuts from the stored sums. Each lane runs the scalar scan's operations in the scalar scan's order, so the gains are bit-identical. A feature under a monotone constraint takes the scalar scan, in feature order, between groups.
- Candidate acceptance folds in the regularizers, each one line: `min_child_hess` (reject a child holding less hessian than the floor, which counts rows under squared error), `min_gain_to_split`, the L1 soft threshold inside `score(G, H, α, λ)` ([`split.hpp`](../../include/bonsai/split.hpp)), monotone rejection and interaction-constraint masking ([chapter 6](6-regularization-and-constraints.md)).
- `HistogramNodeSplitFinder::find` and `find_parallel`: the feature loop, run serially or split into ranges of features across the worker threads, with the per-range bests merged **serially in feature order** afterward so ties break identically to a serial scan (lowest feature id wins). Both the leaf plane and the host level step drive `find_parallel`, so a node's features spread across every worker even when the frontier (the set of nodes waiting for a split) holds one node. The levelwise grower uses a separate *level* finder that picks one split shared by every node in the frontier. Each task scans eight features. It walks each parent once on the same lanes and adds the parent's children score to a per-bin accumulator, in parent order. When a cut is infeasible for a parent, it adds that parent's own score instead. After the last parent, the finder offers the summed gains as candidates. A host without AVX2 keeps the per-feature scalar scan (`update_best_for_feature_for_level`).
- Applying the winner: `host_partition` in [`src/step/primitives.hpp`](../../src/step/primitives.hpp) (called per split by `split_node` for leafwise and once per level by the level step for depthwise and levelwise): a stable two-pass scatter of the parent's rows (bin-compare against the split bin, missing routed by `default_left`), then the subtraction trick from chapter 2 (`finish_split`).

The float threshold stored in the tree is `cuts[bin]`, so bin-space routing during training and float-space routing at predict agree exactly. Out-of-bag routing ([chapter 5](5-sampling.md)) and [DART](7-early-stopping-and-dart.md) (dropping whole trees during training) depend on this agreement.

## Try it

Raise the evidence a split needs, then watch a missing feature still land on a finite leaf:

```python {.run}
import numpy as np
import bonsai

rng = np.random.default_rng(0)
X = rng.normal(size=(3000, 6)).astype(np.float32)
y = (X[:, 0] + X[:, 1] * X[:, 2] + rng.normal(0, 0.1, 3000)).astype(np.float32)

# min_child_hess is the least hessian each child must hold. Under squared error each row weighs 1,
# so 500 means at least 500 rows per child. Higher values give shallower trees and can underfit.
loose = bonsai.BonsaiRegressor(n_iters=50).fit(X, y)
strict = bonsai.BonsaiRegressor(n_iters=50, params={"tree.min_child_hess": 500}).fit(X, y)
print("loose  R2:", round(loose.score(X, y), 4))
print("strict R2:", round(strict.score(X, y), 4))

# Trained without missing values, so every split sends NaN left (default_left); the prediction is still finite.
x = X[:1].copy()
x[0, 0] = np.nan
print("NaN prediction:", float(np.asarray(loose.predict(x))[0]))
```

## Gotchas & war stories

- **Break ties the same way every run.** When two features have bit-equal gain, bonsai keeps the lowest feature id. It compares gains with strictly-greater everywhere and merges per-range bests in ascending feature order. The parallel finder preserves this order.
- **`gain > best.gain` with `best.gain = 0`** means "no positive-gain split" and "no split" are the same condition: a node with nothing to say becomes a leaf even if depth remains.
- **bonsai scores both NaN routings only where the missing bin holds rows.** When a node's missing bin is empty, the finder scores one routing and sets `default_left` to true, so a NaN seen later at predict time on that feature goes left. A model trained without missing values therefore sends NaN left at every split. Where the missing bin is non-empty, the second routing costs one more gain computation at each cut.
