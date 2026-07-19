# 8. Feature importance

## The idea

After training, the first question is "*which features mattered?*" Tree
ensembles offer an unusually direct answer, because every split names the
feature it used and the loss reduction it bought. Feature importance is
just bookkeeping over the trained trees, but the *kind* of bookkeeping
changes the answer, and the most popular kind is the most misleading.

Three classic flavors:

- **split** (a.k.a. weight/frequency): how many times was the feature
  split on?
- **gain**: how much total loss reduction did its splits produce?
- **cover**: how many rows passed through its splits? (Not implemented in
  bonsai; it rarely changes conclusions gain doesn't.)

## The math

Nothing beyond chapter 3. Every internal node already computed

```math
\text{gain} = \mathrm{score}(G_L, H_L) +
\mathrm{score}(G_R, H_R) - \mathrm{score}(G, H)
```

when it was created. Then, over all internal nodes $v$ of all trees:

```math
\text{split}[f] = \sum_v \mathbf{1}[v \text{ splits on } f] \qquad
\text{gain}[f] = \sum_v \text{gain}(v)\,\mathbf{1}[v \text{ splits on } f]
```

Why they disagree: a feature can be split on *often* for *small* gains.
Fine-grained, high-cardinality features invite many low-value splits;
a dominant feature may be spent in a handful of huge root-level splits and
never touched again. Split-count rewards the former; gain rewards the
latter. Gain is the answer to "which feature reduced the loss", which is
usually the question being asked; hence it's LightGBM's recommended type
and sklearn's `feature_importances_` semantics.

## In bonsai

The one design decision: gain must be **recorded at grow time**, because
it's not reconstructible from a stored tree (the tree keeps thresholds and
leaf values, not the histograms that produced them). So:

- Growers stamp each split's gain the moment they create an internal node:
  the `split_gains` bookkeeping next to `split_bins` in
  [`src/grower.cpp`](../../src/grower.cpp); `DenseTree` carries them
  per node id, `ObliviousTree` per level, and both serialize (model
  format v5).
- Accumulation is a ~20-line walk: `internal::accumulate_importance` in
  [`include/bonsai/booster.hpp`](../../include/bonsai/booster.hpp):
  for each internal node, `out[feature] += 1` or `+= gain`. That is the
  entire feature.
- Surfaces: `bonsai importance --model m.msgpack`
  ([`src/cli/importance.cpp`](../../src/cli/importance.cpp)) prints the
  sorted table; Python exposes `Model.feature_importance(type)`, the raw
  `BonsaiRegressor.importance(type)`, and sklearn-style
  `feature_importances_` (gain, normalized to sum to 1).

## Try it

```{.python .run}
import numpy as np
import bonsai

rng = np.random.default_rng(0)
n = 6000
f0 = rng.normal(size=n)           # a step: strong signal in one threshold
f1 = rng.uniform(-3.0, 3.0, n)    # a wiggle: signal spread over many splits
rest = rng.normal(size=(n, 4))
X = np.column_stack([f0, f1, rest]).astype(np.float32)
y = (4.0 * np.sign(f0) + np.sin(3.0 * f1) + rng.normal(0, 0.1, n)).astype(np.float32)

m = bonsai.BonsaiRegressor(n_iters=120).fit(X, y)
gain = np.asarray(m.importance("gain"))
split = np.asarray(m.importance("split"))
print("gain :", gain.round(1))
print("split:", split)
print("gain top:", int(gain.argmax()), " split top:", int(split.argmax()))
```

Read the two rankings against each other. Feature 0 tops **gain**: one
threshold on its step captures most of the loss. Feature 1 tops **split
count**: its wiggle has no single decisive cut, so trees carve it with
many small splits. One model, two rankings, both "correct": they answer
different questions.

On California Housing the same split appears, gain ranking median income
first and split count ranking Longitude and Latitude. LightGBM reproduces
it, pinned by `test_feature_importance_agreement`
([python/tests/test_bindings.py](../../python/tests/test_bindings.py)).

## What to distrust

Importance is the most over-interpreted number in applied ML. The failure
modes to keep in mind:

- **Split count favors high-cardinality features**: more distinct values
  means more candidate thresholds means more chances to be picked for
  marginal gains. An ID-like column can top split-importance while being
  pure leakage.
- **Correlated features split the credit arbitrarily.** If two features
  carry the same signal, the trees will use whichever wins each local
  tie; importance may assign 90/10 or 50/50 between them run to run.
  Neither number means "the other feature doesn't matter".
- **It's a training-set quantity.** Gain measures loss reduction *on the
  training data*, regularization and all. It says nothing about
  generalization; a feature can carry high gain while overfitting noise.
- When the stakes are real, corroborate with **permutation importance**
  (shuffle a feature at predict time, watch the metric drop) or **SHAP**
  (per-prediction attribution with a consistency guarantee), both on the
  backlog ([feature_gap.md](https://github.com/daniel-m-campos/bonsai/blob/main/docs/feature_gap.md) row 15).

## Gotchas & war stories

- The agreement test originally asserted MedInc would top *both* types,
  and failed, with Longitude winning split-count at 1,900 to 1,129. That
  "failure" is the textbook lesson of this chapter, and LightGBM
  reproduced it exactly. The test now asserts the disagreement.
- **Gain is stamped or it's gone.** Storing it costs 4 bytes per node;
  reconstructing it later would require re-running training. If you're
  designing a tree format, record gain (and cover, while you're at it) on
  day one.
- Normalized `feature_importances_` (sums to 1) is for *comparing features
  within one model*: the absolute gains are loss-scale-dependent and not
  comparable across datasets or objectives.
