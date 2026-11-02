# 8. Feature importance

## The idea

After training, the first question is "*which features mattered?*" Tree ensembles offer an unusually direct answer, because every split names the feature it used and the loss reduction it bought. Feature importance is just bookkeeping over the trained trees, but the *kind* of bookkeeping changes the answer. Use gain, not split count, to ask which feature reduced the loss. Split count favors features with many distinct values and can rank an ID-like feature first (see [What to distrust](#what-to-distrust)).

Three classic flavors:

- **split** (a.k.a. weight/frequency): how many times was the feature split on?
- **gain**: how much total loss reduction did its splits produce?
- **cover**: how many rows passed through its splits? (Not implemented in bonsai; it rarely changes conclusions gain doesn't.)

## The math

Nothing beyond chapter 3. Every internal node already computed

```math
\text{gain} = \mathrm{score}(G_L, H_L) +
\mathrm{score}(G_R, H_R) - \mathrm{score}(G, H)
```

when it was created. Here $G$ and $H$ are the sums of the gradients and hessians of the rows in a node ($G_L, H_L$ for the left child, $G_R, H_R$ for the right). $\mathrm{score}(G, H)$ is that node's loss-reduction score from [chapter 3](3-finding-splits.md). Then, over all internal nodes $v$ of all trees:

```math
\text{split}[f] = \sum_v \mathbf{1}[v \text{ splits on } f] \qquad
\text{gain}[f] = \sum_v \text{gain}(v)\,\mathbf{1}[v \text{ splits on } f]
```

Why they disagree: a feature can be split on often, with each split removing little loss. A feature with many distinct values (high cardinality) offers many candidate thresholds, so trees pick it for many marginal splits. A dominant feature may instead win a few splits near the roots that each remove a large share of the loss, and then add little below them. Split count rewards the first pattern; gain rewards the second. Gain answers "which feature reduced the loss", which is usually the question being asked. It is also what `feature_importances_` returns, normalized to sum to 1.

## In bonsai

The one design decision: gain must be **recorded at grow time**, because it's not reconstructible from a stored tree (the tree keeps thresholds and leaf values, not the histograms that produced them). So:

- Growers stamp each split's gain the moment they create an internal node: the `split_gains` bookkeeping next to `split_bins` in [`src/grower_impl.hpp`](../../src/grower_impl.hpp); `DenseTree` carries them per node id, `ObliviousTree` per level, and both serialize. Gains have been in the model format since v5; the format is at v7 today (`k_format_version`, [`src/io/model.cpp`](../../src/io/model.cpp)).
- Accumulation is a ~20-line walk: `detail::accumulate_importance` in [`include/bonsai/detail/ensemble_ops.hpp`](../../include/bonsai/detail/ensemble_ops.hpp): for each internal node, `out[feature] += 1` or `+= gain`. That is the entire feature.
- Surfaces: `bonsai importance --model m.msgpack` ([`src/cli/importance.cpp`](../../src/cli/importance.cpp)) prints the sorted table; Python exposes `Model.feature_importance(type)`, the raw `BonsaiRegressor.importance(type)`, and sklearn-style `feature_importances_` (gain, normalized to sum to 1).

## Try it

```python {.run}
import numpy as np
import bonsai

rng = np.random.default_rng(0)
n = 6000
f0 = rng.normal(size=n)  # a step: strong signal in one threshold
f1 = rng.uniform(-3.0, 3.0, n)  # a wiggle: signal spread over many splits
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

Read the two rankings against each other. Feature 0 tops **gain**: one threshold on its step accounts for 97% of the total gain. Feature 1 tops **split count**: its sine signal has no single threshold that removes most of the loss, so the trees split on it 2,258 times against 560 for feature 0, while it holds 3% of the gain. One model, two rankings, both "correct": they answer different questions.

On California Housing the same split appears, gain ranking median income first and split count ranking Longitude and Latitude. LightGBM reproduces it, pinned by `test_feature_importance_agreement` ([python/tests/test_estimators.py](../../python/tests/test_estimators.py)).

## What to distrust

Importance describes the trained trees, not the world, which makes it easy to misread:

- **Split count favors high-cardinality features**: an ID-like feature can top split count while carrying no real signal. It has many distinct values, which give many candidate thresholds, so it has many chances to win a split that removes little loss, and the trees use it to memorize training rows.
- **Correlated features divide importance between them unpredictably.** If two features carry the same signal, each tree uses whichever one gives the better gain at that node, so importance may be 90/10 in one training run and 50/50 in another. Neither number means "the other feature doesn't matter".
- **It's a training-set quantity.** Gain measures loss reduction *on the training data*, regularization and all. It says nothing about generalization; a feature can carry high gain while overfitting noise.
- When the stakes are real, corroborate with **SHAP** (exact TreeSHAP shipped as `pred_contribs`; [chapter 15](15-explaining-predictions.md) is its chapter) or **permutation importance** (shuffle a feature at predict time, watch the metric drop; a five-line loop over `predict`).
- If the ranking is tempting you to *drop* features, that is a different decision with its own measured answer: [chapter 14](14-feature-selection.md).

## Gotchas & war stories

- The agreement test originally asserted MedInc would top *both* types, and failed, with Longitude winning split-count at 1,900 to 1,129. That disagreement is the difference between the two importance types, and LightGBM reproduced it exactly. The test now asserts the disagreement.
- **bonsai records gain when it grows a node, because a saved tree cannot recover it.** The record costs 4 bytes per node (per level in oblivious trees). Recovering it later means training again.
- Normalized `feature_importances_` (sums to 1) is for *comparing features within one model*: the absolute gains are loss-scale-dependent and not comparable across datasets or objectives.
