# 12. Multiclass: softmax boosting

## The idea

Binary classification needed one score per row ([chapter 1](1-gradient-boosting.md)); $K$ classes need $K$. Gradient boosting's answer is almost embarrassingly direct: keep $K$ running score columns, and each round grow **one tree per class**, each fitting the gradient of the softmax cross-entropy with respect to *its* class's score. Nothing about trees, histograms, splits, or sampling changes: the same grower that fits house prices fits "the gradient of class 3". All the multiclass machinery lives one level up, in a booster that owns the $K$-column score matrix.

Prediction is then a vote: sum each class's trees, take the argmax (or a softmax if you want probabilities).

## The math

Row $i$ has scores $s_1, \dots, s_K$ and a one-hot target $y$. Softmax turns scores into probabilities and cross-entropy scores them:

```math
p_k = \frac{e^{s_k}}{\sum_j e^{s_j}}, \qquad \mathcal{L} = -\log p_{y}
```

The derivatives that feed the trees are the classic softmax pair:

```math
\frac{\partial \mathcal{L}}{\partial s_k} = p_k - \mathbf{1}[y = k], \qquad \frac{\partial^2 \mathcal{L}}{\partial s_k^2} = p_k (1 - p_k)
```

The gradient is beautifully interpretable ("how much probability did you give class $k$ beyond what it deserved"), and the Hessian is largest where the model is uncertain ($p_k \approx \tfrac12$) so confident rows contribute little curvature. Two practical adjustments the code makes: the Hessian ships as the true diagonal $p_k(1 - p_k)$ (a factor-2 'xgboost convention' variant halves every Newton step: the quality campaign measured it as exactly 2× the iterations to match lightgbm at the same learning rate, so bonsai matches lightgbm's step size), and it is floored at $10^{-6}$, because a row the model is *certain* about has $p_k(1-p_k) \to 0$ and would otherwise divide a leaf value by nothing.

Note the diagonal approximation quietly taken: the true softmax Hessian couples classes ($\partial^2 \mathcal{L} / \partial s_k \partial s_j = -p_k p_j$), and every GBT library ignores the off-diagonal terms so each class's tree can be fit independently. It costs iterations, not correctness: the coupled information re-enters through the updated probabilities next round.

## In bonsai

Everything is in [`include/bonsai/multiclass_booster.hpp`](../../include/bonsai/multiclass_booster.hpp):

- **Why a separate booster at all**: the 1-D `Objective` concept says gradients are a function of `(preds, targets)`, one column. The $K$-output shape can't be expressed there without contorting every other objective, so `softmax` dispatches to its own `IBooster` implementation (`BoosterFor<{softmax, G, Sa}>` in the registry) and [`SoftmaxObjective`](../../src/objective.cpp)'s 1-D methods deliberately throw. The dispatch mechanism, not a flag inside the booster, decides the shape: one of the places the typelist registry ([architecture/6](../architecture/6-dispatch.md)) earns its keep.
- **The round loop** (`update_one_iter`): scores are a flat `n × K` row-major matrix; for each class $k$, a softmax pass fills `grad_/hess_` for column $k$, the *same* row sampler draws (GOSS and bagging work unchanged), and the *same* grower grows a tree. `trees_` is flat and round-major: class $k$ of round $r$ sits at `r * K + k`, which is all `predict_at(n_rounds)` needs to truncate cleanly.
- **Prediction**: `raw_scores` accumulates each tree into its class column; `predict` emits argmax class ids; `eval` is the multiclass logloss on the raw scores.
- **What composes for free, and what needed work**: bagging, GOSS, every grower (including the CUDA ones: the engine sees $K$ ordinary trees per round), feature importance, warm start, and the model format all work unchanged, because they never knew about classes. The seams that *do* know the output shape needed their own passes: early stopping runs through a shape-agnostic score matrix (`score_width()` = $K$), and `pred_contribs` returns per-class slices, `(n, K, n_{\text{features}}+1)`, each class's slice sums to that class's raw score.

## Try it

```bash
# 3-class synthetic via the Python module:
python3 - <<'EOF'
import numpy as np, bonsai
rng = np.random.default_rng(0)
X = rng.normal(size=(5000, 8)).astype(np.float32)
y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.float32) + (X[:, 2] > 1)
model = bonsai.train([
    ("dispatch.objective_name", "softmax"),
    ("objective.n_classes", "3"),
    ("booster.n_iters", "50"),
], X, y)
pred = np.asarray(model.predict(X))
print("train accuracy:", (pred == y).mean())
EOF
```

Watch the model size: 50 iterations × 3 classes = **150 trees**. Multiclass training cost scales linearly in $K$: this is true of every GBT library and routinely surprises people coming from neural nets, where extra classes are one wider layer.

## Gotchas & war stories

- **`n_classes` is config, not inference.** The booster trusts `objective.n_classes` and buckets labels by `class_of`; labels outside `[0, K)` silently fold. Validate your label range yourself.
- **The Hessian floor is not decoration.** Remove it and a well-separated dataset produces leaves dividing by ~0 curvature late in training: enormous leaf values, then NaN scores two rounds later. Every library has this floor; nobody advertises it.
- **Per-class trees mean per-class importance.** `feature_importance` aggregates across all $K$ trees per round; a feature that only matters for one rare class dilutes accordingly. Slice `trees_` round-major if you need per-class attribution.
