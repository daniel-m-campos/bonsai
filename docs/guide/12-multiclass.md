# 12. Multiclass: softmax boosting

## The idea

Binary classification needed one score per row ([chapter 1](1-gradient-boosting.md)); $K$ classes need $K$. Gradient boosting keeps $K$ running scores per row. Each round it grows **one tree per class**. Each tree fits the gradient, with respect to that class's score, of the softmax cross-entropy (the loss that turns the $K$ scores into probabilities with softmax, then penalizes the probability given to the true class). Nothing about trees, histograms, splits, or sampling changes: the same grower that fits house prices fits "the gradient of class 3". Only the booster (the class that runs the boosting rounds) knows about classes: it owns the matrix of scores with one column per class, and the tree grower sees one column of gradients and Hessians at a time, exactly as in a regression fit.

To predict, sum each class's trees to get $K$ scores per row, then pick the class with the highest score (the argmax). Apply softmax to those scores if you want probabilities.

## The math

Row $i$ has scores $s_1, \dots, s_K$ and a one-hot target $y$. Softmax turns scores into probabilities and cross-entropy scores them:

```math
p_k = \frac{e^{s_k}}{\sum_j e^{s_j}}, \qquad \mathcal{L} = -\log p_{y}
```

The derivatives that feed the trees are the classic softmax pair:

```math
\frac{\partial \mathcal{L}}{\partial s_k} = p_k - \mathbf{1}[y = k], \qquad \frac{\partial^2 \mathcal{L}}{\partial s_k^2} = p_k (1 - p_k)
```

The gradient $p_k - \mathbf{1}[y = k]$ is the probability the model gave class $k$ minus the target. The Hessian (the second derivative of the loss, which measures curvature) is largest when $p_k \approx \tfrac12$, so rows the model is sure about add little curvature. bonsai makes two choices. First, it uses the true diagonal Hessian $p_k(1 - p_k)$. XGBoost's factor-2 variant halves every Newton step (the leaf value update) and needed 2x the iterations to match LightGBM at the same learning rate ([decision 56](../decisions.md)). Second, it floors the Hessian at $10^{-6}$. For a row the model is certain about, $p_k(1-p_k) \to 0$, and a leaf value is a gradient sum divided by a Hessian sum, so without the floor the leaf value would grow without bound.

Like XGBoost and LightGBM, bonsai uses only the diagonal of the softmax Hessian. The full Hessian couples classes ($\partial^2 \mathcal{L} / \partial s_k \partial s_j = -p_k p_j$), but dropping those terms lets each class's tree fit independently. The cost is extra rounds (not measured here). The dropped coupling re-enters through the updated probabilities in the next round.

## In bonsai

The round lives in [`include/bonsai/multiclass_booster.hpp`](../../include/bonsai/multiclass_booster.hpp). The $K$-wide trees, score matrix and raw-score queries it inherits live in [`include/bonsai/multi_output_ensemble.hpp`](../../include/bonsai/multi_output_ensemble.hpp), which multi-target fits share, and the flat `trees_` store sits in the `Ensemble` base in [`include/bonsai/booster.hpp`](../../include/bonsai/booster.hpp):

- **Why a separate booster at all**: an `Objective` computes gradients from `(preds, targets)` as one column. Softmax needs $K$ columns, so the registry maps `{softmax, G, Sa}` to its own booster, `MulticlassBooster` (`BoosterFor<{softmax, G, Sa}>`). The one-column methods of [`SoftmaxObjective`](../../src/objective.cpp) throw on purpose: no training code path calls them. Choosing the booster through the registry, not through a flag inside one booster, keeps every other objective one-dimensional ([decision 26](../decisions.md)).
- **The round loop** (`update_one_iter`): scores are a flat `n × K` row-major matrix; for each class $k$, a softmax pass fills `grad_/hess_` for column $k$, the *same* row sampler draws (GOSS and bagging work unchanged), and the *same* grower grows a tree. `trees_` is flat and round-major: class $k$ of round $r$ sits at `r * K + k`, which is all `predict_at(n_rounds)` needs to truncate cleanly.
- **Prediction**: `raw_scores` (inherited from `MultiOutputEnsemble` in [`include/bonsai/multi_output_ensemble.hpp`](../../include/bonsai/multi_output_ensemble.hpp)) accumulates each tree into its class column; `predict` emits argmax class ids; `eval` is the multiclass logloss on the raw scores.
- **What works unchanged, and what changed**: bagging, GOSS, every grower (including the CUDA ones), feature importance, warm start, and the model format work as for any objective, because each round is $K$ ordinary trees. Two features needed multiclass code. Early stopping scores a matrix of width `score_width()` = $K$. `pred_contribs` returns shape `(n, K, n_features + 1)`, and each class's slice sums to that class's raw score.

## Try it

```python {.run}
import numpy as np
import bonsai

rng = np.random.default_rng(0)
X = rng.normal(size=(5000, 8)).astype(np.float32)
y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.float32) + (X[:, 2] > 1)
model = bonsai.train(
    {
        "dispatch.objective_name": "softmax",
        "objective.n_classes": 3,
        "booster.n_iters": 50,
    },
    X,
    y,
)
pred = np.asarray(model.predict(X))
print("train accuracy:", (pred == y).mean())
```

Watch the model size: 50 iterations × 3 classes = **150 trees**. Training time and model size grow roughly linearly with $K$, because each round grows $K$ trees: 3 classes at the same `n_iters` give 3 times the trees of a binary fit. A neural network differs here: an extra class adds one output unit to the last layer.

## Gotchas & war stories

- **`n_classes` is config, not inference.** `objective.n_classes` (int, default 3, at least 2) sets $K$, the number of classes and the number of trees per round. The booster does not count the classes in your labels. Training refuses any train or validation label that is not an integer in `[0, K)` with a `ValueError`. Set `n_classes` to the number of classes your labels use, and encode labels to `0..K-1` first.
- **The Hessian floor of $10^{-6}$ prevents NaN scores.** On a well-separated dataset, $p_k(1-p_k)$ approaches 0 late in training. Without the floor, the leaf value (a gradient sum divided by a Hessian sum) grows without bound and then produces NaN scores. Do not remove it.
- **Per-class trees mean per-class importance.** `feature_importance` adds up the gain (or split count) of all $K$ trees in every round into one score per feature, so it cannot say which class a feature serves: a feature that matters only for one class is mixed in with the features of the other $K-1$ classes. For per-class attribution, use `pred_contribs`, which returns a `(n, K, n_features + 1)` array with one slice per class. To inspect the trees, read `model.dump()`, which labels each tree with its round and class.
