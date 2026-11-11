# 1. Gradient boosting

## The idea

You want a function $F(x)$ that predicts a number. One decision tree is a coarse, blocky approximation. Boosting builds $F$ as a *sum* of many shallow trees (bonsai's default `max_depth` is 6), where each new tree is fit not to the target `y` but to **how wrong the current sum is**. After 200 rounds of "add a small correction to the current mistakes", the sum is accurate on the training rows even though every individual tree is weak (alone it predicts poorly). Too many rounds overfit, so score held-out rows and stop when that score stops improving ([chapter 7](7-early-stopping-and-dart.md)).

The word *gradient* is literal: "how wrong the current sum is" is the gradient of the loss (the error measure being minimized) with respect to the current prediction, evaluated at each training row. Fitting a tree to the negative gradient is one step of gradient descent. Ordinary descent adjusts a model's parameters. Boosting adjusts the prediction itself, one tree at a time, which is why this is called descent in function space.

## The math

Model after $m$ trees, with learning rate (shrinkage) $\eta$:

```math
F_m(x) = \text{base} + \eta\, T_1(x) + \eta\, T_2(x) + \cdots + \eta\, T_m(x)
```

For a loss $L(y, F)$, each row $i$ contributes at round $m$:

```math
g_i = \frac{\partial L}{\partial F} \quad\text{(gradient)} \qquad h_i = \frac{\partial^2 L}{\partial F^2} \quad\text{(hessian)}
```

both evaluated at the current prediction $F_{m-1}(x_i)$. For squared error $L = \tfrac{1}{2}(F-y)^2$ these are simply $g = F - y$ (the residual) and $h = 1$.

The hessian lets you take a second-order (Newton) step, the formulation XGBoost uses. Approximate the loss of adding a value $w$ to every row in one leaf with a second-order Taylor expansion (a quadratic that matches the loss near the current prediction). Then add an L2 penalty (a cost on the squared leaf value) $\tfrac{1}{2}\lambda w^2$ and minimize the sum:

```math
\sum_i \left(g_i w + \tfrac{1}{2} h_i w^2\right) + \tfrac{1}{2}\lambda w^2
\;\;\Longrightarrow\;\; w^{\ast} = -\frac{G}{H + \lambda}
```

where $G = \sum_i g_i$ and $H = \sum_i h_i$ over the rows in the leaf. That's the whole formula for a leaf's value. With an L1 penalty $\alpha$ it becomes the soft-thresholded $w^{\ast} = -T(G, \alpha)/(H + \lambda)$ where $T$ shrinks $G$ toward zero by $\alpha$ (see [chapter 6](6-regularization-and-constraints.md)).

Shrinkage $\eta$ then scales each tree's contribution. A smaller $\eta$ makes each tree correct a smaller fraction of the residual, so the model needs more rounds. Each extra tree fixes part of the same errors, which reduces overfitting (see [chapter 6](6-regularization-and-constraints.md)). The default `learning_rate` is 0.05, used with 200 rounds in the example below.

## In bonsai

A plain boosting round (no DART, no sample weights) is [`Booster::update_one_iter`](../../include/bonsai/booster.hpp):

1. `objective_.compute(scores_, labels, grad_, hess_)`: per-row g/h from the current raw scores. The objectives live in [`src/objective.cpp`](../../src/objective.cpp); MSE is the two-line loop `grad[i] = preds[i] - targets[i]; hess[i] = 1`.
2. `sampler_.sample(...)`: pick this round's rows ([chapter 5](5-sampling.md)).
3. `grower_.grow(train, grad_, hess_, rows)`: build one tree on those gradients ([chapters 2–4](2-binning-and-histograms.md)).
4. `scores_[i] += learning_rate * leaf_values[i]`: advance every row's running prediction. `GrowResult.values` carries each row's leaf value, including rows the sampler skipped, which are routed through the finished tree (`route_unsampled` in [`src/grower_impl.hpp`](../../src/grower_impl.hpp); the bug this prevents is [chapter 5's out-of-bag stale-score bug](5-sampling.md#gotchas-war-stories)).

Three legs are left out here: DART drops trees before step 1 ([chapter 7](7-early-stopping-and-dart.md)), sample weights multiply grad and hess after step 1 and weigh the starting point and the renewed leaf values, and constant-hessian objectives renew leaf values between steps 3 and 4 (next section).

The leaf-value formula is `bounded_leaf_weight` in [`include/bonsai/split.hpp`](../../include/bonsai/split.hpp): literally $-T(G, \alpha) / (H + \lambda)$ with $T$ = `l1_thresholded`, clamped to monotone bounds.

The starting point `base` is `Objective::init_score`: the mean for MSE, the median for MAE, the $\alpha$-quantile for quantile loss, log-odds for logloss. Under sample weights each is the weighted statistic (a weighted mean, a weighted quantile, the log-odds of the weighted positive share), so a row with zero weight adds nothing to it, and a fit in which no row has a positive weight is refused. During training, scores stay as raw values on the model's internal scale (log-odds for logloss). The sigmoid, which turns log-odds into probabilities, is applied when you call `predict` (`apply_link_inverse_by_name`).

### The objectives gallery

| name | grad | hess | init | note |
|---|---|---|---|---|
| `mse` | $F - y$ | $1$ | mean | the reference path |
| `logloss` | $p - y$ | $p(1-p)$ | log-odds | raw-score space; sigmoid at predict |
| `mae` | $\mathrm{sign}(F - y)$ | $1$ | median | constant hessian, see below |
| `huber` | $\mathrm{clamp}(F - y, \pm\delta)$ | $1$ | median | `huber_delta` (default 1.0, set under `[objective]`): residuals smaller than it get squared-error treatment, larger ones get absolute-error treatment |
| `quantile` | $1-\alpha$ if $F > y$, else $-\alpha$ | $1$ | $\alpha$-quantile | `quantile_alpha` (default 0.5, set under `[objective]`): the quantile the model predicts, with 0.5 the median |
| `poisson` | $e^F - y$ | $e^F$ | $\log(\bar{y})$ | log link; labels must be ≥ 0; raw scores clamp so $e^F$ can't overflow |

The table covers the one-output objectives. `softmax` (multiclass) boosts K scores per row, one tree per class per round, so it does not fit these columns; see [chapter 12](12-multiclass.md).

### Constant-hessian objectives and leaf renewal

For MAE and quantile the hessian is a constant, so the Newton step $w^{\ast} = -G/(\text{count} + \lambda)$ becomes the mean of the leaf's gradients. Each gradient depends only on the sign of $F - y$ (+1 or -1 for MAE, $1-\alpha$ or $-\alpha$ for quantile). The step therefore gives the direction but not the distance: its length is set by $\eta$ alone, and a lower $\eta$ needs proportionally more rounds. The fix is **leaf renewal**: after the tree's structure is fixed, re-solve each leaf exactly,

```math
w^{\ast}_{\text{leaf}} = \arg\min_{w} \sum_{i \in \text{leaf}} \ell(y_i,\; F_i + w)
```

which for MAE is the **median** of the leaf's residuals $y_i - F_i$ (the minimizer of a sum of absolute values), for `quantile` the residuals' $\alpha$-quantile (the pinball loss's minimizer), and for Huber the residual median plus the mean of the deviations from it, each clamped to plus or minus delta. The Huber value is a one-step approximation, not the exact minimizer. The gradient step chose *which rows* belong together; renewal asks the loss itself *what value* serves them best.

Under sample weights the minimized sum is weighted, so the median and the quantile become weighted ones. bonsai's rule: drop the residuals whose weight is not positive, order the $n$ that remain, scale their weights to average one, and take the first residual whose running weight, less one half, exceeds $\alpha (n - 1)$. With equal weights that is the element the unweighted rule takes, index $\mathrm{round}(\alpha (n - 1))$, so a constant weight changes nothing. Huber's clamped mean becomes a weighted mean around the weighted median. The same rule sets the weighted starting point for MAE, Huber and quantile loss.

In bonsai: `renew_leaf` on the objective ([`include/bonsai/objective.hpp`](../../include/bonsai/objective.hpp)) and the booster's `renew_leaves` pass, gated at compile time on the objective providing the method. The compiler removes the renewal call for MSE and logloss, so those objectives pay no runtime cost for it. On Year Prediction MSD (200 iterations, leafwise), renewal lowered bonsai's MAE from 6.81 to 6.094, which matches LightGBM (6.095) and is slightly below XGBoost (6.155) and CatBoost (6.159) ([the feature-gap record](https://github.com/daniel-m-campos/bonsai/blob/main/benchmarks/feature-gap-2026-07.md), section 10).

## Try it

Fit with the default squared error, then swap the loss with one kwarg:

```python {.run}
import numpy as np
import bonsai

rng = np.random.default_rng(0)
X = rng.normal(size=(4000, 8)).astype(np.float32)
y = (X[:, 0] * 2.0 + X[:, 1] + rng.normal(0, 0.1, 4000)).astype(np.float32)

mse = bonsai.BonsaiRegressor(n_iters=200, learning_rate=0.05).fit(X, y)
slow = bonsai.BonsaiRegressor(n_iters=400, learning_rate=0.025).fit(X, y)
huber = bonsai.BonsaiRegressor(objective="huber", n_iters=200).fit(X, y)
print("mse   R2:", round(mse.score(X, y), 4))
print("slow  R2:", round(slow.score(X, y), 4))
print("huber R2:", round(huber.score(X, y), 4))
```

Halve the learning rate and double `n_iters`: the training R2 moves from 0.9981 to 0.9982. Each tree corrects half as much, and twice as many trees are added, so the total correction is about the same.

## Gotchas & war stories

- **Gradients must be computed against the *real* model.** Every row's score has to advance every round, sampled or not. bonsai shipped for weeks with out-of-bag rows silently frozen. See [the out-of-bag stale-score bug](5-sampling.md#gotchas-war-stories) in chapter 5.
- **`min_child_hess` is a minimum row count under `mse`, `mae`, `huber` and `quantile`**, because each row's hessian is 1 (with `weight_column` set, it is a minimum total row weight). Under `logloss` and `poisson` it is a minimum sum of curvature, so the same value maps to a different row count (for `logloss` each row adds at most 0.25, so a leaf needs at least four times as many rows). Re-tune it when you switch to or from those objectives.
- **Raw scores vs predictions**: everything internal is raw-score space. If you eval logloss models by hand, apply the sigmoid first: the CLI's `predict` already does.
