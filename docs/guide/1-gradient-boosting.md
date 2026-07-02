# 1 — Gradient boosting

## The idea

You want a function $F(x)$ that predicts a number. One decision tree is a
coarse, blocky approximation. Boosting builds $F$ as a *sum* of many small
trees, where each new tree is fit not to the target `y` but to **how wrong
the current sum is**. After 200 rounds of "add a small correction to the
current mistakes", the sum is accurate even though every individual tree
is weak.

The word *gradient* is literal: "how wrong the current sum is" is the
gradient of the loss with respect to the current prediction, evaluated at
each training row. A tree fit to the negative gradient is one step of
gradient descent — performed in function space instead of parameter space.

## The math

Model after $m$ trees, with learning rate (shrinkage) $\eta$:

$$F_m(x) = \text{base} + \eta\, T_1(x) + \eta\, T_2(x) + \cdots + \eta\, T_m(x)$$

For a loss $L(y, F)$, each row $i$ contributes at round $m$:

$$g_i = \frac{\partial L}{\partial F} \quad\text{(gradient)} \qquad h_i = \frac{\partial^2 L}{\partial F^2} \quad\text{(hessian)}$$

both evaluated at the current prediction $F_{m-1}(x_i)$. For squared error
$L = \tfrac{1}{2}(F-y)^2$ these are simply $g = F - y$ (the residual) and
$h = 1$.

Why carry the hessian? Second-order (Newton) boosting, the xgboost
formulation: approximate the loss of adding value $w$ to every row in some
leaf with a second-order Taylor expansion, add an L2 penalty
$\tfrac{1}{2}\lambda w^2$, and minimize:

$$\sum_i \left(g_i w + \tfrac{1}{2} h_i w^2\right) + \tfrac{1}{2}\lambda w^2
\;\;\Longrightarrow\;\; w^{\ast} = -\frac{G}{H + \lambda}$$

where $G = \sum_i g_i$ and $H = \sum_i h_i$ over the rows in the leaf.
That's the whole formula for a leaf's value. With an L1 penalty $\alpha$
it becomes the soft-thresholded $w^{\ast} = -T(G, \alpha)/(H + \lambda)$
where $T$ shrinks $G$ toward zero by $\alpha$
(see [chapter 6](6-regularization-and-constraints.md)).

Shrinkage $\eta$ then scales each tree's contribution. Small $\eta$ means each
tree corrects only a fraction of the residual — slower, but successive
trees get to vote on overlapping mistakes, which regularizes.

## In bonsai

One boosting round is [`Booster::update_one_iter`](../../include/bonsai/booster.hpp):

1. `objective_.compute(scores_, labels, grad_, hess_)` — per-row g/h from
   the current raw scores. The objectives live in
   [`src/objective.cpp`](../../src/objective.cpp); MSE is the two-line loop
   `grad[i] = preds[i] - targets[i]; hess[i] = 1`.
2. `sampler_.sample(...)` — pick this round's rows ([chapter 5](5-sampling.md)).
3. `grower_.grow(train, grad_, hess_, rows)` — build one tree on those
   gradients ([chapters 2–4](2-binning-and-histograms.md)).
4. `scores_[i] += learning_rate * leaf_values[i]` — advance every row's
   running prediction. `GrowResult.values` carries each row's leaf value —
   including rows the sampler skipped, which are routed through the
   finished tree (`route_unsampled` in [`src/grower.cpp`](../../src/grower.cpp);
   the bug this prevents is chapter 5's war story).

The leaf-value formula is `bounded_leaf_weight` in
[`include/bonsai/split.hpp`](../../include/bonsai/split.hpp) — literally
$-\texttt{l1\_thresholded}(G, \alpha) / (H + \lambda)$, clamped to
monotone bounds.

The starting point `base` is `Objective::init_score` — the mean for MSE,
the median for MAE, the $\alpha$-quantile for quantile loss, log-odds for
logloss. Raw scores stay in link space throughout training; the sigmoid
(for logloss) is applied only at the outermost predict
(`apply_link_inverse_by_name`).

### The objectives gallery

| name | grad | hess | init | note |
|---|---|---|---|---|
| `mse` | $F - y$ | $1$ | mean | the reference path |
| `logloss` | $p - y$ | $p(1-p)$ | log-odds | raw-score space; sigmoid at predict |
| `mae` | $\operatorname{sign}(F - y)$ | $1$ | median | constant hessian — see below |
| `huber` | $\operatorname{clamp}(F - y, \pm\delta)$ | $1$ | median | `[objective] huber_delta` |
| `quantile` | $1-\alpha$ if $F > y$, else $-\alpha$ | $1$ | $\alpha$-quantile | `[objective] quantile_alpha` |

Constant-hessian caveat: for MAE/quantile,
$w^{\ast} = -G/(\text{count} + \lambda)$ is the *mean gradient*, and gradients
are all $\pm 1$-ish — so leaf steps are tiny and convergence is slow at
small $\eta$. The reference libraries "renew" such
leaves with the residual median after growing; bonsai doesn't yet
([feature_gap.md](../feature_gap.md) row 10), which costs ~10% MAE at
matched settings. Documented, measured, honest.

## Try it

```bash
# Watch the training loss fall (log a tick every ~20 iters):
bonsai fit -c configs/california_housing.toml --set booster.log_intervals=10

# Same model, robust loss:
bonsai fit -c configs/california_housing.toml --set dispatch.objective_name=huber
```

```python
import bonsai
m = bonsai.BonsaiRegressor(n_iters=200, learning_rate=0.05).fit(X, y)
```

Halve the learning rate and double `n_iters`: RMSE improves slightly —
same total step budget, more votes per mistake.

## Gotchas & war stories

- **Gradients must be computed against the *real* model.** Every row's
  score has to advance every round, sampled or not. bonsai shipped for
  weeks with out-of-bag rows silently frozen — see chapter 5.
- **`hess` is a row-count under constant-hessian objectives**, so
  `min_child_hess` quietly changes meaning between `mse` and `mae`.
- **Raw scores vs predictions**: everything internal is raw-score space.
  If you eval logloss models by hand, apply the sigmoid first — the CLI's
  `predict` already does.
