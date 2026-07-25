# 15. Explaining predictions: TreeSHAP

## The idea

Chapter 8 answers a global question: which features mattered to the model overall. This chapter answers the local one: which features moved *this* prediction, for *this* row, and by how much. That question arrives the moment a model faces a person: why was this loan declined, why is this reading flagged, why did the forecast jump.

A useful local explanation needs a fairness contract, or every method invents its own answer. The Shapley value from cooperative game theory supplies one. Treat the features as players, the prediction as the payout, and give each feature its average marginal contribution over every order in which features could reveal themselves. Three properties follow that heuristics do not give you. Contributions sum exactly to the prediction (efficiency). Two features that contribute identically get identical credit (symmetry). A feature the model never uses gets exactly zero (dummy).

The catch is cost: the definition sums over all $2^p$ feature subsets. TreeSHAP (Lundberg et al., 2018, [the paper](https://arxiv.org/abs/1802.03888)) computes the exact same numbers for tree ensembles in polynomial time. bonsai ships it as `pred_contribs`, and chapter 14's survey already showed the payoff of having it: attribution averaged over held-out rows was the best feature ranking on wide data, measurably ahead of gain.

## The math

For one tree, the value of a feature subset $S$ is the tree's expectation when only $S$ is known:

```math
v(S) = \mathbb{E}\left[\,\text{tree}(x) \mid x_S\,\right]
```

Concretely: walk from the root; at a split on a feature in $S$, follow $x$; at a split on any other feature, average both children weighted by their **cover** (how many training rows passed through each). The Shapley value of feature $f$ is then the weighted average of its marginal contributions:

```math
\phi_f = \sum_{S \subseteq F \setminus \{f\}} \frac{|S|!\,(p - |S| - 1)!}{p!}\,\left[\,v(S \cup \{f\}) - v(S)\,\right]
```

Two structural facts rescue this from its $2^p$ subsets. First, additivity: the ensemble's $\phi$ is the sum of per-tree $\phi$s, so one tree at a time suffices. Second, within a tree, $v(S)$ only depends on which features appear *on a root-to-leaf path*, and a path has at most $D$ features. TreeSHAP walks each path once, carrying for every path prefix the fraction of subsets that reach it with the split feature blocked (`zero_fraction`), the fraction with it active (`one_fraction`), and the accumulated permutation weights. Bookkeeping per path costs $O(D^2)$, so a tree costs $O(L\,D^2)$ per row, thousands of times cheaper than enumeration at even ten features, with identical output.

## In bonsai

The entire feature is one small pair: [`include/bonsai/shap.hpp`](../../include/bonsai/shap.hpp) (29 lines, the contract) and [`src/shap.cpp`](../../src/shap.cpp) (251 lines, Algorithm 2 of the paper). The pieces map one-to-one onto the math:

- `tree_expected_value(tree, X, row, in_subset)` is $v(S)$ verbatim: follow $x$ where conditioned, average children by cover elsewhere. It exists so the tests can check the fast algorithm against the definition (below), and it computes the bias term (the empty subset, the tree's expected value) that lands in the last output column.
- `PathElement` holds one prefix entry: `zero_fraction`, `one_fraction`, permutation weight. `extend` adds a split condition, `unwind` removes one, and a leaf pays out `weight * (one_fraction - zero_fraction) * leaf_value` to every feature on its path.
- A feature that appears twice on a path is folded into one entry (a feature contributes once), and each branch works on its own copy of the path, exactly as the paper's pseudocode duplicates it.
- One bonsai-specific guard: branches with zero cover are skipped rather than assumed away. Empty children exist here by construction (device partitioning keeps them as leaves, and oblivious trees expand with dead slots), and unwinding a zero-cover entry would divide zero by zero. The instance's own branch still recurses whenever the row routes down it, which is what keeps the efficiency property exact even when a row routes into a dead slot.
- Covers are the load-bearing data: a model without per-node covers (formats before v6) cannot answer $v(S)$ and `tree_shap` throws rather than guess. Same lesson as gain in chapter 8: stamp it at training time or it is gone.

The surfaces: `pred_contribs(X)` on any estimator returns `(n, p + 1)` (features plus bias, rows summing to raw predictions); multiclass returns per-class slices `(n, K, p + 1)` (chapter 12); oblivious models answer through their dense expansion. Missing values are attributed like any other routing: `NaN` follows the learned default branch from chapter 3.

The test story is the part worth copying into your own projects. [`tests/unit/test_shap.cpp`](../../tests/unit/test_shap.cpp) implements `brute_force_shapley`: the $2^p$ subset enumeration with the factorial weights, straight from the definition, and requires the fast path to match it to 1e-9 on a hand-built tree. The additivity identity is asserted for whole boosters, oblivious dense expansion is checked against its own routing, and multiclass slices must vote like `predict`. The algorithm is not trusted; it is reconciled against the formula it claims to compute.

## Try it

```{.python .run}
import numpy as np
import bonsai

rng = np.random.default_rng(0)
n = 4000
X = rng.normal(size=(n, 5)).astype(np.float32)
y = (3.0 * np.sign(X[:, 0]) + X[:, 1] * X[:, 2]
     + 0.1 * rng.normal(size=n)).astype(np.float32)

m = bonsai.BonsaiRegressor(n_iters=150).fit(X, y)
phi = np.asarray(m.pred_contribs(X[:500]))
pred = np.asarray(m.predict(X[:500]))
print("phi shape:", phi.shape)
print("max |sum(phi) - prediction|:", f"{np.abs(phi.sum(axis=1) - pred).max():.2e}")
r = 0
print("row 0: x =", X[r].round(2))
print("row 0: phi =", phi[r, :-1].round(3), " bias =", round(float(phi[r, -1]), 3),
      " prediction =", round(float(pred[r]), 3))
global_shap = np.abs(phi[:, :-1]).mean(axis=0)
print("mean-abs-SHAP ranking:", np.argsort(global_shap)[::-1])
print("gain ranking         :", np.argsort(np.asarray(m.importance('gain')))[::-1])
print("corr(phi_1, x1*sign(x2)):",
      round(float(np.corrcoef(phi[:, 1], X[:500, 1] * np.sign(X[:500, 2]))[0, 1]), 3))
```

Read the output top to bottom. The identity holds to float precision (about 3e-06 here): every row's contributions plus bias reproduce its prediction, the efficiency axiom as a runtime check. Row 0's explanation is legible: $x_0$ sits just above zero, the sign term pays +3, so $\phi_0 \approx 3.07$ and everything else is noise. The global rankings from mean-absolute-SHAP and gain agree on the dominant feature and differ only in the tail. The last line is the one no global method can produce: feature 1's per-row credit tracks $x_1 \cdot \text{sign}(x_2)$ at correlation 0.8, because in the product term $x_1 x_2$, whether $x_1$ *helped or hurt* depends on its partner's sign, and local attribution sees that per row.

## What to distrust

- **SHAP explains the model, not the world.** $\phi_f$ is the model's bookkeeping, not a causal effect. A feature standing in for a confounder gets the credit the confounder earned.
- **The background matters.** bonsai attributes the cover-weighted conditional expectation, the paper's path-dependent variant: absent features average over where *training rows actually went*. The `shap` package's default interventional variant answers against an explicit background dataset instead. Under correlated features the two disagree, and neither is wrong; they answer different counterfactuals. Chapter 8's warning applies here too: correlated features split credit between them in either variant.
- **Global SHAP is a sum of locals, priced accordingly.** Cost is $O(\text{rows} \times \text{trees} \times L D^2)$. For a global ranking, a few hundred rows is usually plenty; chapter 14's `shap_val` used the validation slice, which is also the better-measured choice on wide data (it discounts features the fit merely memorized).
- **The additivity identity is a free integrity check.** If `sum(phi)` drifts from the raw prediction by more than float noise, something upstream is wrong (mismatched model file, wrong objective transform). Assert it in pipelines; it costs one line.
