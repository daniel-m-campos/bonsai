# 5. Sampling

## The idea

Histogram building costs rows × features. If each tree trains on fewer rows, each tree is cheaper. Each tree also sees a different random sample of rows, so it differs from the other trees, and that variation reduces overfitting, as in bagging (training each model on a different random sample of the rows). Two disciplines ship:

- **Bernoulli** (uniform): keep each row with the same probability `sampler.subsample` per iteration (default 1.0, meaning all rows). Kept rows are not reweighted. This is the same idea as `subsample` in XGBoost and `bagging_fraction` in LightGBM.
- **GOSS**: Gradient-based One-Side Sampling (LightGBM's invention). Rows with large |gradient| have the largest prediction error, so they move split gains the most. GOSS keeps all of the top `top_rate` fraction of rows by |gradient| (default 20%) and uniformly samples `other_rate` of the total rows (default 10%) from the remainder.

## The math

GOSS keeps the top $a$ fraction of rows by $|g|$ and uniformly samples $b$ of the total from the remainder. The sampled remainder stands in for $(1-a)$ of the data using only $b$ of it, so its rows are **amplified**:

```math
g_i \leftarrow g_i \cdot \frac{1-a}{b} \qquad
h_i \leftarrow h_i \cdot \frac{1-a}{b} \qquad \text{(sampled rest only)}
```

After the reweighting, the expected gradient and hessian sums in every histogram cell equal the full-data sums. Histogram work drops to $(a + b)$ of the rows, which is 30% at the defaults $a = 0.2$ and $b = 0.1$.

## In bonsai

A sampler is a C++ type that fills an index buffer with the rows to train on and returns how many it kept ([`include/bonsai/sampler.hpp`](../../include/bonsai/sampler.hpp)). GOSS is about 140 lines in [`src/sampler.cpp`](../../src/sampler.cpp), constructor validation included: `nth_element` to rank by $|g|$, `std::sample` for the rest, amplify in place, emit row ids in the order of the Dataset's row view (ascending for a plain fit, so later memory reads are sequential). On a bootstrap view a row can appear in the candidates more than once. GOSS amplifies such a row once, and a row with copies among both the kept and the picked rows takes the mean factor. A sampler receives the gradients and hessians as mutable spans so GOSS can scale the sampled rows in place; the booster recomputes both from the objective every iteration, so the scaling never outlives its tree.

Selection: `dispatch.sampler_name = all_rows | bernoulli | goss`, with `sampler.subsample` / `sampler.top_rate` / `sampler.other_rate`.

## Try it

```python {.run}
import numpy as np
import bonsai

rng = np.random.default_rng(0)
X = rng.normal(size=(6000, 12)).astype(np.float32)
y = (X[:, 0] * 2.0 + X[:, 3] + rng.normal(0, 0.1, 6000)).astype(np.float32)

for sampler in ("all_rows", "goss"):
    m = bonsai.BonsaiRegressor(n_iters=100, sampler=sampler).fit(X, y)
    print(f"{sampler:9s}  R2={m.score(X, y):.4f}")
```

On YearPredictionMSD (200 iterations, `top_rate` 0.2, `other_rate` 0.1), GOSS lowered leafwise RMSE from 9.0871 to 9.0757 (-0.13%) and cut fit time by 13% ([the feature-gap record, section 2](https://github.com/daniel-m-campos/bonsai/blob/main/benchmarks/feature-gap-2026-07.md)). LightGBM's GOSS lowered its RMSE by 0.14% on the same data and cut its fit time by 44%.

**Limits**: GOSS shrinks histogram building, not the work around it. Every iteration it still ranks all rows by |gradient| with `nth_element` and scans all rows to build the index buffer, so the fit-time saving is well below the 70% cut in rows: 33% for depthwise and 13% for leafwise in the same record. GOSS reads and reweights host gradients, so it is not eligible for the resident device objective ([decision 77](../decisions.md), [the resident objective](../learn/engine/4-the-resident-objective.md)). Measure against `all_rows` on your own data before adopting it.

## Gotchas & war stories

**The out-of-bag stale-score bug**: rows the sampler skipped (out-of-bag rows) once kept a stale score. This made GOSS diverge (RMSE 24.7) and Bernoulli about 2% worse (9.19 vs 8.99). A cross-library benchmark found it, not a unit test (decision 34).

The grower returns each training row's leaf value so the booster can advance `scores_` without re-predicting. Originally it only wrote values for rows *in the tree*: the sampled ones. Out-of-bag rows kept a zero, their running scores silently froze, and the next round's gradients for them were computed against a model missing whole trees.

With Bernoulli, every row was usually sampled again within a round or two, so RMSE was about 2% worse (9.19 vs 8.99). With GOSS the same bug **diverged**: RMSE reached 24.7, worse than predicting the mean. GOSS re-selects rows by |gradient| every round. Frozen rows keep large stale gradients, so GOSS selects them again every round and the error feeds back into the loop.

The fix is `route_unsampled` in [`src/grower_impl.hpp`](../../src/grower_impl.hpp). After growing a tree, it sends every skipped row through the finished tree and records the leaf value as that row's training value. It compares each row's bin index (the bucket its feature value fell into) against the split bin recorded when the tree was grown. This picks the same leaf as the float threshold, because `threshold = cuts[bin]`. The regression test pins the contract: *every row's train value equals the tree's prediction for that row, sampled or not.*

Two lessons: a sampler that reacts to model state (GOSS) turns quiet staleness into a feedback loop; and cross-library benchmarks are correctness tests: LightGBM's GOSS landing at 9.07 while ours sat at 24.7 was the whole diagnosis.
