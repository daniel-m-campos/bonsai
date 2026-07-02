# 5 — Sampling

## The idea

Histogram building costs rows × features. If each tree trains on fewer
rows, each tree is cheaper — and, usefully, slightly different from its
siblings, which regularizes like bagging. Two disciplines ship:

- **Bernoulli** (uniform): keep each row with probability `p` per
  iteration. Simple, unbiased, the classic `subsample`.
- **GOSS** — Gradient-based One-Side Sampling (lightgbm's invention): rows
  with large |gradient| are the ones the model is *wrong about*, so keep
  all of those and only a thin uniform sample of the well-predicted rest.
  Most of the information for choosing splits lives in the big gradients.

## The math

GOSS keeps the top $a$ fraction of rows by $|g|$ and uniformly samples $b$
of the total from the remainder. The sampled remainder stands in for
$(1-a)$ of the data using only $b$ of it, so its rows are **amplified**:

$$g_i \leftarrow g_i \cdot \frac{1-a}{b} \qquad
h_i \leftarrow h_i \cdot \frac{1-a}{b} \qquad \text{(sampled rest only)}$$

With that reweighting, every histogram cell's expected sums equal the
full-data sums — split gains are unbiased estimates — while histogram work
drops to $(a + b)$ of the rows (default $a = 0.2$, $b = 0.1$: 30%).

## In bonsai

The `Sampler` concept ([`include/bonsai/sampler.hpp`](../../include/bonsai/sampler.hpp))
fills an index buffer and returns how many it kept. GOSS is ~40 lines in
[`src/sampler.cpp`](../../src/sampler.cpp): `nth_element` to rank by $|g|$,
`std::sample` for the rest, amplify in place, emit indices in ascending
row order (locality downstream). Note the concept takes **mutable**
grad/hess spans precisely so GOSS can scale in place — the booster
recomputes both from the objective every iteration, so the scaling never
outlives its tree.

Selection: `dispatch.sampler_name = all_rows | bernoulli | goss`, with
`sampler.subsample` / `sampler.top_rate` / `sampler.other_rate`.

## Try it

```bash
uv run scripts/compare.py --config configs/year_prediction_msd.toml \
    --growers leafwise --samplers all_rows,goss
```

Measured on YearPredictionMSD (feature_gap.md §2): GOSS *improved*
leafwise RMSE (9.0871 → 9.0757) while cutting fit time — the same
direction and magnitude as lightgbm's own GOSS delta.

## Gotchas & war stories

**The out-of-bag stale-score bug** — the best bug this project produced,
because a benchmark caught what unit tests hadn't (decision 34).

The grower returns each training row's leaf value so the booster can
advance `scores_` without re-predicting. Originally it only wrote values
for rows *in the tree* — the sampled ones. Out-of-bag rows kept a zero,
their running scores silently froze, and the next round's gradients for
them were computed against a model missing whole trees.

With Bernoulli the damage hid inside plausible noise: every row was
usually sampled again within a round or two, so RMSE was quietly ~2% worse
(9.19 vs 8.99) and nobody suspected. With GOSS the same bug **diverged** —
RMSE 24.7, worse than predicting the mean — because GOSS re-selects by
|gradient| every round: frozen rows keep large stale gradients, get
re-selected forever, and the amplification feeds the loop.

The fix is `route_unsampled` in [`src/grower.cpp`](../../src/grower.cpp):
after growing, walk every skipped row down the finished tree *in bin
space* (the split bin recorded at grow time; exact w.r.t. the float
threshold since `threshold = cuts[bin]`) and stamp its leaf value. The
regression test pins the contract: *every row's train value equals the
tree's prediction for that row, sampled or not.*

Two lessons: a sampler that reacts to model state (GOSS) turns quiet
staleness into a feedback loop; and cross-library benchmarks are
correctness tests — lightgbm's GOSS landing at 9.07 while ours sat at 24.7
was the whole diagnosis.
