# 0 — A tree by hand

## The idea

Before any chapter's math or code, run the whole machine once on data small enough to hold in your head: **eight rows, one feature, one boosting round**. Every number below is computed with the exact formulas the library ships (`grad = pred − y`, `hess = 1` for MSE; gain and leaf values from [`split.hpp`](../../include/bonsai/split.hpp)); when you later meet the same quantities at 16 million rows, nothing will have changed but the row count.

The data — predicting `y` from a single feature `x`:

| row | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| x | 0.2 | 0.5 | 0.9 | 1.4 | 2.1 | 2.6 | 3.3 | 3.8 |
| y | 1.0 | 1.2 | 0.9 | 1.1 | 3.0 | 3.2 | 2.9 | 3.1 |

Clearly two regimes: `y ≈ 1` below `x ≈ 2`, `y ≈ 3` above.

## Step 1 — the base score and the gradients

Boosting starts from a constant. For MSE that's the mean: $F_0 = \bar{y} = 2.05$. The per-row gradient and hessian of $\tfrac12(F - y)^2$ at $F_0$:

```math
g_i = F_0 - y_i, \qquad h_i = 1
```

| row | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| $g_i$ | 1.05 | 0.85 | 1.15 | 0.95 | −0.95 | −1.15 | −0.85 | −1.05 |

The tree we're about to grow will fit *these* — not `y`. Rows the model over-predicts carry positive gradients; the leaf they land in will get a negative value that pulls them down.

## Step 2 — binning

bonsai's cuts are quantile *values of the data itself* ([`create_cuts`](../../src/bin_mapper.cpp)): sort the sampled column, take every stride-th value, so here $c = (0.5,\, 0.9,\, 1.4,\, 2.1,\, 2.6,\, 3.3,\, 3.8,\, +\infty)$, and bin $b$ holds values in $(c_{b-1}, c_b]$. Rows 0 and 1 (x = 0.2, 0.5) both land in **bin 0**; every other row gets its own bin. (At scale this is the whole point of [chapter 2](2-binning-and-histograms.md): thresholds are decided once, up front, and rows become bytes.)

## Step 3 — the histogram

One cell per bin, each accumulating $(\sum g, \sum h)$ of its rows: cell 0 holds rows 0–1 $(G{=}1.90, H{=}2)$; cells 1–6 hold one row each. Node totals: $G = 0$, $H = 8$.

## Step 4 — the gain scan

A split "bin ≤ b" (equivalently $x \le c_b$) scores by how much structure the two sides explain ([chapter 3](3-finding-splits.md), with $\lambda = 1$):

```math
\text{gain}(b) = \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{G^2}{H + \lambda}
```

One prefix scan over the cells, keeping running $G_L, H_L$:

| after bin $b$ | 0 | 1 | **2** | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| $G_L$ | 1.90 | 3.05 | 4.00 | 3.05 | 1.90 | 1.05 |
| $H_L$ | 2 | 3 | 4 | 5 | 6 | 7 |
| gain | 1.72 | 3.88 | **6.40** | 3.88 | 1.72 | 0.69 |

(Sample: at $b=2$, $\text{gain} = \tfrac{4.0^2}{5} + \tfrac{(-4.0)^2}{5} - 0 = 6.40$.)

The winner is $b = 2$ — threshold $c_2 = 1.4$, i.e. "$x \le 1.4$", exactly the boundary between the two regimes. No sorting happened at split time; finding it cost one pass over the cells.

## Step 5 — leaf values

Each child's Newton step:

```math
w^{\ast} = -\frac{G}{H + \lambda}: \qquad w_L = -\frac{4.0}{4 + 1} = -0.80, \qquad w_R = -\frac{-4.0}{4 + 1} = +0.80
```

Signs check out: the left rows (true $y \approx 1$, predicted 2.05) get pulled down, the right rows up. Note the shrinkage built into the formula — with only 4 rows of evidence and $\lambda = 1$, the leaf moves $0.80$, not the full $1.0$ the residuals suggest. That's L2 regularization working ([chapter 6](6-regularization-and-constraints.md)).

## Step 6 — the update

With learning rate $\eta = 0.3$: rows left of the split move to $2.05 - 0.3 \times 0.80 = 1.81$; right rows to $2.29$. Round two recomputes gradients at these *new* predictions and grows the next tree against them — each round fits what the previous ensemble still gets wrong. Twenty rounds in, the left rows sit within a few percent of 1.0.

That's the entire algorithm. Everything else in this guide — deeper trees, more features, sampling, constraints, GPUs — is this loop, made fast and made robust.

## Now run it for real

Python (in-memory numpy, the same eight rows):

```bash
make python && PYTHONPATH=build/python python3 - <<'EOF'
import numpy as np, bonsai
X = np.array([[0.2],[0.5],[0.9],[1.4],[2.1],[2.6],[3.3],[3.8]], dtype=np.float32)
y = np.array([1.0, 1.2, 0.9, 1.1, 3.0, 3.2, 2.9, 3.1], dtype=np.float32)
model = bonsai.train([
    ("booster.n_iters", "20"), ("booster.learning_rate", "0.3"),
    ("tree.max_depth", "2"), ("tree.min_data_in_leaf", "1"),
], X, y)
print(np.asarray(model.predict(X)).round(2))
print(model.dump())   # the trees, with the split thresholds and gains
EOF
```

`model.dump()`'s first tree reads `f0 <= 1.400000 ... gain=6.400000` with leaves `-0.8` and `+0.8` — the threshold, gain, and leaf values you just derived, to the printed digit. From here:

- The CLI equivalents (`bonsai fit/predict/eval/bench/dump/importance/info/params`) are in the [README](../../README.md) quickstart.
- [Chapter 1](1-gradient-boosting.md) does step 1 properly (why gradients, why second order); chapters 2–4 do steps 2–5 at production scale.
- Real-data configs to play with live in [`configs/`](../../configs) (California Housing, YearPredictionMSD), and `scripts/compare.py` runs the same experiment against xgboost/lightgbm/catboost.
