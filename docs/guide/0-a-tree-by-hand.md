# 0. A tree by hand

## The idea

Before any chapter's math or code, run the full training loop once on **eight rows, one feature, one boosting round** (one new tree fitted to the current errors). The result is one tree with a single split, $x \le 1.4$, and leaf values of $-0.80$ and $+0.80$ before the learning rate is applied. Every number below is computed with the exact formulas the library ships for mean squared error (MSE): the gradient (`grad = pred − y`, how much the loss changes as the prediction rises) and the hessian (`hess = 1`, the curvature of the loss). Gain and leaf values come from [`split.hpp`](../../include/bonsai/split.hpp). When you later meet the same quantities at 16 million rows, nothing will have changed but the row count.

The data: predicting `y` from a single feature `x`:

| row | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| x | 0.2 | 0.5 | 0.9 | 1.4 | 2.1 | 2.6 | 3.3 | 3.8 |
| y | 1.0 | 1.2 | 0.9 | 1.1 | 3.0 | 3.2 | 2.9 | 3.1 |

Clearly two regimes: `y ≈ 1` below `x ≈ 2`, `y ≈ 3` above.

## Step 1: the base score and the gradients

Boosting starts from a constant. For MSE that's the mean: $F_0 = \bar{y} = 2.05$. The per-row gradient and hessian of $\tfrac12(F - y)^2$ at $F_0$:

```math
g_i = F_0 - y_i, \qquad h_i = 1
```

| row | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| $g_i$ | 1.05 | 0.85 | 1.15 | 0.95 | −0.95 | −1.15 | −0.85 | −1.05 |

The tree we're about to grow will fit *these*, not `y`. Rows the model over-predicts carry positive gradients; the leaf they land in will get a negative value that pulls them down.

## Step 2: binning

bonsai splits each feature at cuts (threshold values taken from the data itself, by [`create_cuts`](../../src/bin_mapper.cpp)). A feature with at most `max_bin - 2` distinct values gets one cut per distinct value, so here $c = (0.2,\, 0.5,\, 0.9,\, 1.4,\, 2.1,\, 2.6,\, 3.3,\, 3.8)$, followed by the `FLT_MAX` and $+\infty$ sentinels. Bin $b$ holds the values in $(c_{b-1}, c_b]$, so every row gets its own bin: row $i$ lands in **bin $i$**. (With more distinct values than that budget, `create_cuts` instead sorts a sample of the feature and keeps evenly spaced values, or weights the cuts when ties are heavy.) At scale, [chapter 2](2-binning-and-histograms.md) covers this: the cuts are chosen once before training, and each row then stores only its bin index per feature, one byte at the default `max_bin` of 255.

## Step 3: the histogram

One cell per bin, each accumulating $(\sum g, \sum h)$ of its rows: cells 0–7 hold one row each, so cell $b$ carries row $b$'s gradient with $H = 1$ (cell 0: $G{=}1.05$; cell 1: $G{=}0.85$; and so on). Node totals: $G = 0$, $H = 8$.

## Step 4: the gain scan

A split "bin ≤ b" (equivalently $x \le c_b$) scores by how much loss it is estimated to remove ([chapter 3](3-finding-splits.md)). Here $\lambda = 1$ is the L2 penalty, which shrinks every leaf value toward zero:

```math
\text{gain}(b) = \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{G^2}{H + \lambda}
```

One prefix scan over the cells, keeping running $G_L, H_L$:

| after bin $b$ | 0 | 1 | 2 | **3** | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| $G_L$ | 1.05 | 1.90 | 3.05 | 4.00 | 3.05 | 1.90 | 1.05 |
| $H_L$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| gain | 0.69 | 1.72 | 3.88 | **6.40** | 3.88 | 1.72 | 0.69 |

(Sample: at $b=3$, $\text{gain} = \tfrac{4.0^2}{5} + \tfrac{(-4.0)^2}{5} - 0 = 6.40$.)

The winner is $b = 3$: threshold $c_3 = 1.4$, i.e. "$x \le 1.4$", exactly the boundary between the two regimes. No sorting happened at split time; finding it cost one pass over the cells.

## Step 5: leaf values

Each child's Newton step:

```math
w^{\ast} = -\frac{G}{H + \lambda}: \qquad w_L = -\frac{4.0}{4 + 1} = -0.80, \qquad w_R = -\frac{-4.0}{4 + 1} = +0.80
```

The signs are correct: the left rows (true $y \approx 1$, predicted 2.05) move down and the right rows move up. The formula also shrinks the step: with 4 rows and $\lambda = 1$, each leaf moves $0.80$ instead of the full $1.0$ the residuals suggest. This is L2 regularization ([chapter 6](6-regularization-and-constraints.md)).

## Step 6: the update

With learning rate $\eta = 0.3$: rows left of the split move to $2.05 - 0.3 \times 0.80 = 1.81$; right rows to $2.29$. Round two recomputes gradients at these *new* predictions and grows the next tree against them. Each round fits what the previous ensemble still gets wrong. Twenty rounds in, the four left predictions are 1.07, 1.12, 1.00 and 1.10 (the first four values the example below prints), near their own targets of 1.0, 1.2, 0.9 and 1.1.

That's the entire algorithm. Everything else in this guide (deeper trees, more features, sampling, constraints, GPUs) is this loop, made fast and made robust.

## Now run it for real

Python (in-memory numpy, the same eight rows):

```python {.run}
import numpy as np
import bonsai

X = np.array([[0.2], [0.5], [0.9], [1.4], [2.1], [2.6], [3.3], [3.8]], dtype=np.float32)
y = np.array([1.0, 1.2, 0.9, 1.1, 3.0, 3.2, 2.9, 3.1], dtype=np.float32)
model = bonsai.train(
    {
        "booster.n_iters": 20,
        "booster.learning_rate": 0.3,
        "tree.max_depth": 2,
        "tree.min_data_in_leaf": 1,
    },
    X,
    y,
)
print(np.asarray(model.predict(X)).round(2))
print(model.dump())  # the trees, with the split thresholds and gains
```

`model.dump()`'s first tree reads `f0 <= 1.4 ... gain=6.4` with leaves `-0.79999995` and `0.8`: the threshold, gain, and leaf values you just derived. The dump prints enough digits to read each number back exactly. The left leaf therefore shows `-0.79999995` instead of `-0.8`: the sum is computed in float32 (32-bit floating point, about 7 significant digits) and ends one representable step away from -0.8. From here:

- The CLI equivalents (`bonsai fit/predict/eval/info/params`) are in the [README](../../README.md) quickstart; `bonsai bench` and `bonsai importance` appear in [chapter 11](11-performance-engineering.md) and [chapter 8](8-feature-importance.md), and `bonsai --help` lists all eight subcommands, including `dump`.
- [Chapter 1](1-gradient-boosting.md) explains step 1: why the tree fits gradients, and why it also uses the hessian (the second derivative of the loss). Chapters 2 to 4 cover steps 2 to 5 at production scale.
- Real-data configs to play with live in [`configs/`](../../configs) (California Housing, YearPredictionMSD, Higgs, Covertype, a9a), and `scripts/compare.py` runs the same experiment against XGBoost, LightGBM, and CatBoost.
