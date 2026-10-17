# 16. Multi-target regression: one tree per target

## The idea

A model that predicts several quantities from one feature matrix (the next five horizons of a price, the flow at eight river gauges, a few correlated sensors) can be fit as several models or as one. Fitting several costs several ingests of the same matrix, several binnings, several objects to version, and no shared early stopping. Fitting one raises the question this chapter answers: how should the trees share the work?

bonsai's answer is the one [chapter 12](12-multiclass.md) already gave for classes. Keep $K$ running score columns, one per target, and each round grow $K$ trees, tree $k$ on target $k$'s own gradient. Nothing in the histogram engine changes: a round is $K$ ordinary single-target rounds that happen to share one binned matrix, one row selection and one model.

## The math

Target $k$ of row $i$ has score $s_{ik}$ and label $y_{ik}$. For a separable loss $\mathcal{L} = \sum_i \sum_k \ell(s_{ik}, y_{ik})$, the gradient and hessian of target $k$ depend only on column $k$:

$$g_{ik} = \frac{\partial \ell}{\partial s_{ik}}, \qquad h_{ik} = \frac{\partial^2 \ell}{\partial s_{ik}^2}.$$

So the tree for target $k$ is exactly the tree a single-target fit on $(X, y_{\cdot k})$ would grow, from the same histograms with the same cuts. This is the fact the implementation pins: with every row selected, a multi-target fit predicts each target bit for bit as the single-target fit on that column alone (`tests/unit/test_multi_output_booster.cpp`, the invariant `multi-target-fit-matches-single-target-fits`). A sampled fit (Bernoulli, GOSS) draws its rows once per tree, so there the pin is determinism under a seed rather than equality with $K$ separate fits.

Prediction is $K$ tree walks per round, one column each. Early stopping scores the objective's loss pooled over the whole block, which for squared error is the mean over targets of each target's MSE.

## The alternative that was measured and declined

The other shape, one tree per round with $K$-wide leaves, scores a split by the sum of the per-target gains and stores a vector in every leaf. XGBoost ships it as `multi_strategy="multi_output_tree"` and CatBoost as `MultiRMSE`, so before any engine code the two shapes were raced in those libraries on six multi-target sets and a synthetic multi-horizon set (`benchmarks/multitarget-tradeoff-2026-10.md`, decision 161). Shared structure lost on five of seven by 4.5 to 18% aRRMSE, trained slower on XGBoost, and only predicted faster. Each target getting its own 200 splits beat $K$ targets sharing one budget. bonsai's $K$ separate fits, which the per-target booster reproduces, sat within 1.2% of XGBoost's per-target arm on every set over 1000 rows and led on two.

## What to expect in practice

Training cost scales linearly in $K$, as for multiclass: $K$ targets is $K$ times the trees and $K$ times the histogram work, with the ingest and binning paid once. The model is `n_iters * n_targets` trees, round-major, so `num_iteration` on `predict` still counts rounds.

The CPU growers serve it today, and the one ingest shows: on an EPYC host a four-target depthwise fit at 1M x 64 cost 0.84x of four single fits. A CUDA grower trains it too, through the host round trip every K-wide fit takes (host gradients up, leaf ids down), which a single-target device fit no longer pays since its objective went resident: on an RTX PRO 6000 the same four-target fit cost about 1.65x the four single device fits, and equals them bit for bit and in cost when they take the same round trip (decision 161). XGBoost on the same pod reads the opposite way, 0.36x on the device and 0.64x on the host, because its per-fit matrix build is large enough to be worth sharing; bonsai's is not, so a K-wide fit saves little ingest and, on the device, pays the round trip. Device prediction and SHAP decline a K-wide model and run on the host, as they do for multiclass.

## In the code

- **The label block.** A `Dataset` carries its labels as an `n_rows x n_targets` row-major block (`Dataset::n_targets`), a vector `y` being the block at width 1, and a row view gathers whole rows of it. `bonsai.Dataset(X, Y)` with a 2-D `Y` builds one.
- **The booster.** [`include/bonsai/multi_output_booster.hpp`](../../include/bonsai/multi_output_booster.hpp): `MultiOutputBooster<Obj, Gr, Sa>` copies target $k$'s column out of the score matrix, runs the single-output objective's `compute` and leaf renewal on it unchanged, and writes the tree's leaf values back into column $k$. The $K$-wide scaffolding it shares with the softmax booster (scores, init scores, raw scores, save and load) lives in `MultiOutputEnsemble`.
- **The knob.** `objective.n_targets`, read off the label block by the Python API and refused when it disagrees with it. A saved model carries it only when set, so a model saved before the knob existed loads unchanged.

## Quick use

```python
import bonsai

model = bonsai.BonsaiRegressor(n_iters=200, grower="depthwise").fit(X, Y)  # Y is (n_rows, K)
model.n_outputs_  # K
model.predict(X).shape  # (n_rows, K)
model.pred_contribs(X).shape  # (n_rows, K, n_features + 1)

ds = bonsai.Dataset(X, Y)
fitted = bonsai.train({"booster.n_iters": 200}, ds)
fitted.n_targets  # K
```

A `(n_rows, 1)` target is read as one target, so `predict` returns a vector, the way scikit-learn's regressors read it.
