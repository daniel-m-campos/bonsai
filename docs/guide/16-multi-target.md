# 16. Multi-target regression: one tree per target

## The idea

A model that predicts several quantities from one feature matrix (the next five horizons of a price, the flow at eight river gauges, a few correlated sensors) can be fit as several models or as one. Fitting several costs several ingests of the same matrix, several binnings, several objects to version, and no shared early stopping. Fitting one raises the question this chapter answers: how should the trees share the work?

bonsai's answer is the one [chapter 12](12-multiclass.md) already gave for classes. Keep $K$ running score columns, one per target, and each round grow $K$ trees, tree $k$ on target $k$'s own gradient. Nothing in the histogram engine changes: a round is $K$ ordinary single-target rounds that happen to share one binned matrix, one row selection and one model.

## The math

Target $k$ of row $i$ has score $s_{ik}$ and label $y_{ik}$. For a separable loss $\mathcal{L} = \sum_i \sum_k \ell(s_{ik}, y_{ik})$, the gradient and hessian of target $k$ depend only on column $k$:

$$g_{ik} = \frac{\partial \ell}{\partial s_{ik}}, \qquad h_{ik} = \frac{\partial^2 \ell}{\partial s_{ik}^2}.$$

So the tree for target $k$ is exactly the tree a single-target fit on $(X, y_{\cdot k})$ would grow, from the same histograms with the same cuts. This is the fact the implementation pins: with every row selected, a multi-target fit predicts each target bit for bit as the single-target fit on that column alone (`tests/unit/test_multi_output_booster.cpp`, the invariant `multi-target-fit-matches-single-target-fits`). A sampled fit (Bernoulli, GOSS) draws its rows once per tree, so there the pin is determinism under a seed rather than equality with $K$ separate fits.

Prediction is $K$ tree walks per round, one column each. Early stopping scores the objective's loss pooled over the whole block, which for squared error is the mean over targets of each target's MSE, so the target with the largest variance decides when the fit stops. The pool is also where a target's scale enters the fit, beside the absolute knobs `min_split_gain` and Huber's `delta` (the MSE and quantile splits themselves are scale-free). Targets measured in different units are therefore standardized per column before the fit and the prediction mapped back, one affine map per output that leaves the trees untouched.

## The alternative that was measured and declined

The other shape, one tree per round with $K$-wide leaves, scores a split by the sum of the per-target gains and stores a vector in every leaf. XGBoost ships it as `multi_strategy="multi_output_tree"` and CatBoost as `MultiRMSE`, so before any engine code the two shapes were raced in those libraries on six multi-target sets and a synthetic multi-horizon set (`benchmarks/multitarget-tradeoff-2026-10.md`, decision 161). Shared structure lost on five of seven by 4.5 to 18% aRRMSE, trained slower on XGBoost, and only predicted faster. Each target getting its own 200 splits beat $K$ targets sharing one budget. bonsai's $K$ separate fits, which the per-target booster reproduces, sat within 1.2% of XGBoost's per-target arm on every set over 1000 rows and led on two.

## What to expect in practice

Training cost scales linearly in $K$, as for multiclass: $K$ targets is $K$ times the trees and $K$ times the histogram work, with the ingest and binning paid once. The model is `n_iters * n_targets` trees, round-major, so `num_iteration` on `predict` still counts rounds.

Every grower serves it, and the shared ingest shows on both planes: on an EPYC host a four-target depthwise fit at 1M x 64 cost 0.84x of four single fits, and on an RTX PRO 6000 the same fit on a CUDA grower cost 0.75 to 0.79x of four single device fits (decision 161). On the device the round is resident under the same rules as a single-target fit ([chapter 10](10-gpu-training.md)): the label and score blocks live on the device as K columns, tree k derives its gradients from column k and fuses its leaves back into it, so the K-wide fit equals K single device fits bit for bit; `BONSAI_HOST_OBJECTIVE=1` sends it through the host round trip like any other fit. XGBoost on the same pod reads 0.36x on the device and 0.64x on the host, because its per-fit matrix build is large enough to be worth sharing; bonsai's K-wide device fit still reads 5.6x under XGBoost's. A device-resident `Dataset` predicts and explains on the device too, as K width-1 walks, one per target: the same four-target model predicted a million rows in 7 ms where the host walk took 0.3 to 0.8 s, and explained them in 1.1 to 1.3 s where the host took 26 to 62 s, both matching the host walk (predict bit for bit, SHAP within the device walk's fp32 rounding). An eval set is scored on the device as well: the validation plane holds the K seed columns, each round adds tree k into column k and sums its loss there, and the pooled loss is the mean over the columns, so early stopping never walks the eval set on the host. Same pod, the four-target fit with a million-row eval set took 0.76 to 1.06 s against 1.7 to 2.3 s with the host walk, where the fit alone takes 0.34 to 0.72 s and a single-target fit with the same eval set 0.44 to 0.55 s (decision 163). The device sums the loss in double partials where the host sums in single precision, so the two series agree to the third or fourth significant digit at a million rows, not bit for bit.

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
