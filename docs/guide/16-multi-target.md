# 16. Multi-target regression: one tree per target

## The idea

A model that predicts $K$ quantities from one feature matrix (the next five horizons of a price, the flow at eight river gauges, a few correlated sensors) can be fit as $K$ models or as one. Fitting $K$ separate models ingests the matrix (reads it and bins it, bucketing feature values so splits are scored over bins) $K$ times, gives you $K$ objects to version, and stops each fit on its own early-stopping signal rather than a shared one. Fitting one raises the question this chapter answers: how should the trees share the work?

bonsai's answer is the one [chapter 12](12-multiclass.md) already gave for classes. Keep $K$ running score columns, one per target, and each round grow $K$ trees, tree $k$ on target $k$'s own gradient. Nothing in the histogram engine changes: a round is $K$ ordinary single-target rounds that happen to share one binned matrix and one model (each tree draws its own rows when a sampler is on).

## The math

Target $k$ of row $i$ has score $s_{ik}$ and label $y_{ik}$. For a separable loss $\mathcal{L} = \sum_i \sum_k \ell(s_{ik}, y_{ik})$, the gradient and hessian of target $k$ depend only on column $k$:

$$g_{ik} = \frac{\partial \ell}{\partial s_{ik}}, \qquad h_{ik} = \frac{\partial^2 \ell}{\partial s_{ik}^2}.$$

So the tree for target $k$ is exactly the tree a single-target fit on $(X, y_{\cdot k})$ would grow, from the same histograms with the same cuts. This is the fact the implementation pins: with every row selected, a multi-target fit predicts each target bit for bit as the single-target fit on that column alone (`tests/unit/test_multi_output_booster.cpp`, the invariant `multi-target-fit-matches-single-target-fits`). A sampled fit (Bernoulli, GOSS) draws its rows once per tree, so there the pin is determinism under a seed rather than equality with $K$ separate fits.

Prediction is $K$ tree walks per round, one column each. Early stopping scores the objective's loss pooled over the whole block, which for squared error is the mean over targets of each target's MSE, so the target with the largest variance decides when the fit stops. The pool is also where a target's scale enters the fit, beside the absolute knobs `min_gain_to_split` and `huber_delta` (the MSE and quantile splits themselves are scale-free). If your targets use different units, standardize each target before the fit (subtract its mean, divide by its standard deviation) and map each prediction back with the same shift and scale; bonsai does neither for you. The mapping back is one affine map per target and does not touch the trees.

## The alternative that was measured and declined

The other shape, one tree per round with $K$-wide leaves, scores a split by the sum of the per-target gains and stores a vector in every leaf. XGBoost ships it as `multi_strategy="multi_output_tree"` and CatBoost as `MultiRMSE`, so before any engine code the two shapes were raced in those libraries on six multi-target sets and a synthetic multi-horizon set (`benchmarks/multitarget-tradeoff-2026-10.md`, decision 161). Shared structure had a higher aRRMSE (the mean over targets of test RMSE divided by the RMSE of predicting the training mean) than separate per-target trees in the same library on five of seven sets, by 4.5 to 18%. On XGBoost it trained 1.4 to 1.8x slower at 8 and 16 targets. Across both libraries it predicted 3 to 15x faster. Each target getting its own 200 trees beat $K$ targets sharing one set of 200 trees. bonsai's $K$ separate fits scored within 1.2% of XGBoost's `one_output_per_tree` aRRMSE on every set over 1000 rows, and had the lowest aRRMSE of all compared configurations on two sets (rf1 and rf2). The per-target booster reproduces these fits.

## What to expect in practice

Training cost scales linearly in $K$, as for multiclass: $K$ targets is $K$ times the trees and $K$ times the histogram work, with the ingest and binning paid once. The model is `n_iters * n_targets` trees, round-major, so `num_iteration` on `predict` still counts rounds.

Every grower (the tree-growth strategy, such as depthwise or leafwise) supports multi-target fits, and the shared ingest lowers the cost on both CPU and GPU. Every row below is measured with four targets (K = 4) at 1M x 64 with 50 rounds at depth 6, except the last row, which is single-target. All pods are RTX PRO 6000 Blackwell Server Edition. The pod column numbers the pods in decision 161 (pods 1 to 4) and decision 163 (pod 5); a ratio compares runs on one pod only ([decision 161](../decisions.md), [decision 163](../decisions.md)).

| measurement | pod | result | compared with |
|---|--:|--:|--:|
| bonsai host depthwise fit | 1 | 0.84x | four single host fits |
| bonsai device fit, three CUDA growers | 3 | 0.75 to 0.79x | four single device fits |
| XGBoost `hist` fit, device / host | 2 | 0.36x / 0.64x | four single XGBoost fits |
| bonsai device fit through the host round trip | 2 | 0.59 s | XGBoost device fit, 1.34 s (bonsai 2.3x faster) |
| device predict, 1M rows | 4 | 7 ms | host walk, 0.3 to 0.8 s |
| device SHAP, 1M rows | 4 | 1.1 to 1.3 s | host walk, 26 to 62 s |
| fit with a 1M-row eval set, scored on the device | 5 | 0.76 to 1.06 s | eval set on the host walk, 1.7 to 2.3 s |
| fit alone | 5 | 0.34 to 0.72 s | fit with the eval set on the device, 0.76 to 1.06 s |
| single-target fit with the same eval set | 5 | 0.44 to 0.55 s | four-target fit with the same eval set, 0.76 to 1.06 s |

XGBoost's ratios (0.36x and 0.64x) sit below bonsai's because its per-fit data-matrix build (XGBoost's DMatrix, which includes the quantile sketch of bin boundaries) is the cost the four targets share (decision 161). The 2.3x row predates the resident round described next.

On the device the round is resident under the same rules as a single-target fit ([chapter 10](10-gpu-training.md)): the label and score blocks live on the device as K columns, tree k derives its gradients from column k and fuses its leaves back into it, so the K-wide fit equals K single device fits bit for bit; `BONSAI_HOST_OBJECTIVE=1` sends it through the host round trip like any other fit.

A device-resident `Dataset` predicts and explains on the device too, as K width-1 walks, one per target. Both match the host walk: predict bit for bit, SHAP within the device walk's fp32 rounding.

An eval set is scored on the device as well: the validation plane holds the K seed columns, each round adds tree k into column k and sums its loss there, and the pooled loss is the mean over the columns, so early stopping never walks the eval set on the host.

Limit: the device and host eval loss series are not bit-identical. The device sums the loss in double precision and the host in single precision, so the two series agree to the third or fourth significant digit at a million rows.

## In the code

- **The label block.** A `Dataset` carries its labels as an `n_rows x n_targets` row-major block (`Dataset::n_targets`), a vector `y` being the block at width 1, and a row view gathers whole rows of it. `bonsai.Dataset(X, Y)` with a 2-D `Y` builds one.
- **The booster.** [`include/bonsai/multi_output_booster.hpp`](../../include/bonsai/multi_output_booster.hpp): `MultiOutputBooster<Obj, Gr, Sa>` copies target $k$'s column out of the score matrix, runs the single-output objective's `compute` and leaf renewal on it unchanged, and writes the tree's leaf values back into column $k$. `MultiOutputEnsemble` holds the code that this booster shares with the multiclass booster ([chapter 12](12-multiclass.md)): the $K$-wide score matrix, init scores, raw scores, save, and load.
- **The parameter.** `objective.n_targets` (integer, default 1) sets how many trees each round grows, one per target. The Python estimators set it from the width of a 2-D `y` and do not take it as an argument. `bonsai.train` raises `ValueError` if you set a value other than 1 that differs from the number of label columns in the `Dataset`. A saved model stores it only when it is above 1, so a model saved before the parameter existed loads unchanged.

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
