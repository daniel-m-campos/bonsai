# Feature gap: bonsai vs xgboost / lightgbm / catboost

Regression-relevant features the reference libraries have and bonsai lacks,
sorted by expected fit/predict impact on the benchmark datasets (California
Housing, Year Prediction MSD). Categorical-variable features are out of scope
by request. Engine-internal performance work (uint8 bins, row-major layouts,
histogram pooling) is tracked separately — those aren't features one can
"enable" in the reference libraries for an A/B.

| # | Feature | xgboost | lightgbm | catboost | Expected impact | Status |
|---|---------|---------|----------|----------|-----------------|--------|
| 1 | Feature (column) subsampling per tree | `colsample_bytree` | `feature_fraction` | `rsm` | Fit latency ↓ ~proportionally to fraction (fewer histogram scans); RMSE neutral-to-better (decorrelates trees) | **landed** |
| 2 | GOSS (gradient one-side sampling) | — (GPU only) | `data_sample_strategy=goss` | — | Fit latency ↓ 2–3× at near-equal RMSE; only lightgbm comparable | **landed** |
| 3 | Early stopping on validation | `early_stopping_rounds` | `early_stopping_round` | `od_wait` | Fit latency ↓ whenever the model converges before `n_iters`; RMSE ~best-iteration | **landed** |
| 4 | L1 leaf regularization | `reg_alpha` | `lambda_l1` | — | RMSE small (usually ±0.1–0.5%); latency negligible | planned |
| 5 | Robust objectives (Huber / MAE / quantile) | `reg:pseudohubererror` etc. | `huber`, `mae`, `quantile` | `Huber`, `MAE`, `Quantile` | None on RMSE-scored benchmarks (MSE is the matched loss); useful capability, no A/B signal | deferred |
| 6 | Monotone constraints | `monotone_constraints` | same | same | Constraints can only cost RMSE on unconstrained benchmarks; no fit/predict win | not planned |
| 7 | Interaction constraints | yes | yes | — | Same reasoning as monotone | not planned |
| 8 | DART boosting | `booster=dart` | `boosting=dart` | — | Fits *slower* (re-predicts per iter); rare RMSE gains | not planned |
| 9 | Sparse-input handling / EFB | yes | yes (EFB) | yes | Benchmarks are dense; no signal | not planned |

Measurement protocol, per implemented feature: enable the equivalent knob in
bonsai and every reference library that has it, run
`scripts/compare.py` on both configs, and record RMSE and fit/predict
seconds deltas vs the feature-off baseline
(`benchmarks/results/{msd,ch}_final.*`).

## Results log

### 1. Feature subsampling — `tree.feature_fraction` (landed)

Year Prediction MSD, 200 iters, `feature_fraction = 0.8` vs baseline
(`msd_final` → `msd_ff08`):

| library | rmse | fit_s | Δrmse | Δfit |
|---|---|---|---|---|
| bonsai (depthwise) | 8.9859 | 24.3 | −0.06% | −11% |
| bonsai (leafwise) | 9.0863 | 10.6 | −0.01% | −15% |
| xgboost (`colsample_bytree`) | 9.1307 | 5.3 | −0.09% | −26% |
| lightgbm (`feature_fraction`) | 9.0810 | 6.2 | −0.02% | −30% |
| catboost (`rsm`) | 9.1468 | 9.7 | +0.03% | −15% |

Slight RMSE improvement and a fit speedup for every library. bonsai's
speedup is smaller than xgboost/lightgbm's because its per-node fixed
costs (parallel-region launches, histogram allocation, row partition)
don't shrink with the feature count.

### 2. GOSS — `dispatch.sampler_name = "goss"` (landed)

`sampler.top_rate = 0.2`, `sampler.other_rate = 0.1` (LightGBM defaults).
Year Prediction MSD, 200 iters (`msd_goss2`), vs the all_rows baseline:

| library | rmse | fit_s | Δrmse | Δfit |
|---|---|---|---|---|
| bonsai (depthwise, goss) | 8.9934 | 18.4 | +0.03% | −33% |
| bonsai (leafwise, goss) | 9.0757 | 10.9 | −0.13% | −13% |
| lightgbm (goss) | 9.0697 | 5.0 | −0.14% | −44% |

GOSS *improves* leafwise RMSE while cutting fit time, matching the
direction and magnitude of LightGBM's own GOSS delta (−0.14%).

Implementing this surfaced a real bug affecting every subsampling
sampler: `GrowResult.values` was only written for sampled rows, so
out-of-bag rows' training scores went stale relative to the model
(gradients computed against predictions that were missing whole trees).
Bernoulli quietly lost accuracy — depthwise 9.1873 → **8.9916** after
the fix — and GOSS diverged outright (RMSE 24.7, worse than predicting
the mean) because it re-selects rows by |grad| every iteration, feeding
on its own staleness. Fixed by routing unsampled rows through the
finished tree in bin space (`route_unsampled`), exactly matching the
float-threshold predict path.

### 3. Early stopping — `booster.early_stopping_rounds` (landed)

All libraries share a 90/10 train/valid split of the train file, lr
0.15, up to 400 iters, patience 20 (`msd_es`). Baseline for context is
the fixed 200-iter lr 0.05 run (`msd_final`):

| library | rmse | fit_s | rmse @ baseline |
|---|---|---|---|
| bonsai (depthwise) | 8.9918 | 20.6 | 8.9911 |
| bonsai (leafwise) | **8.9977** | 12.9 | 9.0871 |
| xgboost | 8.9900 | 8.8 | 9.1389 |
| lightgbm | 9.0016 | 9.1 | 9.0826 |
| catboost | 8.9593 | 22.4 | 9.1441 |

Early stopping lets every library run at a higher learning rate to its
own best iteration instead of a hand-tuned count; all five converge to
8.96–9.00. bonsai leafwise lands between xgboost and lightgbm on RMSE.
bonsai's per-iteration valid eval is incremental (score_base +
accumulate_last_tree), so the overhead is one tree-predict over the
valid set per iteration, and the kept model is truncated to the best
iteration like the references.
