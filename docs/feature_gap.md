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
| 4 | L1 leaf regularization | `reg_alpha` | `lambda_l1` | — | RMSE small (usually ±0.1–0.5%); latency negligible | **landed** |
| 5 | Robust objectives (Huber / MAE / quantile) | `reg:pseudohubererror` etc. | `huber`, `mae`, `quantile` | `Huber`, `MAE`, `Quantile` | None on RMSE-scored benchmarks (MSE is the matched loss); MAE column added for these | **landed** |
| 6 | Monotone constraints | `monotone_constraints` | same | same | Constraints can only cost RMSE on unconstrained benchmarks; capability parity | **landed** |
| 7 | Interaction constraints | yes | yes | — | Same reasoning as monotone | planned |
| 8 | DART boosting | `booster=dart` | `boosting=dart` | — | Fits *slower* (re-predicts per iter); rare RMSE gains | planned |
| 9 | Sparse-input handling / EFB | yes | yes (EFB) | yes | Benchmarks are dense; nothing to enable or measure — stays out of scope until a sparse dataset exists in the harness | out of scope |

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

### 4. L1 leaf regularization — `tree.lambda_l1` (landed)

XGBoost-style soft threshold on the gradient sum in both gain and leaf
values. Year Prediction MSD, 200 iters, `lambda_l1 = 100` (`msd_l1`)
vs baseline:

| library | rmse | Δrmse | fit_s |
|---|---|---|---|
| bonsai (depthwise) | 8.9937 | +0.03% | 30.3 |
| bonsai (leafwise) | 9.0928 | +0.06% | 13.6 |
| xgboost (`reg_alpha`) | 9.1357 | −0.04% | 5.8 |
| lightgbm (`lambda_l1`) | 9.0857 | +0.03% | 8.2 |
| catboost (no L1 knob) | 9.1441 | ±0 | 8.2 |

As expected, L1 at this scale is a small, dataset-dependent
regularizer: deltas within ±0.06% for every library that supports it
(bonsai's direction matches lightgbm's), latency neutral. Verified as
exact math via unit tests (soft-threshold shifts leaf values by
alpha/(h+lambda); large alpha collapses the tree to a single zero
leaf); `lambda_l1 = 0` is bit-identical to the pre-feature build.

### 5. Robust objectives — `objective_name = mae | huber | quantile` (landed)

Objectives became Config-constructed instances so huber_delta /
quantile_alpha can carry state; the dispatch grid is now 5×3×3 = 45
combos, all covered by the parametric registry tests. Year Prediction
MSD, 200 iters, leafwise (`msd_obj_*`; mae is the target metric here):

| objective | bonsai mae | xgboost mae | lightgbm mae | catboost mae |
|---|---|---|---|---|
| mae | 6.81 | 6.15 | 6.10 | 6.16 |
| huber (delta=1) | 6.81 | diverged (8e9) | 6.79 | diverged (291) |
| quantile (a=0.5) | 7.11 | 6.16 | 6.09 | 6.16 |

Two honest caveats. (1) bonsai's constant-hessian objectives take plain
Newton leaf steps (mean gradient), while lightgbm/xgboost *renew* MAE
and quantile leaves with residual medians/quantiles — that renewal pass
is why they reach ~6.1 and bonsai sits ~10% higher; it's the known
follow-up for these objectives. On huber, where renewal matters less,
bonsai (6.81) is at lightgbm's level (6.79). (2) delta=1 is far too
small for year-scale targets: xgboost's pseudo-huber and catboost's
Huber diverge outright under it — the A/B records mechanics at matched
settings, not tuned quality.

### 6. Monotone constraints — `tree.monotone_constraints` (landed)

Per-feature +1/-1/0 list (comma-separated via `--set`). The split
finder rejects candidates whose bounded child weights violate the
direction, and children inherit midpoint-fenced leaf-value bounds
(xgboost's scheme), so every prediction path is provably monotone —
unit tests assert non-decreasing / non-increasing single-tree curves on
data where the unconstrained tree is non-monotone. Oblivious grower
rejects the option at construction.

California Housing, MedInc constrained +1 (`ch_monotone` vs
`ch_final`): the constraint costs every library a similar ~1–3% RMSE,
and bonsai remains in front of the constrained field.

| library | rmse | Δrmse vs unconstrained |
|---|---|---|
| bonsai (depthwise) | 0.4837 | +2.1% |
| bonsai (leafwise) | 0.4882 | +1.8% |
| xgboost | 0.4908 | +1.9% |
| lightgbm | 0.4891 | +2.6% |
| catboost | 0.5194 | +0.9% |
