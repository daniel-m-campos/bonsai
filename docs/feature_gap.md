# Feature gap: bonsai vs xgboost / lightgbm / catboost

Features the reference libraries have and bonsai lacks, sorted by expected
fit/predict impact on the benchmark datasets (California Housing, Year
Prediction MSD). Everything below is in scope; rows whose A/B needs harness
work first (a sparse or categorical dataset, an AUC metric) say so in their
impact column. Engine-internal performance work (uint8 bins, row-major
layouts, histogram pooling) is tracked separately — those aren't features
one can "enable" in the reference libraries for an A/B.

| # | Feature | xgboost | lightgbm | catboost | Expected impact | Status |
|---|---------|---------|----------|----------|-----------------|--------|
| 1 | Feature (column) subsampling per tree | `colsample_bytree` | `feature_fraction` | `rsm` | Fit latency ↓ ~proportionally to fraction (fewer histogram scans); RMSE neutral-to-better (decorrelates trees) | **landed** |
| 2 | GOSS (gradient one-side sampling) | — (GPU only) | `data_sample_strategy=goss` | — | Fit latency ↓ 2–3× at near-equal RMSE; only lightgbm comparable | **landed** |
| 3 | Early stopping on validation | `early_stopping_rounds` | `early_stopping_round` | `od_wait` | Fit latency ↓ whenever the model converges before `n_iters`; RMSE ~best-iteration | **landed** |
| 4 | L1 leaf regularization | `reg_alpha` | `lambda_l1` | — | RMSE small (usually ±0.1–0.5%); latency negligible | **landed** |
| 5 | Robust objectives (Huber / MAE / quantile) | `reg:pseudohubererror` etc. | `huber`, `mae`, `quantile` | `Huber`, `MAE`, `Quantile` | None on RMSE-scored benchmarks (MSE is the matched loss); MAE column added for these | **landed** |
| 6 | Monotone constraints | `monotone_constraints` | same | same | Constraints can only cost RMSE on unconstrained benchmarks; capability parity | **landed** |
| 7 | Interaction constraints | yes | yes | — | Same reasoning as monotone | **landed** |
| 8 | DART boosting | `booster=dart` | `boosting=dart` | — | Fits *slower* (re-predicts per iter); rare RMSE gains | **landed** |
| 9 | Feature importance (split count + gain) | `get_score` | `feature_importance` | `get_feature_importance` | No fit/predict impact; table-stakes introspection. Split-count is free; gain needs per-node gains stored at grow time | **landed** |
| 10 | Leaf renewal for MAE/quantile | built in (adaptive trees) | `RenewTreeOutput` | built in | Closes the known ~10% MAE gap vs refs on MAE/quantile objectives (see results log §5) | **landed** |
| 11 | Classification benchmark (AUC) | n/a (harness) | n/a | n/a | logloss objective exists but is only unit-tested; needs a live dataset (e.g. Higgs subset) + AUC column in compare.py | **landed** |
| 12 | Categorical features | one-hot / partition | native (Fisher) | ordered target stats | Biggest real-world capability gap; needs a categorical dataset (e.g. Adult, Amazon) in the harness first | planned |
| 13 | Prediction extras: staged predict, `pred_leaf`, tree dump | yes | yes | yes | Debug/introspection; small, each independent | **landed** |
| 14 | Warm start / training continuation | `xgb_model` | `init_model` | `init_model` | Fit latency for iterative workflows; booster already saves/loads full state | **landed** |
| 15 | TreeSHAP (`pred_contribs`) | yes | yes | yes | Modern attribution standard; real algorithm, sized as its own project | planned |
| 16 | Multi-class / softmax objective | yes | yes | yes | K-output leaves touch Objective, Booster, and Tree — largest structural change | planned |
| 17 | Sparse-input handling / EFB | yes | yes (EFB) | yes | Needs a sparse dataset in the harness to enable or measure anything | planned |

Measurement protocol, per implemented feature: enable the equivalent knob in
bonsai and every reference library that has it, run
`scripts/compare.py` on both configs, and record RMSE and fit/predict
seconds deltas vs the feature-off baseline
(`benchmarks/results/{msd,ch}_final.*`). Features without a knob to A/B
(importance, prediction extras) are instead verified for agreement against
the reference libraries' outputs on the same model shape.

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

### 7. Interaction constraints — `tree.interaction_constraints` (landed)

Feature groups as strings (`["0,1", "2,3"]` in TOML, `0+1,2+3` via
`--set`). A node may split on a feature only if some group contains it
together with every feature already on the path (features outside all
groups may keep splitting alone); the allowed set propagates through
`SplitInput::allowed/path`. Oblivious grower rejects the option. Unit
test walks every root-to-leaf path and asserts groups never mix.

California Housing, groups {0–3 economic} / {4–7 geographic}
(`ch_interaction`): forbidding cross-group interaction costs every
library the same ~9%; catboost (no such option) is the unchanged
control at 0.5147.

| library | rmse | Δrmse |
|---|---|---|
| bonsai (depthwise) | 0.5195 | +9.6% |
| bonsai (leafwise) | 0.5208 | +8.6% |
| xgboost | 0.5236 | +8.7% |
| lightgbm | 0.5198 | +9.1% |

### 8. DART — `booster.dart_drop_rate` (landed)

Each iteration drops every existing tree with probability
`dart_drop_rate`, computes gradients against the reduced model
(dropped trees' train contributions recovered by bin-space routing, no
per-tree caches), then rescales: new tree 1/(k+lr), dropped trees
k/(k+lr) — xgboost's `normalize_type="tree"` factors. The DART paper's
1/(k+1) starves the new tree by ~1/lr under shrinkage; implementing it
literally cost bonsai 0.88 RMSE before switching to the reference
scheme. Incompatible with early stopping (rescaling invalidates the
incremental valid scores) — the combination throws.

California Housing, `drop_rate = 0.1`, 200 iters (`ch_dart2`): DART
regularizes (everyone is worse than plain GBDT at this fixed budget)
and bonsai's implementation degrades the least; catboost (no DART) is
the control.

| library | rmse | fit_s |
|---|---|---|
| bonsai (depthwise, dart) | 0.5930 | 1.9 |
| bonsai (leafwise, dart) | 0.6031 | 1.2 |
| xgboost (dart) | 0.6255 | 2.9 |
| lightgbm (dart) | 0.7022 | 1.1 |
| catboost (control) | 0.5147 | 0.3 |

### 9. Feature importance — `bonsai importance` / `feature_importances_` (landed)

Growers stamp each split's gain at grow time (`split_gains` on
`DenseTree`, `level_gains` on `ObliviousTree`, serialized — format v5);
`IBooster::feature_importance(type)` accumulates over trees. Surfaces:
CLI `bonsai importance --model m.msgpack`, Python
`Model.feature_importance(type)`, `BonsaiRegressor.importance()` and
sklearn-style normalized `feature_importances_`.

Agreement check (the knob-less protocol) on California Housing:
bonsai and lightgbm agree on **both** types — including their
disagreement with each other. Gain crowns `MedInc`; split-count crowns
geography (`Longitude`/`Latitude` take ~2x more splits at lower gain
each), the textbook case for preferring gain. Asserted in
`python/tests/test_bindings.py::test_feature_importance_agreement`.

| feature | bonsai gain | bonsai split | lgbm split |
|---|---|---|---|
| MedInc | 113,908 (top) | 1,129 | 764 |
| Longitude | 22,255 | **1,900** (top) | **1,144** (top) |
| Latitude | 19,821 | 1,889 | 1,142 |

### 10. Leaf renewal — automatic for `mae` / `huber` / `quantile` (landed)

Growers report per-row leaf assignments (`GrowResult::leaf_ids`); when an
objective defines `renew_leaf`, the booster regroups rows by leaf and
replaces each Newton step with the loss-optimal value over that leaf's
residuals (median / alpha-quantile / LightGBM-style clamped-mean huber)
before the score update — under DART, before rescaling. One-iteration
exact-math test: with lr = 1, an MAE tree reproduces the labels.

Year Prediction MSD, 200 iters, leafwise, MAE metric (`msd_obj2_*` vs
`msd_obj_*`):

| objective | bonsai (before → after) | xgboost | lightgbm | catboost |
|---|---|---|---|---|
| mae | 6.81 → **6.094** | 6.155 | 6.095 | 6.159 |
| huber (δ=1) | 6.81 → **6.093** | diverged | 6.792 | diverged |
| quantile (α=.5) | 7.11 → **6.096** | 6.158 | 6.094 | 6.159 |

The ~10% gap is fully closed: bonsai now ties lightgbm on mae/quantile
and posts the best huber of the field at matched δ=1.

### 13+14. Prediction extras & warm start (landed)

`predict_at(k)` / `--num-iteration` / `predict(X, num_iteration=k)`;
one-pass `staged_predict` (n_iters × n_rows); `predict_leaf` per-tree
leaf indices; `bonsai dump` / `Model.dump()` indented tree text with
feature names and gains. Warm start continues a saved model — CLI
`fit --init-model m.msgpack`, Python `fit(..., init_model=...)` —
rebinning with the loaded model's mappers and rebuilding training
scores by bin-space routing; a 3+3-iteration continuation matches a
straight 6-iteration run within FP-regrouping tolerance (tested), and
early stopping seeds its incremental valid scores from the pre-existing
trees.

### 11. Classification benchmark — HIGGS + AUC (landed)

`scripts/fetch_higgs.py` streams the first 550k rows of UCI HIGGS
(500k train / 50k test, 28 features); `configs/higgs.toml` runs the
logloss objective; compare.py maps it to `binary:logistic` / `binary` /
`Logloss` (CatBoostClassifier) and adds an AUC column (regression rows
show "-"). A rank-sum `auc` metric also joined the C++ metric registry
for `bonsai eval` / fit ticks.

First live outing for the logloss path (previously unit-tested only),
200 iters, lr 0.1:

| library | auc | logloss-ish rmse | fit_s |
|---|---|---|---|
| bonsai (depthwise) | **0.8218** | 0.4146 | 11.7 |
| bonsai (leafwise) | 0.8139 | 0.4191 | 5.5 |
| xgboost | 0.8111 | 0.4207 | 1.8 |
| lightgbm | 0.8146 | 0.4188 | 2.4 |
| catboost | 0.7321 | 0.5161 | 6.4 |

bonsai leafwise lands between xgboost and lightgbm on AUC at matched
settings; depthwise (16x the nodes) leads everyone. catboost's 0.73 at
these matched knobs is an outlier — its defaults want ordered boosting
and auto-tuned lr; recorded as-is per protocol, not tuned.