# Feature-selection probe: does honest shadow-feature selection buy accuracy, and how does it price against CatBoost select_features? (2026-07)

Feature selection is the recurring "the other libraries have it" pitch: CatBoost ships `select_features`, and the shadow-feature / Boruta idea (append permuted copies of every column, keep only real features that beat the permuted noise) is a well-known honest wrapper. This probe runs the feature-admission method (measure the benefit at zero core cost first): a roughly 100-line shadow prototype fit around stock bonsai, priced against CatBoost's own in-library selector and against the plain top-k-by-gain truncation the prototype's importance ranking already provides. If the prototype earns its keep the recommendation is a `bonsai.select` Python module (zero bonsai-core cost) and this probe is its ceiling measurement; if it does not, the honest finding and the noise-recovery exhibit feed the guide chapter that follows.

Probe script: `scripts/probe_feature_selection.py`. Raw rows: `benchmarks/results/feature-selection-probe-2026-07.jsonl`. Everything ran locally (CPU) in the TabArena-Lite gauge venv (CatBoost 1.2.10, xgboost 3.3.0, scikit-learn 1.7.2), with bonsai from `build-tabarena/python`. Knobs are imported from `bonsai.bench.params` (`catboost_core`, `xgb_core`) and metrics from `bonsai.bench.metrics`, per the one-source-of-truth provenance rule.

## Two regimes, one pool

The datasets are drawn from the rung-0 12-dataset pool (`scripts/probe_ordered_boosting_rung0.py`, imported read-only for its loader and its splits: gauge fold-0 for gauge sets, stratified 75/25 seed 42 for the extensions, a 20% stratified validation slice seed 42 for early stopping). Two regimes probe the two ways selection can matter.

REAL-WIDE, where the feature count is the point and selection is measured on real redundancy: QSAR-TID-11 (1024 features, the flagship), superconductivity (81), spambase (57).

NOISE-INJECTED, where the ground truth is known: houses, concrete_compressive_strength, wind, breast_cancer, pima_diabetes, MagicTelescope, each augmented with shuffled-copy noise columns equal to the original feature count. Each real column is permuted independently with a fixed seed-42 rng, which destroys its target association while keeping its marginal, so the injected columns are known-noise and the selection arms get a precision/recall grade. The injected count and the seed are recorded in the jsonl.

## Arms (per dataset)

The truncation arms run at arm 4's budget: k is the count the shadow arm keeps, so arm 3 versus arm 4 isolates the shadow machinery from mere truncation at equal budget.

1. bonsai_all: all features, the baseline; also the source of the gain ranking arm 3 uses.
2. cat_all: CatBoost Plain matched, all features (separates the selection effect from the library effect).
3. bonsai_topk_gain: refit bonsai on the top-k features by bonsai's own gain importance from arm 1's fit.
4. bonsai_shadow: the prototype. Over 5 seeds (42..46), append one permuted shadow copy of every feature, fit bonsai, and keep every real feature whose gain importance exceeds the 95th percentile of all shadow importances; keep features selected in at least 3 of 5 seeds; refit on the kept set and measure the holdout.
5. cat_select: CatBoost `select_features` with algorithm RecursiveByLossFunctionChange at the matched knobs, eliminating down to k in 5 steps, final model refit on its chosen set.
6. xgb_topk_gain: xgboost matched all-features fit, then refit on its top-k by gain, same k (ecosystem reference).

## Protocol

Matched knobs throughout (the quality-campaign shape): depth 6, learning_rate 0.05, iterations cap 1000, early_stopping_rounds 50, min_data_in_leaf 20, lambda_l2 1.0, max_bin 255 bonsai / 254 CatBoost, CatBoost boosting_type Plain. Metric_error is lower-better: rmse for regression, 1 minus roc_auc for binary. The shadow selection touches only the train and validation slices (the holdout is untouched by selection); the shadow copies are appended per-seed and read for importances only, never predicted on. Chance band per decision 55: about 2% relative of the metric for rmse (computed as 0.02 times bonsai_all per dataset), 0.001 absolute for 1 minus roc_auc.

Sign convention: every "vs" delta is `bonsai_all minus arm` or `arm minus arm`, positive when the named lever LOWERS error. `shadow_vs_all = bonsai_all minus bonsai_shadow`; `shadow_vs_topk = bonsai_topk_gain minus bonsai_shadow` (positive means the shadow machinery beats plain truncation at the same k).

## Pre-registered verdicts (written before running)

- ADOPT-SIGNAL: bonsai_shadow beats bonsai_all beyond the band on at least 2 datasets (either regime) without losing beyond the band anywhere, OR it matches cat_select's accuracy pool-wide at half or less of its selection wall time. Then the recommendation is a `bonsai.select` Python module (zero core cost), and this probe is its ceiling measurement.
- DECLINE: bonsai_shadow moves nothing beyond the band anywhere, OR its wins are matched by plain top-k truncation (arm 3), meaning the shadow machinery adds nothing over the importance ranking it already had. The verdict records that selection is not an accuracy lever on this pool, wall-time-and-interpretability-only, and the planned guide chapter carries that honest finding.
- Either way the noise-recovery table (precision/recall on the injected columns) is a deliverable: it is the pedagogical exhibit for the guide chapter that follows.

## REAL-WIDE regime (metric_error, lower better)

| dataset | metric | k / total | bonsai_all | bonsai_shadow | bonsai_topk | cat_all | cat_select | xgb_all | xgb_topk | shadow_vs_all | band | verdict |
|--|--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|
| QSAR-TID-11 | rmse | 306 / 1024 | 0.88287 | 0.89051 | 0.89300 | 0.89455 | 0.89338 | 0.87950 | 0.88247 | -0.00764 | 0.01766 | in band |
| superconductivity | rmse | 67 / 81 | 9.85628 | 9.92407 | 9.91435 | 10.17022 | 10.14472 | 9.83903 | 9.89486 | -0.06779 | 0.19713 | in band |
| spambase | 1-auc | 24 / 57 | 0.01050 | 0.01169 | 0.01197 | 0.01104 | 0.01289 | 0.01747 | 0.01735 | -0.00119 | 0.00100 | **LOSS** |

On real-wide data selection never helps bonsai beyond the band: two datasets are inside the band and spambase is a beyond-band LOSS. Dropping features from an already-real feature set trades a little accuracy for a smaller model. XGBoost with all features is the best learner on both regression sets here (a library difference, not a selection one); its own top-k truncation costs it accuracy just as bonsai's does.

## NOISE-INJECTED regime (metric_error, lower better)

Each set carries injected noise columns equal to its original feature count, so total_features is twice orig_features and bonsai_all is fit on the contaminated matrix.

| dataset | metric | orig / total | k | bonsai_all | bonsai_shadow | bonsai_topk | cat_all | cat_select | xgb_all | xgb_topk | shadow_vs_all | band | verdict |
|--|--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|
| houses | rmse | 8 / 16 | 11 | 0.23289 | 0.23149 | 0.23149 | 0.23012 | 0.22809 | 0.23463 | 0.23119 | +0.00139 | 0.00466 | in band |
| concrete_compressive_strength | rmse | 8 / 16 | 8 | 5.32359 | 4.58744 | 4.58744 | 5.56991 | 4.65285 | 5.45975 | 4.54448 | +0.73615 | 0.10647 | **WIN** |
| wind | rmse | 14 / 28 | 14 | 3.07949 | 3.02061 | 3.02729 | 3.05329 | 2.99762 | 3.11607 | 3.04982 | +0.05888 | 0.06159 | in band |
| breast_cancer | 1-auc | 30 / 60 | 17 | 0.00818 | 0.00692 | 0.00650 | 0.00335 | 0.00797 | 0.01226 | 0.01226 | +0.00126 | 0.00100 | **WIN** |
| pima_diabetes | 1-auc | 8 / 16 | 6 | 0.19104 | 0.20203 | 0.20203 | 0.20704 | 0.19116 | 0.21481 | 0.19361 | -0.01099 | 0.00100 | **LOSS** |
| MagicTelescope | 1-auc | 10 / 20 | 11 | 0.06516 | 0.06344 | 0.06344 | 0.06311 | 0.05787 | 0.06526 | 0.06328 | +0.00172 | 0.00100 | **WIN** |

Injected noise is where selection is supposed to shine, and it does move standings: 3 beyond-band wins (concrete +0.736, breast_cancer +0.00126, MagicTelescope +0.00172). concrete is the exhibit: 8 shuffled columns push bonsai_all from a clean-data neighborhood near 4.59 up to 5.32, and selection recovers all of it. But the wins are not the shadow machinery's: at each of the three the plain top-k-by-gain arm recovers the same accuracy (shadow_vs_topk is +0.00000 on concrete and MagicTelescope, where the two arms pick a bit-identical kept set, and -0.00042 on breast_cancer, inside its band). And the two beyond-band LOSSES are both here-or-adjacent low-dimensional sets (pima -0.01099, spambase -0.00119) where forcing a k below the real feature count discards signal.

## The decomposition: shadow versus plain top-k

The load-bearing column is `shadow_vs_topk`, the shadow machinery's accuracy over plain truncation at the same k.

| dataset | bonsai_shadow | bonsai_topk | shadow_vs_topk | band | shadow beyond-band win vs all | top-k also wins beyond band |
|--|--:|--:|--:|--:|:--:|:--:|
| QSAR-TID-11 | 0.89051 | 0.89300 | +0.00248 | 0.01766 | no | no |
| superconductivity | 9.92407 | 9.91435 | -0.00972 | 0.19713 | no | no |
| spambase | 0.01169 | 0.01197 | +0.00027 | 0.00100 | no | no |
| houses | 0.23149 | 0.23149 | +0.00000 | 0.00466 | no | no |
| concrete_compressive_strength | 4.58744 | 4.58744 | +0.00000 | 0.10647 | yes | yes |
| wind | 3.02061 | 3.02729 | +0.00668 | 0.06159 | no | no |
| breast_cancer | 0.00692 | 0.00650 | -0.00042 | 0.00100 | yes | yes |
| pima_diabetes | 0.20203 | 0.20203 | +0.00000 | 0.00100 | no | no |
| MagicTelescope | 0.06344 | 0.06344 | +0.00000 | 0.00100 | yes | yes |

`shadow_vs_topk` is inside the chance band on all 9 datasets, and on every one of the 3 beyond-band shadow wins the top-k arm wins beyond the band too. Where k lands at or below the real feature count the two arms often select a bit-identical set (the +0.00000 rows) because bonsai's own gain ranking already ranks the real features above the permuted ones, which is the same ranking the shadow threshold reads. The shadow vote adds a data-driven cutoff (how many features to keep) but not a better ranking, and the cutoff it picks is not systematically better than truncating at the same k: pool-mean `shadow_vs_topk` is -0.00232 on real-wide and +0.00104 on noise-injected, both inside any band.

## Noise-recovery table (the guide exhibit)

For the injected regime, treating "drop a noise column" as the positive detection. `noise_kept` is injected columns that survived; `real_dropped` is real columns that were discarded; recall is noise_dropped / injected, precision is noise_dropped / total_dropped.

| dataset | injected | arm | kept | noise_kept | real_dropped | noise recall | noise precision |
|--|--:|--|--:|--:|--:|--:|--:|
| houses | 8 | shadow | 11 | 3 | 0 | 0.625 | 1.000 |
| houses | 8 | cat_select | 11 | 3 | 0 | 0.625 | 1.000 |
| concrete_compressive_strength | 8 | shadow | 8 | 0 | 0 | 1.000 | 1.000 |
| concrete_compressive_strength | 8 | cat_select | 8 | 0 | 0 | 1.000 | 1.000 |
| wind | 14 | shadow | 14 | 1 | 1 | 0.929 | 0.929 |
| wind | 14 | cat_select | 14 | 2 | 2 | 0.857 | 0.857 |
| breast_cancer | 30 | shadow | 17 | 3 | 16 | 0.900 | 0.628 |
| breast_cancer | 30 | cat_select | 17 | 8 | 21 | 0.733 | 0.512 |
| pima_diabetes | 8 | shadow | 6 | 2 | 4 | 0.750 | 0.600 |
| pima_diabetes | 8 | cat_select | 6 | 1 | 3 | 0.875 | 0.700 |
| MagicTelescope | 10 | shadow | 11 | 1 | 0 | 0.900 | 1.000 |
| MagicTelescope | 10 | cat_select | 10 | 1 | 0 | 0.900 | 1.000 |

The exhibit teaches the real lesson: concrete is the clean case (both selectors drop all 8 noise, keep all 8 real, and full accuracy returns), but the moment the real features are correlated or the pool is high-dimensional the recovery is imperfect for both methods. On breast_cancer the shadow arm drops 16 of 30 real features to hold precision, and cat_select drops 21 real while keeping 8 noise; both still net a small accuracy change because breast_cancer's signal is redundant. The shadow method is a comparable noise detector to CatBoost's recursive elimination here, neither dominating, and both are only perfect when the noise is truly independent (concrete) rather than a shuffle of a correlated real column.

## Wall time: bonsai_shadow versus cat_select

Total selection wall (all fits included): the shadow arm counts its 5 shadow fits plus the final refit; cat_select counts the whole `select_features` call including its final refit.

| dataset | bonsai_shadow total (s) | cat_select total (s) |
|--|--:|--:|
| QSAR-TID-11 | 23.6 | 45.8 |
| superconductivity | 36.8 | 23.6 |
| spambase | 3.6 | 5.3 |
| houses | 14.2 | 11.8 |
| concrete_compressive_strength | 3.2 | 3.9 |
| wind | 5.0 | 10.6 |
| breast_cancer | 1.9 | 3.1 |
| pima_diabetes | 1.6 | 1.3 |
| MagicTelescope | 7.8 | 21.5 |
| **pool total** | **97.7** | **126.9** |

The shadow prototype's selection wall is 0.77 of cat_select's pool total (97.7 s versus 126.9 s), cheaper on 6 of 9 but not the half-or-less that the ADOPT-SIGNAL wall-time clause required. The two methods have different cost shapes: the shadow arm scales with rows (5 fits on a 2p-wide matrix, so superconductivity's 11k rows dominate), while cat_select scales with the feature count and step count (its recursive SHAP elimination makes QSAR-TID-11 and MagicTelescope its most expensive cells).

## Verdict: DECLINE, the shadow machinery adds nothing over top-k-by-gain

Of the two pre-registered outcomes, DECLINE fired, on its second clause.

ADOPT-SIGNAL did not fire. Its first branch asked for at least 2 beyond-band wins WITHOUT any beyond-band loss; the probe returned 3 beyond-band wins (concrete, breast_cancer, MagicTelescope) but also 2 beyond-band LOSSES (spambase -0.00119, pima_diabetes -0.01099), so the no-loss condition is not met. Its second branch asked for matching cat_select pool-wide at half or less of its wall time; the shadow arm neither matches cat_select pool-wide (it differs beyond the band on 5 of 9 datasets, better on 2, worse on 3) nor runs at half the wall time (0.77, not 0.5 or less).

DECLINE fired because every beyond-band shadow win is matched by plain top-k truncation at the same k. On all 3 wins `shadow_vs_topk` is inside the band, and on concrete and MagicTelescope the two arms select a bit-identical kept set (`shadow_vs_topk` exactly 0.00000). bonsai's own gain ranking already sorts real features above permuted noise, which is the exact ranking the shadow threshold reads, so the shadow vote contributes a cutoff rule and 6x the compute of a single fit, not a better selection. On the low-dimensional informative sets the shadow cutoff is actively worse than keeping all features (the two beyond-band losses). Selection is not an accuracy lever on this pool: where it recovers accuracy it is recovering from injected junk that a one-line top-k-by-gain already removes, and where the features are all real it costs accuracy.

The consequence is that a `bonsai.select` module, if it ships, is a wall-time-and-interpretability tool (smaller models, a noise-detection report), not an accuracy feature, and the guide chapter that follows carries that honest finding with the noise-recovery table as its exhibit. No bonsai-core change follows: the entire prototype is Python around the shipped `importance("gain")` call, and the measurement says even that Python wrapper does not beat a sort of the same importances.

Reopener: a workload where the beyond-band win survives against top-k at the same k (a `shadow_vs_topk` beyond the band on 2 or more datasets), or a high-noise-fraction regime where the data-driven cutoff the shadow vote provides beats every fixed k a user would guess, would reopen the accuracy case. The interpretability and model-size case is not refuted here and is a separate, non-accuracy decision.

## Costs

Fresh compute, 9 datasets, local CPU, run in three foreground batches (real-wide, then two noise-injected batches) and merged. The full run is dominated by the 6 fits per dataset in the shadow arm and the recursive elimination in cat_select; the two selection methods together are the pool totals in the wall-time table (shadow 97.7 s, cat_select 126.9 s), and the baseline and reference arms (bonsai_all, cat_all, xgb) add the rest. QSAR-TID-11 at 1024 features was the budgeted worst cell and completed in about 86 s of its own, so no seed count was reduced anywhere; the full 5-seed shadow protocol ran on every dataset.

## Deviations, flagged

1. xgboost 3.3.0 was installed into the gauge venv for this probe (the rung-0 and bagging probes recorded xgboost as absent). It provides only the arm-6 ecosystem reference columns; it touches no bonsai/CatBoost arm and no verdict criterion. LightGBM remains absent and is out of scope.
2. The probe runs from a worktree with no local bonsai build; `BONSAI_PYTHON` points at the main checkout's `build-tabarena/python`, and `TABARENA_DIR` at the tabarena checkout, exactly as the sibling probes are invoked. bonsai_all reproduces the rung-0 bonsai baseline on the shared datasets (for example superconductivity 9.85628 against rung-0's 9.8563 and spambase 0.01050 against rung-0's 0.01050), which confirms the same rows and metric.
3. cat_select uses RecursiveByLossFunctionChange with 5 elimination steps and `train_final_model=True` (the final model refits on the chosen set, which is arm 5's "refit on its chosen set"); the algorithm and step count are recorded in the jsonl. No dataset triggered a select_features failure, so the honest-failure branch was not exercised.
4. The noise injection permutes each real column independently with a fixed seed-42 rng and appends the result, so the injected count equals the original feature count and the injected indices are the upper half of the matrix; both are recorded in the jsonl for the recovery grade. The shadow arm's own per-seed shadow copies use seeds 42..46 and are separate from the injected-noise seed.
5. Wall-time pool totals are the sum of the per-dataset cells rounded to one decimal (97.7 and 126.9); the unrounded sums are within 0.1 of these and the 0.77 ratio is unaffected.

## Environment

Local CPU, TabArena-Lite gauge venv (CatBoost 1.2.10, xgboost 3.3.0, scikit-learn 1.7.2), bonsai from `build-tabarena/python`. Run with the gauge venv interpreter, `BONSAI_PYTHON` pointing at a real bonsai build and `TABARENA_DIR` at the tabarena checkout; `PROBE_DATASETS` overrides the pool for smoke runs and `--out` sets the jsonl path.
