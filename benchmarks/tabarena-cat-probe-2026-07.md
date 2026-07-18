# TabArena categorical reopener probe: pricing catboost's categorical machinery (2026-07)

Decision 58 declined native categorical splits and shipped `OrderedTargetEncoder` (OTE) instead, with an escape hatch: reopen if crossed-TS preprocessing fails to close the catboost gap AND a workload exists where that gap is load-bearing. The TabArena-Lite gauge (2026-07-15, 51 tasks x 1 fold, M2) is the candidate workload: bonsai_ts Elo 1204 vs catboost 1340, and the earlier diagnosis showed catboost outlier-best on pure-numeric data too, suggesting the lead was mostly ordered-boosting-on-small-data rather than categoricals. This probe decomposes that lead with the sharpest available instrument, catboost's own reference-library toggle (feature-admission step 1): run catboost twice at the gauge's matched budget, (a) native with categorical columns declared, (b) ablated with the identical columns ordinal-encoded as plain integers and nothing declared. The (a) minus (b) delta per dataset IS catboost's categorical machinery, priced by catboost itself.

Probe script: `scripts/probe_tabarena_cat.py`. Raw rows: `results/tabarena-cat-probe-2026-07.jsonl`. Both catboost arms ran locally (M2, CPU, catboost 1.2.10) inside the gauge's TabArena harness at the exact `CatBoost_c1_BAG_L1` protocol: AutoGluon CatBoostModel defaults, 8-fold bagged ensemble, TabArena-Lite fold-0 split, `debug_mode` sequential fitting. The ablation is a single lever, a subclass that casts the already-code-mapped category columns to `int64` after preprocessing so `select_dtypes(include="category")` resolves to empty and `Pool(cat_features=[])`; everything else (config, seeds, folds, splits) is bit-identical between arms. bonsai, bonsai_ts, xgb, lgbm and the cached catboost reference come from the gauge's per-task CSV; only the two fresh catboost arms were computed here. Fresh-native vs cached-catboost agreement on the cat-heavy set: max |delta| 0.0050 (credit-g, 1000 rows), typically under 0.002, so the protocol reproduces.

## The partition

From the committed curated task metadata (which carries `percentage_cat_features` but no cardinalities), define `num_cat_est = round(num_features * percentage_cat_features / 100)`. PURE-NUMERIC control: `num_cat_est == 0`, exactly 6 datasets (all regression). CAT-HEAVY: `num_cat_est >= 3`, which matches 30 of 51; per the pre-registered budget cap this was cut to the 12 with the largest relative cached catboost lead over bonsai_ts ((bts - cat) / |bts|), stated here explicitly: anneal, customer_satisfaction_in_airline, in_vehicle_coupon_recommendation, qsar-biodeg, Amazon_employee_access, Marketing_Campaign, splice, credit-g, kddcup09_appetency, NATICUSdroid, MIC, Bank_Customer_Churn. All 12 happen to be classification (roc_auc error or log_loss), so subset means mix no rmse scales. Datasets with 1-2 token categoricals (15 of 51) sit in neither bucket and were not run.

## The three gaps, cat-heavy subset (lower metric_error is better; shares negative when they favor catboost)

| dataset | metric | rows | cats | cat native | cat ablated | bonsai_ts | categorical share (a-b) | remaining gap (a-bts) | non-cat share (b-bts) |
|--|--|--:|--:|--:|--:|--:|--:|--:|--:|
| Amazon_employee_access | roc_auc | 32769 | 10 | 0.12874 | 0.17650 | 0.15224 | **-0.04776** | -0.02350 | **+0.02426** |
| kddcup09_appetency | roc_auc | 50000 | 39 | 0.15808 | 0.17733 | 0.16963 | **-0.01925** | -0.01155 | **+0.00770** |
| splice | log_loss | 3190 | 61 | 0.09051 | 0.10975 | 0.10538 | **-0.01924** | -0.01486 | **+0.00438** |
| in_vehicle_coupon_recommendation | roc_auc | 12684 | 22 | 0.15334 | 0.16689 | 0.19014 | -0.01354 | -0.03680 | -0.02326 |
| Marketing_Campaign | roc_auc | 2240 | 9 | 0.06701 | 0.07535 | 0.07432 | **-0.00834** | -0.00731 | **+0.00103** |
| credit-g | roc_auc | 1000 | 14 | 0.21226 | 0.22021 | 0.22466 | -0.00795 | -0.01239 | -0.00444 |
| anneal | log_loss | 898 | 33 | 0.05176 | 0.05387 | 0.07004 | -0.00211 | -0.01829 | -0.01617 |
| Bank_Customer_Churn | roc_auc | 10000 | 5 | 0.12294 | 0.12475 | 0.13010 | -0.00181 | -0.00716 | -0.00535 |
| customer_satisfaction_in_airline | roc_auc | 129880 | 17 | 0.00520 | 0.00568 | 0.00653 | -0.00048 | -0.00133 | -0.00084 |
| NATICUSdroid | roc_auc | 7491 | 87 | 0.01404 | 0.01404 | 0.01516 | 0.00000 | -0.00112 | -0.00112 |
| qsar-biodeg | roc_auc | 1054 | 6 | 0.07531 | 0.07523 | 0.08887 | +0.00007 | -0.01356 | -0.01363 |
| MIC | log_loss | 1699 | 95 | 0.46338 | 0.46176 | 0.49134 | +0.00162 | -0.02796 | -0.02958 |
| **mean** | | | | | | | **-0.00990** | **-0.01465** | **-0.00475** |

Aggregate decomposition: catboost's remaining lead over bonsai_ts on the cat-heavy dozen is 0.01465 mean metric_error, of which the categorical machinery is 0.00990 (68%) and the non-categorical residual (ordered boosting, oblivious regularization, defaults) is 0.00475 (32%). Median view: share 0.00503 on a 0.01298 gap (39%). Leave-one-out sensitivity of the mean ratio: 0.47 (dropping Amazon_employee_access) to 0.81 (dropping MIC), so the aggregate criterion clears half on the full pre-registered set and sits at the boundary only if the single largest dataset is excluded. Against the decision-55 chance band (about 0.001): the machinery helps beyond the band on 8/12, is inside the band on 3/12 (NATICUSdroid, qsar-biodeg, customer_satisfaction), and hurts beyond the band on 1/12 (MIC, +0.0016 log_loss).

The honest losses, and the tell inside them: on the 4 datasets where the machinery is priced largest (Amazon, kddcup09, splice, Marketing_Campaign), the non-cat share flips positive, meaning catboost-without-categoricals LOSES to bonsai_ts there. Where categoricals matter most, OTE preprocessing already beats ordinal-fed catboost; what it cannot match is the native machinery itself (per-split ordered target statistics recomputed inside the tree, ctr feature combinations, one-hot mixing). On those 4 the categorical share exceeds the entire remaining gap (ratios 1.1x to 2.0x). Conversely NATICUSdroid's share is exactly zero because its 87 nominal features are binary flags that AutoGluon types as int, so neither arm declared anything and the arms were bit-identical: metadata cat counts overstate the togglable mass.

## The control, pure-numeric subset (rmse; the toggle must be zero by construction)

| dataset | rows | cat native | cat ablated | bonsai_ts | categorical share | remaining gap |
|--|--:|--:|--:|--:|--:|--:|
| QSAR-TID-11 | 5742 | 0.84791 | 0.84791 | 0.86130 | 0.00000 | -0.01338 |
| QSAR_fish_toxicity | 907 | 0.91758 | 0.91758 | 0.94161 | 0.00000 | -0.02403 |
| concrete_compressive_strength | 1030 | 4.54208 | 4.54208 | 4.52072 | 0.00000 | +0.02136 |
| houses | 20640 | 0.21331 | 0.21331 | 0.22026 | 0.00000 | -0.00694 |
| physiochemical_protein | 45730 | 3.55660 | 3.55660 | 3.73243 | 0.00000 | -0.17583 |
| superconductivity | 21263 | 9.26382 | 9.26382 | 9.45516 | 0.00000 | -0.19135 |

The two arms are bit-identical on all 6 (categorical share exactly 0.0), so the instrument has no artifact: whatever it measures on cat-heavy data is the categorical machinery and nothing else. Catboost's lead over bonsai_ts persists on 5/6 here with zero categorical involvement (it loses concrete), which is the ordered-boosting-on-small-data story, unchanged from the gauge diagnosis.

## Verdict: REOPEN fired

Pre-registered criteria: REOPEN if the categorical share on cat-heavy datasets is at least half of catboost's remaining lead over bonsai_ts there and the control shows no comparable artifact; RECONFIRM if the share is small and the lead persists ablated; INCONCLUSIVE only if the ablated arm cannot be produced. The REOPEN criterion fired: the share is 68% of the remaining lead (median-based 39%, leave-one-out 47% to 81%), and the control is exactly zero. The RECONFIRM clause fails on its first conjunct (the share is not small), even though its second conjunct is partially true: an ablated-arm lead does persist on 8 of 12 datasets and on the pure-numeric control.

What the decomposition actually licenses: the escape hatch's factual predicate is now established for the TabArena workload, crossed-TS/OTE preprocessing has not closed the catboost gap on cat-heavy data, and the majority of that gap is categorical machinery priced by catboost's own toggle, concentrated in high-cardinality datasets (Amazon 8 high-card cols, kddcup09, splice 60 DNA positions, coupon recommendation). It does not promise that native splits close the whole TabArena Elo gap: roughly a third of the cat-heavy lead plus essentially all of the pure-numeric lead belongs to the ordered-boosting/small-data campaign, which reopening categoricals will not touch. Whether TabArena is load-bearing enough to spend the complexity is the launch-strategy call (issue #157, crown path), not this probe's.

## Budget, protocol notes, deviations

Fresh compute: 18 datasets x 2 arms x 8 bag folds, all local M2 CPU, wall 10462 s (2.9 h) end to end, catboost train time 8399 s native vs 2010 s ablated (the machinery also costs 4.2x train time at these sizes). Reused from the gauge cache: all bonsai, bonsai_ts, xgb, lgbm rows and the catboost reference column (51 tasks). Deviations to flag: (1) catboost 1.2.10 was pip-installed into the gauge venv for this probe (the gauge itself had used cached leaderboard baselines and never ran catboost locally); (2) the cached gauge numbers were produced via Ray backend, the fresh arms via the native in-process backend, same seeds and splits, reproduction checked above; (3) single fold-0 throughout, per the gauge's lite protocol, so per-dataset deltas inside the 0.001 band are noise and only the aggregate is read; (4) this probe never imports bonsai, so the bonsai.bench knob-provenance rule does not apply, the matched budget is TabArena's own c1 default by construction.
