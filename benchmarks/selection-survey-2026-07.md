# Feature-selection method survey: nine methods, one budget ladder, one honest judge (2026-07)

This is a survey measurement for guide chapter 14, not an admission probe: it does not open or close any feature decision (decision 86 already settled that selection ships as a recipe, not an API). The question here is comparative: given that a feature budget binds, WHICH selection method earns its cost? Every method produces a feature ranking; every ranking is evaluated the same way, by refitting bonsai at matched knobs on the ranking's top-k for each rung of a budget ladder and reading error on an untouched holdout. Curve differences are therefore ranking-quality differences, and each method also carries its measured selection cost.

Survey script: `scripts/probe_selection_survey.py`. Raw rows: `benchmarks/results/selection-survey-2026-07.jsonl`. Splits are exported from the rung-0 loader (`scripts/probe_ordered_boosting_rung0.py`, imported read-only) so the baselines are directly comparable to the committed probes; the run mode needs only bonsai, numpy and scikit-learn. Everything ran locally (M2, CPU); the bonsai build is `build-tabarena/python`.

## Datasets

superconductivity (11,340 train rows, 81 features, regression) is the worked example: real, physically meaningful features with heavy built-in redundancy (min/max/mean/weighted-variant families of element properties), the regime where method differences should show. QSAR-TID-11 (3,062 train rows, 1,024 fingerprint features) is the wide-short footnote. Splits are the gauge fold-0 with a 20% stratified validation slice (seed 42), identical to the sibling probes; selection touches only train and validation rows; the holdout is untouched by every method.

## Protocol

Matched knobs throughout: depthwise, depth 6, learning_rate 0.05, iterations cap 1000, early_stopping_rounds 50, min_data_in_leaf 20, lambda_l2 1.0, max_bin 255. Ties are declared by a noise floor, the decision-55 chance band: refit-to-refit luck at these sizes is about 2% relative rmse (superconductivity 0.197, QSAR 0.0177), so differences smaller than that are reported as ties and only larger gaps as real. Harness continuity: both all-features baselines reproduce the committed feature-selection-probe numbers exactly (9.85628 and 0.88287).

Methods (each yields a full ranking, best first):

| family | method | mechanics | selection cost |
|---|---|---|---|
| filter | corr | abs Pearson vs target, train rows | ~0 s |
| filter | mutual_info | sklearn mutual_info_regression, train rows | 2.0 s |
| embedded | gain | bonsai `importance("gain")` from the all-features fit | one fit (4.9 s) |
| embedded | split | `importance("split")`, same fit | 0 extra |
| embedded | shap_train | mean abs `pred_contribs` on train rows | + predict pass |
| embedded | shap_val | mean abs `pred_contribs` on validation rows | + predict pass |
| validation | perm_val | permutation importance on validation, 5 repeats | p x 5 predicts |
| wrapper | rfe_gain | backward recursive elimination down the ladder rungs, gain recomputed each rung; drop order = ranking | one fit per rung |
| wrapper | forward | greedy forward selection to k=24, candidate fits at 200 iterations in 4 parallel workers, chosen prefixes refit at matched knobs | ~1,660 cheap fits |

forward is omitted on QSAR-TID-11: its candidate count scales with the feature count (about 24,000 fits at p=1024), and the point it makes is already made at p=81.

## superconductivity: the curves (holdout rmse, baseline 9.85628; noise floor 0.197, so cells at or under 10.053 tie the baseline)

| method | wall (s) | k=64 | k=48 | k=32 | k=24 | k=16 | k=12 | k=8 | k=4 |
|--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| corr | 0.0 | 9.934 | 10.095 | 10.021 | 10.250 | 10.451 | 10.698 | 11.449 | 13.297 |
| mutual_info | 2.0 | 9.874 | 9.947 | 10.134 | 10.217 | 11.004 | 11.727 | 11.794 | 13.921 |
| gain | 4.9 | 9.880 | 10.026 | 9.965 | 10.099 | 10.157 | 10.404 | 10.749 | 12.773 |
| split | 4.9 | 9.970 | 10.014 | 10.427 | 10.526 | 10.864 | 11.073 | 11.386 | 12.553 |
| shap_train | 12.6 | 9.856 | 9.950 | 10.083 | 10.093 | 10.183 | 10.420 | 10.697 | 13.709 |
| shap_val | 6.9 | 9.856 | 9.950 | 9.996 | 10.093 | 10.183 | 10.420 | 10.697 | 13.709 |
| perm_val | 21.8 | 9.944 | 10.012 | 9.988 | 10.053 | 10.267 | 10.409 | 10.994 | 12.878 |
| rfe_gain | 33.0 | 9.880 | 9.946 | 9.954 | 10.084 | 10.255 | 10.500 | 10.743 | 12.671 |
| forward | 238.8 | - | - | - | 10.076 | 10.131 | 10.168 | 10.428 | 11.416 |

Readings, loose budgets first:

1. **No method beats the all-features baseline anywhere.** Every cell is above 9.856; selection on this real, non-contaminated dataset is a size lever, not an error lever. This restates decision 86's verdict from a nine-method angle without touching it.
2. **At loose budgets (k >= 32) the model-based rankings are interchangeable and even filters are close.** gain, shap, perm, rfe all tie the baseline down to k=32 (a free 2.5x reduction); corr ties it at 64 and 32 for literally zero selection cost. Paying more than one fit buys nothing here.
3. **At tight budgets the ranking family separates from the filters.** At k=16: gain 10.157 vs mutual_info 11.004, a gap of 4.3 times the noise floor. split-count is reliably the worst embedded ranking (its cardinality bias, chapter 8), landing nearer the filters than its gain sibling.
4. **shap_train and shap_val are the same ranking here** (identical cells except one rung): with 11k clean rows, held-out attribution does not change the order. Suspiciously equal, so it was checked: the two importance vectors are genuinely different (different input matrices, max element gap 0.13, both contribution matrices pass the sum-to-prediction identity) and their rankings first diverge at position 21, but the top-k SETS coincide at every rung except k=32 (symmetric difference 2), and an identical set refits deterministically to the identical rmse. The one differing cell is the k=32 pair. rfe matches gain (recursion has nothing to repair at these k). perm matches gain at 4.5x the wall.
5. **forward selection beats every ranking at k <= 12 by more than the noise floor** (10.168 vs 10.404 at k=12, a 0.24 gap against the 0.197 floor; 10.428 vs 10.697 at k=8; 11.416 vs 12.553 at k=4, a 1.14 gap, nearly six floors). It is the only method that evaluates SETS rather than individual features, so it is the only one that picks complements instead of redundant twins. It pays 239 s, roughly 50x the gain ranking, for that power.

## The redundancy diagnostic

Feature-set overlap with gain's top-16 (Jaccard) and the top-5 features by method:

| method | Jaccard vs gain@16 | top-5 flavor |
|--|--:|--|
| gain / shap / perm / rfe | 1.00 / 0.88 / 0.78 / 0.78 | ThermalConductivity + atomic_radius + ElectronAffinity variants |
| corr | 0.19 | THREE ThermalConductivity variants of the same cluster in the top 5 |
| mutual_info | 0.14 | fie/ElectronAffinity variants, again same-cluster heavy |
| split | 0.19 | entropy/std variants (many-bins features, the cardinality bias) |
| forward | 0.28 | five DIFFERENT property families in five picks |

The two numbers that matter: corr's 0.19 (filters score features one at a time, so a strong cluster floods their top ranks with copies) and forward's 0.28 (it disagrees with gain MORE than the filters agree with each other, yet wins tight budgets, because its picks are complements). Bootstrap stability of the gain ranking (10 resamples): mean top-16 selection frequency 0.90, with a core of 8 features at 1.0, so the gain ranking itself is stable here and forward's advantage is genuinely about set composition, not ranking noise.

## QSAR-TID-11: the wide-short footnote (baseline 0.88287; noise floor 0.0177, so cells at or under 0.90053 tie the baseline)

| method | wall (s) | k=512 | k=256 | k=128 | k=64 | k=32 |
|--|--:|--:|--:|--:|--:|--:|
| corr | 0.0 | 0.892 | 0.927 | 0.965 | 0.996 | 1.122 |
| mutual_info | 7.5 | 0.889 | 0.896 | 0.941 | 1.011 | 1.149 |
| gain | 3.4 | 0.895 | 0.890 | 0.918 | 0.973 | 1.081 |
| split | 3.4 | 0.890 | 0.912 | 0.968 | 1.023 | 1.197 |
| shap_train | 4.7 | 0.886 | 0.892 | 0.907 | 0.947 | 1.056 |
| shap_val | 3.7 | 0.887 | 0.889 | 0.904 | 0.944 | 1.063 |
| perm_val | 164.4 | 0.889 | 0.890 | 0.918 | 0.966 | 1.043 |
| rfe_gain | 13.2 | 0.895 | 0.895 | 0.922 | 0.954 | 1.049 |

At 1,024 features the rankings finally separate where they were interchangeable at 81: **shap_val is the best ranking at k=256 (a tie with the baseline, so a free 4x reduction), k=128 and k=64**, ahead of raw gain by 1.4 to 2.9 times the noise floor at the tight end; perm_val takes the k=32 cell but pays 164 s (5,120 shuffled predict passes) against shap_val's 3.7 s. The mechanism: on wide-short data the gain lottery (many features, few rows) inflates weak features' training gain, and attribution measured on rows the fit did not memorize discounts them. rfe's first rung reproduces gain's k=512 cell exactly (structural identity, the ranking-reconstruction check) and its recursion starts paying below k=128.

## What the survey recommends (the guide chapter's summary)

- Budget loose (keep >= 40% of features): any model-based ranking, or even corr, all tie the baseline; use the free one.
- Budget tight, p moderate: forward selection if its wall is affordable, else gain top-k and accept the measured gap.
- Budget tight, p large (wide-short): shap_val top-k, the best accuracy-per-second in the survey; perm_val only if its wall is nothing to you.
- Never split-count, never mutual_info for tight budgets on redundant data; both are dominated everywhere it matters.
- Nothing here beats all features on either dataset; the budget, not accuracy hope, is the reason to select. This is consistent with decision 86 and adds the between-methods hierarchy that decision deliberately did not rank.

## Deviations, flagged

1. forward's candidate fits use 200 iterations with early stopping 30 for the greedy search (the chosen prefixes are refit at full matched knobs for every reported number); the search shortcut is recorded here and its ranking is still the best tight-budget performer, so the shortcut did not cost it the verdict.
2. forward's candidate scoring reads validation error, the same slice that drives early stopping; standard greedy practice, and the untouched holdout grades the final sets.
3. An RFE ranking-reconstruction bug in the first run (dropped batches concatenated in the wrong order) was caught by the structural identity check (rfe top-512 must equal gain top-512 on QSAR because the first rung's survivors are exactly gain's top-512, and the k=512 cells must match; they read 0.945 vs 0.895) and fixed; both rfe arms were rerun with the corrected reconstruction. The committed jsonl contains only corrected rows, and the identity now holds exactly (0.89518 both).
4. This survey supersedes the guide14 budget-curve addendum in `benchmarks/feature-selection-probe-2026-07.md`; the gain arm here reproduces those rows digit-for-digit at every shared k.

## Environment

Local M2, CPU. Gauge-venv-exported npz splits; run mode used the repo python with `PYTHONPATH=build-tabarena/python` (bonsai + numpy + scikit-learn 1.7.2 only). Selection walls quoted are single-machine, 4 forward workers, and comparable within the table, not across machines.
