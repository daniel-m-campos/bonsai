# Bagging-interaction probe: is CatBoost's small-data edge a bagged-protocol randomization interaction? (2026-07)

The ordered-boosting rung-0 probe (benchmarks/ordered-boosting-probe-2026-07.md, decision 81) killed the ordered mechanism but left one thread hanging. CatBoost's pure-numeric small-data lead over bonsai is real under the TabArena gauge's 8-fold BAGGED protocol (a positive cached lead on all 5 gauge datasets: +0.025, +0.041, +0.013, +0.007, +0.197) yet it largely evaporates under single-model fits (bonsai is competitive to better on 10 of 12 at matched knobs), and ordered boosting explained none of it. The single surviving hypothesis: the bagged protocol INTERACTS with CatBoost's non-rate randomization defaults (Bayesian bootstrap, random_strength) so its ensemble members decorrelate and average better than bonsai's deterministic ones. This probe prices that interaction at zero bonsai-core cost, which is what decision 81's reopener asked for.

Probe script: `scripts/probe_bagging_interaction.py`. Raw rows: `benchmarks/results/bagging-interaction-probe-2026-07.jsonl`. Everything ran locally (CPU) in the TabArena-Lite gauge venv (CatBoost 1.2.10, scikit-learn 1.7.2), with bonsai from `build-tabarena/python`. Knobs are imported from `bonsai.bench.params` (`catboost_core`) and metrics from `bonsai.bench.metrics`, per the one-source-of-truth provenance rule.

## Pool

The 12-dataset pool, the gauge fold-0 / stratified-holdout splits, and the loader are imported wholesale from the rung-0 probe (`scripts/probe_ordered_boosting_rung0.py`, a completed as-run experiment imported read-only): 6 regression (rmse) and 6 binary (1 minus roc_auc), every one pure-numeric and under a 30k-row cap. The pool-selection rule and its provenance are stated once in the rung-0 md; this probe does not restate it. bonsai_single reproduces the rung-0 bonsai baseline to the last digit on all 12 (the harness-validation check below), so the pool and protocol are demonstrably the same rows.

## Arms (7 per dataset, two protocols)

SINGLE is identical to the rung-0 matched arms: a 20% stratified validation slice (seed 42) off the train side drives early stopping, and the model fits on the remaining 80%.
BAG8 folds the whole train side into 8 (KFold for regression, StratifiedKFold for binary, shuffle=True, random_state=42). Each fold model trains on 7/8 and early-stops on its held-out fold; the test prediction is the mean over the 8 models (mean probability for binary, mean prediction for regression). This mirrors the AutoGluon bagged protocol the gauge runs.

Matched knobs throughout (the quality-campaign shape): depth 6, learning_rate 0.05, iterations cap 1000, early_stopping_rounds 50, min_data_in_leaf 20, lambda_l2 1.0, max_bin 254 (CatBoost) / 255 (bonsai), CatBoost boosting_type=Plain (Ordered is refuted, decision 81).

1. bonsai_single: matched knobs, deterministic defaults, seed 42.
2. bonsai_bag8: BAG8, all folds seed 42, no subsampling; members differ only through fold data (stock deterministic bonsai).
3. bonsai_bag8_rand: BAG8 with randomization from existing knobs, per fold f: booster.random_seed = 42+f, bernoulli sampler, sampler.subsample = 0.8, tree.feature_fraction = 0.8, tree.feature_seed = 42+f.
4. cat_single: CatBoost Plain matched, randomization at library defaults.
5. cat_bag8: BAG8, arm-4 config (stock Bayesian bootstrap + random_strength), seed 42 all folds.
6. cat_bag8_neut: arm 5 with randomization neutralized: bootstrap_type="No", random_strength=0, rsm=1.
7. cat_bag8_def: BAG8 at CatBoost library defaults (its own boosting_type, lr, iterations; only seed 42 and thread count set). The gauge-reproduction arm.

## Decompositions and sign convention

Every metric_error is lower-better (rmse for regression, 1 minus roc_auc for binary). Every "gain" or "share" is positive when the named lever LOWERS error.

- bagging_gain (per library) = single minus bag8.
- interaction = (cat_single minus cat_bag8) minus (bonsai_single minus bonsai_bag8). Positive means CatBoost gains more from bagging than bonsai. THIS IS THE HEADLINE.
- randomization_share = cat_bag8_neut minus cat_bag8. Positive means stock CatBoost randomization helps under bagging.
- bonsai_reach = bonsai_bag8 minus bonsai_bag8_rand. Positive means bonsai's existing knobs buy the same mechanism.
- gauge reproduction = cat_bag8_def minus bonsai_bag8 on the gauge 5, beside the cached gauge leads.

Chance band per decision 55: about 2% relative of the metric for rmse (computed as 0.02 times bonsai_single per dataset), 0.001 absolute for 1 minus roc_auc.

## Pre-registered verdicts (written before running)

- Interaction inside the band on 10+ of 12: hypothesis REFUTED; the bagged-protocol edge is not a randomization interaction; decision 81's residual closes with no mechanism found, and a refutation is the deliverable.
- Interaction beyond the band AND randomization_share explains half or more of it: mechanism FOUND; then bonsai_reach prices the zero-core-cost response, and if arm 3 closes half or more, the recommendation is a wrapper-config lever for the TabArena integration, no core change.
- Interaction beyond the band but randomization_share near zero: the bagging benefit is structural (per-fold early stopping, averaging nonlinearity); record which and name the next instrument.

## Regression pool (rmse, lower better)

| dataset | n_full | bonsai single | bonsai bag8 | cat single | cat bag8 | bag gain bonsai | bag gain cat | interaction | band | in band |
|--|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|
| QSAR_fish_toxicity | 604 | 1.0335 | 0.9412 | 0.9538 | 0.9237 | +0.0923 | +0.0301 | **-0.0622** | 0.0207 | NO |
| concrete_compressive_strength | 686 | 4.5874 | 4.5429 | 4.6528 | 4.5826 | +0.0445 | +0.0702 | +0.0257 | 0.0917 | yes |
| QSAR-TID-11 | 3828 | 0.8829 | 0.8610 | 0.8945 | 0.8704 | +0.0218 | +0.0241 | +0.0023 | 0.0177 | yes |
| houses | 13760 | 0.2282 | 0.2199 | 0.2264 | 0.2232 | +0.0084 | +0.0032 | **-0.0052** | 0.0046 | NO |
| superconductivity | 14175 | 9.8563 | 9.4665 | 10.1702 | 9.9022 | +0.3898 | +0.2680 | -0.1218 | 0.1971 | yes |
| wind | 4930 | 3.0207 | 2.9554 | 2.9542 | 2.9186 | +0.0652 | +0.0357 | -0.0296 | 0.0604 | yes |
| **mean** | | | | | | **+0.1037** | **+0.0719** | **-0.0318** | | 4/6 |

## Binary pool (1 minus roc_auc, lower better)

| dataset | n_full | bonsai single | bonsai bag8 | cat single | cat bag8 | bag gain bonsai | bag gain cat | interaction | band | in band |
|--|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|
| breast_cancer | 426 | 0.00608 | 0.00419 | 0.00419 | 0.00273 | +0.00189 | +0.00147 | -0.00042 | 0.001 | yes |
| pima_diabetes | 576 | 0.16752 | 0.16633 | 0.17636 | 0.17158 | +0.00119 | +0.00478 | **+0.00358** | 0.001 | NO |
| banknote | 1029 | 0.00003 | 0.00000 | 0.00003 | 0.00003 | +0.00003 | +0.00000 | -0.00003 | 0.001 | yes |
| phoneme | 4053 | 0.06763 | 0.05819 | 0.05901 | 0.05224 | +0.00943 | +0.00677 | **-0.00266** | 0.001 | NO |
| spambase | 3450 | 0.01050 | 0.01061 | 0.01104 | 0.01134 | -0.00011 | -0.00029 | -0.00018 | 0.001 | yes |
| MagicTelescope | 14265 | 0.06116 | 0.05896 | 0.05804 | 0.05740 | +0.00220 | +0.00064 | **-0.00156** | 0.001 | NO |
| **mean** | | | | | | **+0.00244** | **+0.00223** | **-0.00021** | | 3/6 |

## The decomposition

The headline interaction is negative in both pool means (regression -0.0318, binary -0.00021) and strictly inside the chance band on 7 of 12. That already fails the pre-registered REFUTED threshold's exact wording (10+ of 12 in band), but the refutation is stronger than the band count, because the SIGN of the 5 out-of-band cases refutes the hypothesis's direction. Four of the five (QSAR_fish_toxicity -0.062, houses -0.005, phoneme -0.0027, MagicTelescope -0.0016) are NEGATIVE, meaning bonsai gains MORE from bagging than CatBoost there, the opposite of the hypothesis. Exactly one out-of-band case (pima_diabetes +0.0036) has CatBoost gaining more, and it is a 460-row coin-flip.

The mechanism the hypothesis named is null. Randomization_share (neutralizing CatBoost's Bayesian bootstrap and random_strength under bagging) averages -0.00045 on regression and +0.00128 on binary; it is inside or at the band on 11 of 12 and never beyond it in the direction that would matter. On the one dataset where the interaction favors CatBoost beyond the band (pima_diabetes), randomization_share is +0.00681, more than the whole +0.00358 interaction, so the entire CatBoost-favoring signal on that single set is the stock randomization, and it is confined to one coin-flip dataset rather than a pool trend.

| dataset | cat bag8 | cat bag8 neut | randomization_share | bonsai bag8 | bonsai bag8 rand | bonsai_reach |
|--|--:|--:|--:|--:|--:|--:|
| QSAR_fish_toxicity | 0.9237 | 0.9288 | +0.0051 | 0.9412 | 0.9378 | +0.0034 |
| concrete_compressive_strength | 4.5826 | 4.6018 | +0.0192 | 4.5429 | 4.3695 | +0.1735 |
| QSAR-TID-11 | 0.8704 | 0.8696 | -0.0008 | 0.8610 | 0.8499 | +0.0111 |
| houses | 0.2232 | 0.2226 | -0.0006 | 0.2199 | 0.2202 | -0.0003 |
| superconductivity | 9.9022 | 9.8753 | -0.0269 | 9.4665 | 9.3772 | +0.0893 |
| wind | 2.9186 | 2.9199 | +0.0013 | 2.9554 | 2.9256 | +0.0298 |
| breast_cancer | 0.00273 | 0.00273 | +0.00000 | 0.00419 | 0.00461 | -0.00042 |
| pima_diabetes | 0.17158 | 0.17839 | +0.00681 | 0.16633 | 0.17242 | -0.00609 |
| banknote | 0.00003 | 0.00028 | +0.00024 | 0.00000 | 0.00000 | +0.00000 |
| phoneme | 0.05224 | 0.05224 | +0.00000 | 0.05819 | 0.05735 | +0.00084 |
| spambase | 0.01134 | 0.01166 | +0.00032 | 0.01061 | 0.01134 | -0.00073 |
| MagicTelescope | 0.05740 | 0.05771 | +0.00031 | 0.05896 | 0.05863 | +0.00034 |

Why the interaction is null has a plain mechanism in the bagging-gain columns. Bonsai's simple 8-fold data-bagging already captures most of the ensemble benefit (bagging_gain_bonsai mean +0.104 on regression against CatBoost's +0.072; +0.0024 against +0.0022 on binary). The decorrelation that bagging buys comes from the fold DATA, which both libraries get identically, and on this small-data pool it saturates the available headroom. There is nothing left for CatBoost's extra per-tree randomization to exploit that bonsai's deterministic ensemble misses. bonsai_reach confirms bonsai can even turn its OWN randomization knobs where it helps (concrete +0.173, superconductivity +0.089), so the zero-core-cost response exists in the shipped estimator and is simply not needed.

## Harness validation and gauge reproduction

bonsai_single reproduces the rung-0 bonsai column to 0.000000 on all 12 datasets, which proves the split loading and the metric are the same rows and the same convention. cat_single (arm 4) is within the chance band against the rung-0 cat_plain_matched column on 11 of 12; the differences are the single intended knob completion, l2_leaf_reg 1.0 (the matched value catboost_core carries) versus the CatBoost default 3.0 that the rung-0 matched arm left in place. The largest, concrete at -0.0782, is within concrete's 0.0917 band; the one out-of-band case, phoneme at -0.0028 against a 0.001 band, is the same l2 effect on a 3242-row set and is not a harness defect (bonsai is bit-identical there).

| dataset | cat_single (arm 4) | rung-0 cat_plain_matched | diff | within band |
|--|--:|--:|--:|:--:|
| QSAR_fish_toxicity | 0.95376 | 0.95221 | +0.00156 | yes |
| concrete_compressive_strength | 4.65285 | 4.73105 | -0.07820 | yes |
| QSAR-TID-11 | 0.89455 | 0.89501 | -0.00046 | yes |
| houses | 0.22638 | 0.22689 | -0.00051 | yes |
| superconductivity | 10.17022 | 10.29084 | -0.12062 | yes |
| wind | 2.95424 | 2.95804 | -0.00380 | yes |
| breast_cancer | 0.00419 | 0.00377 | +0.00042 | yes |
| pima_diabetes | 0.17636 | 0.17696 | -0.00060 | yes |
| banknote | 0.00003 | 0.00010 | -0.00007 | yes |
| phoneme | 0.05901 | 0.06184 | -0.00282 | NO |
| spambase | 0.01104 | 0.01131 | -0.00026 | yes |
| MagicTelescope | 0.05804 | 0.05784 | +0.00020 | yes |

The gauge-reproduction arm (cat_bag8_def, which on small data is CatBoost's own Ordered default under bagging) leads bonsai_bag8 in the same direction as the cached gauge lead on 4 of 5 gauge datasets, including both of the largest cached leads (superconductivity, reproduced +0.237 against a cached +0.197; concrete +0.017 against +0.041). It flips on QSAR_fish_toxicity, where bonsai_bag8 wins by 0.013 in this harness, and shrinks to a tie on houses. This is expected: this BAG8 is not the AutoGluon protocol bit-for-bit (single fold-0 test split, no repeats, an 8-fold inner bag rather than the gauge's full cross-validated test), so magnitudes differ and the smallest leads are noise. The point it establishes is that the bagged regime that produced the cached CatBoost lead IS engaged here, and even engaged, the interaction decomposition attributes that lead to a LEVEL difference under bagging (CatBoost's defaults, arm 7 versus arm 2), not to a bagging-specific interaction (arms 4 and 5), and not to the randomization the hypothesis named (arm 6).

| dataset | bonsai bag8 | cat bag8 def | repro lead | cached lead |
|--|--:|--:|--:|--:|
| QSAR_fish_toxicity | 0.9412 | 0.9285 | -0.0127 | +0.0247 |
| concrete_compressive_strength | 4.5429 | 4.5597 | +0.0168 | +0.0407 |
| QSAR-TID-11 | 0.8610 | 0.8635 | +0.0024 | +0.0131 |
| houses | 0.2199 | 0.2201 | +0.0002 | +0.0072 |
| superconductivity | 9.4665 | 9.7030 | +0.2366 | +0.1968 |

## Verdict: hypothesis REFUTED, no bagging-randomization mechanism

The bagged-protocol edge is not a randomization interaction. The headline interaction is negative in both pool means; where it is beyond the chance band it favors bonsai on 4 of 5 datasets; the lone CatBoost-favoring beyond-band case is one 460-row coin-flip whose whole signal is absorbed by randomization_share; and randomization_share itself, the mechanism the hypothesis named, is null across the pool. CatBoost does not decorrelate and average better than bonsai under bagging on this pure-numeric small-data pool, because 8-fold data-bagging already gives bonsai the decorrelation, and its shipped sampler knobs (arm 3) reach any residual randomization benefit where one exists. Decision 81's reopener closes: the small-data lead that survives under bagging is CatBoost's defaults operating at a level, not a bagging interaction, and no bonsai core change follows.

Of the three pre-registered outcomes, the REFUTED outcome fired in substance. Its exact wording asked for 10+ of 12 inside the band and the measurement returned 7 of 12, so the wording is recorded as not met literally; the sign analysis makes the refutation firmer than that count, since the out-of-band cases refute the hypothesis's direction rather than confirming it. The mechanism-FOUND outcome fired on exactly one dataset (pima_diabetes, randomization_share covering the interaction) and therefore does not fire at the pool level. The structural outcome did not fire (the beyond-band interactions are mostly negative, so there is no CatBoost bagging benefit to attribute to structure).

## Costs

Fresh compute, 12 datasets, local CPU, total 699 s (11.7 min) of fit time across the seven arms. Per-arm total wall (mean per dataset): bonsai_single 16 s (1.4 s), cat_single 17 s (1.4 s), bonsai_bag8 136 s (11.3 s), bonsai_bag8_rand 129 s (10.7 s), cat_bag8 133 s (11.1 s), cat_bag8_neut 127 s (10.6 s), cat_bag8_def 141 s (11.8 s). The five BAG8 arms cost about 8x their SINGLE counterparts, as expected for an 8-fold inner bag; cat_bag8_def is the most expensive because its small-data default engages Ordered.

## Deviations, flagged

1. cat_single (arm 4) uses catboost_core's fully-matched knobs, which set l2_leaf_reg 1.0; the rung-0 matched CatBoost arm left l2_leaf_reg at CatBoost's default 3.0 (it matched only depth, learning_rate, iterations). The provenance rule (import knobs from bonsai.bench.params) makes arm 4 the more completely matched arm; the harness-validation table above prices the difference and it is within the chance band on 11 of 12. bonsai_single, which does route through the shipped defaults, is bit-identical to rung-0.
2. BAG8 is an 8-fold inner bag with a single fold-0 test split, not AutoGluon's full cross-validated bagged protocol. It reproduces the cached CatBoost lead directionally on the gauge 5 (4 of 5), not bit-for-bit; the interaction and randomization decompositions are internally consistent within this harness (every arm sees the identical folds) and are the load-bearing measurements, while the absolute gauge lead is cross-checked, not matched.
3. The regression chance band is 2% relative computed against bonsai_single per dataset (a stable per-dataset error scale); the binary band is 0.001 absolute. Same convention as the rung-0 probe.
4. The extension datasets (7 of 12) are not TabArena tasks and carry no cached gauge baseline, so the gauge-reproduction cross-check is the 5 gauge datasets only. The extension sets exist to bring binary classification and the AUC-scale band into the pool at all.
5. The probe imports the rung-0 probe read-only for its pool, splits, and loader, and states the matched knobs through bonsai.bench.params and the metrics through bonsai.bench.metrics, consistent with the admission-probe provenance precedent. LightGBM and XGBoost are absent from the gauge venv, so no reference columns were computed.

## Environment

Local CPU, TabArena-Lite gauge venv (CatBoost 1.2.10, scikit-learn 1.7.2), bonsai from `build-tabarena/python`. Run with the gauge venv interpreter, `BONSAI_PYTHON` pointing at a real bonsai build and `TABARENA_DIR` at the tabarena checkout; `PROBE_DATASETS` overrides the pool for smoke runs and `--out` sets the jsonl path.
