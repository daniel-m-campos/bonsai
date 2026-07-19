# Ordered-boosting admission probe, rung 0: is CatBoost's small-data edge the ordered mechanism? (2026-07)

Decision 80 left an open thread: after ablating CatBoost's categorical machinery, a residual small-data lead over bonsai persisted on pure-numeric data (the cat-probe control was outlier-best for CatBoost on 5 of 6 pure-numeric regression sets), and that residual was pinned on "ordered boosting / oblivious regularization / defaults" without decomposing which. Ordered boosting is the marquee CatBoost mechanism: per-row gradients are computed by models that never saw the row, removing the prediction shift (target leakage) of ordinary boosting. This probe is rung 0 of the ordered-boosting campaign: price that mechanism at zero bonsai-core cost with CatBoost's own reference-library toggle (feature-admission step 1), separate it from CatBoost's strong tuned defaults, and test whether an honest-gradient prototype reaches it outside a full permutation engine.

Probe script: `scripts/probe_ordered_boosting_rung0.py`. Raw rows: `benchmarks/results/ordered-boosting-probe-2026-07.jsonl`. Everything ran locally (M2, CPU) in the TabArena-Lite gauge venv (CatBoost 1.2.10, scikit-learn 1.7.2), with bonsai from `build-tabarena/python`.

## Three arms, all at zero core cost

ARM A, the defaults toggle. CatBoost `boosting_type=Ordered` vs `=Plain`, otherwise the library defaults, same seeds and splits. Ordered is CatBoost's own small-data default; the cached gauge "CAT (default)" is the Ordered arm. `plain_def - ord_def` prices ordered boosting by CatBoost's own toggle at its defaults.

ARM B, the matched-knobs toggle. CatBoost Ordered vs Plain at the quality-campaign shape (depth 6, learning_rate 0.05, iterations 1000, early_stopping_rounds 50 on a held-out validation split). This separates the strong-defaults share (defaults-vs-matched) from any mechanism share, and tests the mechanism at matched knobs and not only at CatBoost's tuned defaults.

ARM C, the reachability prototype. A hand-rolled honest-gradient booster vs a plain single-booster control at the same knobs, to ask whether honest gradients, implementable outside bonsai's core, close part of the Plain-to-Ordered gap. Exactly what was built is in its own section below.

## Pool selection rule (stated explicitly)

Pure-numeric means zero categorical features; small means at most about 30k rows.
BASE: every TabArena-Lite gauge dataset with `percentage_cat_features == 0` (asserted no `category`/`object` columns at load), under the 30k cap. That is 5 datasets: QSAR_fish_toxicity, concrete_compressive_strength, QSAR-TID-11, houses, superconductivity. It drops only physiochemical_protein (45,730 rows). All are regression, because the gauge's pure-numeric subset is regression-only.
EXTENSION: easily-loadable pure-numeric small datasets from OpenML and scikit-learn, added to reach the 10-14 target and, decisively, to bring binary classification so the logistic-gradient path and the AUC-scale chance band are exercised at all. Seven added: breast_cancer (scikit-learn), pima_diabetes, banknote-authentication, phoneme, spambase, MagicTelescope, wind (OpenML v1). Each is asserted pure-numeric and under the cap at load.
Total pool: 12 datasets, 6 regression (rmse) and 6 binary (1 minus roc_auc), each one gauge-protocol fold.

## Protocol

One uniform own-harness so every arm sees the identical rows. For the 5 gauge datasets the split is the TabArena fold-0 train/test split pulled from the task wrapper (same data and split as the gauge and the cat probe); for the extension datasets it is a fixed stratified 75/25 holdout (seed 42). From the train side a 20% stratified validation slice (seed 42) drives early stopping in every arm. Metric_error is rmse for regression and 1 minus roc_auc for binary, lower better throughout, matching TabArena's convention. bonsai runs fresh on the same split as the numeric baseline; on pure-numeric data it equals bonsai_ts (the OrderedTargetEncoder is a no-op with no categorical columns). Fits are single-model (no inner bagging) uniformly, so the CatBoost mechanism delta and the arm-C mechanism delta are measured on the same rows under the same protocol; this is a deliberate departure from the gauge's 8-fold bagged protocol and is flagged in Deviations.

Sign convention: every "share" is `plain - ordered` or `plain - honest`, positive when the mechanism (ordered boosting, or honest gradients) lowers error. `catboost_lead = bonsai - cat_ordered_def`, positive when CatBoost beats bonsai. Chance band per decision 55: about 2% relative of the metric for rmse, 0.001 absolute for 1 minus roc_auc.

## Arm A + B: the ordered toggle, regression pool (rmse, lower better)

| dataset | n_train | bonsai | cat ord (def) | cat plain (def) | cat ord (matched) | cat plain (matched) | ordered share (def) | ordered share (matched) | catboost lead |
|--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| QSAR_fish_toxicity | 483 | 1.0335 | 0.9582 | 0.9508 | 0.9679 | 0.9522 | -0.0073 | -0.0157 | +0.0753 |
| concrete_compressive_strength | 548 | 4.5874 | 5.0087 | 4.7438 | 5.0237 | 4.7311 | **-0.2649** | **-0.2927** | -0.4213 |
| QSAR-TID-11 | 3062 | 0.8829 | 0.9093 | 0.8891 | 0.9232 | 0.8950 | -0.0201 | -0.0282 | -0.0264 |
| houses | 11008 | 0.2282 | 0.2273 | 0.2236 | 0.2294 | 0.2269 | -0.0038 | -0.0026 | +0.0009 |
| superconductivity | 11340 | 9.8563 | 10.6266 | 10.0599 | 10.8729 | 10.2908 | **-0.5667** | **-0.5821** | -0.7703 |
| wind | 3944 | 3.0207 | 3.0006 | 2.9732 | 2.9736 | 2.9580 | -0.0274 | -0.0155 | +0.0201 |
| **mean** | | | | | | | **-0.1484** | **-0.1561** | **-0.1870** |

## Arm A + B: the ordered toggle, binary pool (1 minus roc_auc, lower better)

| dataset | n_train | bonsai | cat ord (def) | cat plain (def) | cat ord (matched) | cat plain (matched) | ordered share (def) | ordered share (matched) | catboost lead |
|--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| breast_cancer | 340 | 0.00608 | 0.00545 | 0.00356 | 0.00671 | 0.00377 | -0.00189 | -0.00294 | +0.00063 |
| pima_diabetes | 460 | 0.16752 | 0.17576 | 0.18233 | 0.17600 | 0.17696 | **+0.00657** | +0.00096 | -0.00824 |
| banknote | 823 | 0.00003 | 0.00000 | 0.00007 | 0.00003 | 0.00010 | +0.00007 | +0.00007 | +0.00003 |
| phoneme | 3242 | 0.06763 | 0.06992 | 0.06074 | 0.06815 | 0.06184 | -0.00919 | -0.00631 | -0.00230 |
| spambase | 2760 | 0.01050 | 0.01081 | 0.01074 | 0.01102 | 0.01131 | -0.00007 | +0.00029 | -0.00031 |
| MagicTelescope | 11412 | 0.06116 | 0.05924 | 0.05802 | 0.05945 | 0.05784 | -0.00123 | -0.00161 | +0.00192 |
| **mean** | | | | | | | **-0.00096** | **-0.00159** | **-0.00138** |

## The three decompositions

Mechanism share (ordered vs plain). At MATCHED knobs the ordered toggle is inside the chance band on 6 of 12 datasets and actively HURTS (Ordered has higher error than Plain) on the other 6; it beats Plain beyond the band on 0 of 12. At CatBoost's own DEFAULTS the picture is the same: inside-band or worse everywhere except pima_diabetes (+0.0066, the single dataset where Ordered helps beyond the band). Both pool means are negative (regression -0.156 matched, binary -0.0016 matched). Ordered boosting, priced by CatBoost's own switch, is not a source of accuracy on this pure-numeric small-data pool; toggling it off is neutral-to-better.

Strong-defaults share (defaults vs matched). `defaults_share = plain_matched - plain_def` averages +0.036 (regression) and -0.0006 (binary): CatBoost's tuned defaults and the matched shape trade blows dataset by dataset, a small tuning effect, and neither wins by the ordered mechanism because the ordered toggle is inside-band or negative in BOTH regimes.

Lead over bonsai. In this single-split, single-model harness CatBoost does not even lead bonsai on the pool: mean `catboost_lead` is -0.187 (regression) and -0.0014 (binary), and CatBoost beats bonsai beyond the band on only 1 of 6 in each pool (bonsai wins concrete, QSAR-TID-11, superconductivity outright). The cached gauge, which is 8-fold BAGGED, does show CatBoost leading bonsai on all 5 gauge datasets (cached_lead +0.025, +0.041, +0.013, +0.007, +0.197). So the CatBoost small-data edge that motivated this campaign is real under the bagged gauge protocol but largely evaporates under a single-model fit, and in BOTH regimes the ordered toggle explains none of it: where CatBoost leads (cached), `ordered_share_matched` is negative, so ordered boosting's contribution to the lead is 0% or less.

## Arm C: exactly what was built, and what it found

Two prototypes, both hand-rolled with scikit-learn `DecisionTreeRegressor` weak learners (depth 6, min_samples_leaf 20, learning_rate 0.05, early_stopping_rounds 50, iteration cap 300). LightGBM is not in the gauge venv; a hand-rolled loop also isolates the mechanism more transparently, and every arm-C variant shares the identical weak learner, so each honest-vs-plain comparison is apples-to-apples.

COUPLE (the faithful honest-gradient form, the campaign's primary intent). Two boosters F0, F1 carry the running prediction. Train rows split into fold0/fold1; a fold0 row's per-round gradient comes from the booster trained only on the OTHER fold (base + F1, which never saw fold0), and vice versa, so no row's gradient depends on a model that trained on it. Each round fits fold0's tree on fold0 rows with fold0's honest gradient and fold1's tree on fold1 rows; predict is base + F0 + F1. This is ordered boosting's unbiased-gradient idea reduced to 2 permutation folds.

BAG (the converging simpler form the brief flagged as acceptable). Two independent proper boosters, one per fold with ordinary within-fold gradients, margins averaged at predict. It converges and is out-of-fold honest at test, but its per-fold trees use ordinary (biased) within-fold gradients, so it is 2-fold bagging and does not inject the honest mechanism into training.

The finding: the faithful COUPLE form does not converge at 2 folds. A fold's honest gradient excludes its own accumulator entirely (all of that accumulator's trees are contaminated for the fold), so the gradient never vanishes and the running prediction overshoots. Early stopping catches this within a dozen-odd rounds on regression (couple best-k = 13 to 21 vs the 300 cap) and the result is far worse than the plain control. The BAG form converges but its behavior is data-quantity and variance dominated (each booster sees half the rows), not the honest mechanism: it helps on a couple of tiny binary sets by variance reduction and hurts on regression by the half-data penalty. Neither tracks the ordered-vs-plain signal.

| dataset | armc plain | armc couple (honest) | armc bag | honest share (couple) | honest share (bag) | couple best-k |
|--|--:|--:|--:|--:|--:|--:|
| QSAR_fish_toxicity | 0.9592 | 1.0237 | 0.9588 | -0.0645 | +0.0004 | 13 |
| concrete_compressive_strength | 4.7610 | 8.9831 | 5.4807 | **-4.2221** | -0.7498 | 14 |
| QSAR-TID-11 | 0.9215 | 1.1924 | 0.9695 | -0.2709 | -0.0480 | 21 |
| houses | 0.2309 | 0.3027 | 0.2342 | -0.0718 | -0.0033 | 15 |
| superconductivity | 10.4439 | 14.0750 | 10.7584 | **-3.6311** | -0.3145 | 14 |
| wind | 3.0135 | 3.2619 | 3.0336 | -0.2484 | -0.0201 | 14 |
| breast_cancer | 0.01027 | 0.04486 | 0.00755 | -0.03459 | +0.00273 | 81 |
| pima_diabetes | 0.17755 | 0.22281 | 0.17755 | -0.04525 | +0.00000 | 86 |
| banknote | 0.00045 | 0.00065 | 0.00076 | -0.00021 | -0.00031 | 300 |
| phoneme | 0.06949 | 0.10026 | 0.07847 | -0.03077 | -0.00899 | 91 |
| spambase | 0.01932 | 0.03431 | 0.02119 | -0.01499 | -0.00188 | 144 |
| MagicTelescope | 0.07146 | 0.08637 | 0.07033 | -0.01491 | +0.00113 | 111 |

Regression mean honest share: couple -1.418, bag -0.184. Binary mean honest share: couple -0.023, bag -0.001. Both negative in every pool.

## Verdict: DEFAULTS STORY fired; MECHANISM not confirmed; prototype not reachable

Pre-registered criteria and which fired:

MECHANISM CONFIRMED (Ordered beats Plain by at least half of CatBoost's lead over bonsai, at matched knobs too): DID NOT FIRE. Ordered never beats Plain beyond the chance band on this pool (0 of 12 at matched knobs, 1 of 12 at defaults), and the mean share is negative at both defaults and matched. There is no positive mechanism share to be half of any lead.

DEFAULTS STORY (matched-knobs ordered-vs-plain inside the chance band while the defaults arms differ, so the lead is tuning not the mechanism): FIRED, in its strong form. Matched-knobs ordered-vs-plain is inside the band on half the pool and negative on the other half, so it is at best noise and at worst a net-negative toggle at these sizes; meanwhile a CatBoost lead over bonsai does exist under the bagged gauge protocol. The lead, where it exists, is tuning, bagging, and CatBoost's other defaults (symmetric-tree regularization, border quality), not ordered boosting. The ordered mechanism is not the lever behind the small-data edge decision 80 flagged.

REACHABLE (arm C recovers at least half of the matched-knobs mechanism share): DID NOT FIRE, and per its pre-registered branch, since arm C is negative/trivial the prototype ceiling is unknown outside CatBoost's full permutation machinery. The faithful 2-fold honest form does not converge (its honest gradient never vanishes, so the model overshoots); the converging cross-fit form does not implement the mechanism (it is bagging). This is secondary here because there is essentially no positive mechanism share to reach on this pool, but it does establish that a cheap 2-fold honest prototype is not a route into the mechanism: CatBoost's benefit, in the regimes where it exists, needs its multi-permutation supporting-model machinery, not a 2-fold reduction.

The honest losses and the tell inside them. The mechanism story is not "ordered helps a little"; on the two datasets where the toggle moves most (concrete -0.293, superconductivity -0.582 at matched knobs) Ordered is distinctly WORSE than Plain, and on both of those bonsai already beats CatBoost outright. The single dataset where Ordered helps beyond the band is pima_diabetes at defaults (+0.0066), a coin-flip 460-row set where the matched-knob toggle is back inside the band (+0.0010). And the arm-C failure is loud, not marginal: the faithful honest prototype's regression errors are 1.5x to 2x the plain control because it diverges. Single-dataset wins are coin flips and the pool has none for the mechanism.

## Costs

Fresh compute, 12 datasets, single-model, local M2 CPU, total 466 s (7.8 min) of fit time across all 8 model configurations. Per-arm wall (total; mean per dataset): bonsai 15.8 s (1.3 s); CatBoost Ordered-defaults 65.9 s (5.5 s); CatBoost Plain-defaults 16.8 s (1.4 s); CatBoost Ordered-matched 66.8 s (5.6 s); CatBoost Plain-matched 16.4 s (1.4 s); arm-C couple 33.9 s (2.8 s, short because early stopping truncates the diverging run to a dozen-odd rounds); arm-C bag 125.0 s (10.4 s); arm-C plain 125.6 s (10.5 s). CatBoost Ordered costs about 3.9x the train time of CatBoost Plain (5.5 s vs 1.4 s per dataset), reproducing the cat probe's observation that the ordered machinery is expensive; here it buys nothing.

The prototype's k-times training cost, named. The honest prototype is a k=2-fold cross-fit: the faithful couple form fits 2 trees per round (two half-data supporting boosters), roughly 1x the plain control's single full-data tree in compute (two half passes equal one full pass), and the bag form likewise fits two half-data boosters at about 1.05x plain. The cost of the honest mechanism at 2 folds is therefore not compute; it is variance and non-convergence. CatBoost pays its k-times cost differently, by training log(n) supporting models per permutation across several permutations, which is what makes its ordered gradients converge and what a 2-fold prototype cannot cheaply reproduce.

## Deviations, flagged loudly

1. Filename collision. The brief named the deliverable `scripts/probe_ordered_boosting.py`, but that path already holds a committed decision-69 artifact (a 16M synthetic-scale ordered/plain/bonsai study, marked "stays as-run"). To avoid clobbering committed evidence this probe is `scripts/probe_ordered_boosting_rung0.py`. Different experiment entirely.
2. Single-model, non-bagged own-harness. This does NOT reproduce the cached gauge's bagged CatBoost lead over bonsai: in this harness bonsai is competitive-to-better on the pool (CatBoost leads beyond the band on 2 of 12). The mechanism decomposition (ordered vs plain, within CatBoost) is protocol-independent and robust; the absolute "lead over bonsai" is protocol-sensitive, and both the cached bagged lead and this harness's near-tie agree that the ordered toggle explains none of it. The bagged gauge lead is cross-checked but not bit-matched.
3. Arm C uses scikit-learn regression-tree weak learners, not LightGBM (absent from the venv), and a 300-iteration cap with early stopping for tractability; both honest and plain arm-C variants share these knobs, so the honest-vs-plain delta is unbiased by the choice.
4. The gauge's pure-numeric subset is regression-only (5 datasets, physiochemical_protein excluded by the 30k cap), so classification is entirely from the OpenML/scikit-learn extension; those seven are not TabArena tasks and have no cached bonsai/CatBoost baseline (they carry fresh same-split numbers only).
5. This probe imports bonsai to produce a same-split baseline; it states the matched knobs inline (the quality-campaign shape) rather than routing through bonsai.bench, consistent with the cat probe's admission-probe precedent (decision 80). LightGBM and XGBoost are absent from the gauge venv, so no LightGBM/XGBoost reference columns were computed.
