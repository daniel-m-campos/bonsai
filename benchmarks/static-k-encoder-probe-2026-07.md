# Static-K encoder probe: permutation-averaged ordered TS vs CatBoost's categorical share (2026-07)

Rung 1 of the categorical reopener. Decision 80 priced CatBoost's categorical machinery at a mean -0.0099 metric_error on the 12-dataset cat-heavy TabArena pool, 68% of its remaining lead over bonsai_ts. CatBoost's ordered target statistics involve multiple random permutations; bonsai's `OrderedTargetEncoder` (OTE) uses one ordering. This probe asks the cheapest question first (feature-admission step 1): how much of that share does static K-permutation averaging recover, as plain preprocessing, before anyone builds doc 17's engine feature?

Probe script: `scripts/probe_static_k_encoder.py`. Raw rows: `results/static-k-encoder-probe-2026-07.jsonl` (12 rows). Reference metrics (cat_native, cat_ablated, cached bonsai_ts) come from decision 80's committed pool, `results/tabarena-cat-probe-2026-07.jsonl`; only the three bonsai arms were computed fresh here.

## Method

Protocol matched to decision 80 exactly: same 12 cat-heavy datasets, same TabArena-Lite fold-0 splits, same 8-fold bagged fits (`sequential_local`, `debug_mode`), same in-process native backend, all local M2 CPU. The three arms are AutoGluon model wrappers subclassing the gauge's `BonsaiTSModel`; the bonsai config (depthwise, depth 6, lr 0.05, 1000 iters, es 50, seed 42) is untouched in every arm.

The lever lives entirely in the wrapper, outside the library. Arm `bonsai_ts_k1` is the stock gauge wrapper re-run unmodified: the reproduction control and the K=1 rung. Arms `k4`/`k8`: member 0 is the stock OTE at the identity ordering (byte-identical to arm 1's encoding, so the K curve nests); members 1..K-1 permute the training rows with `default_rng(k)`, fit a fresh stock OTE on the permuted rows, inverse-permute the encoded output, and the K encodings are averaged per column. Only the training matrix's causal TS columns change; the validation/test encoding uses OTE's full-training-set statistics, which are permutation-independent, so K moves nothing at transform time, exactly parallel to the train-time-only role of CatBoost's permutations.

Recovery per dataset, pre-registered: `recovery = (k1_error - kK_error) / (k1_error - cat_native_error)`, the fraction of the k1-to-CatBoost-native gap that K-averaging closes (lower metric_error is better throughout). The decision-55 chance band of about 0.001 applies to every per-dataset delta (single fold-0).

## Reproduction check: PASS, exact

Fresh `bonsai_ts_k1` under this harness vs the cached gauge bonsai_ts: max |delta| over the 12 datasets is 8.3e-17, machine epsilon, on every dataset. bonsai's determinism plus identical splits means the control does not merely land within the pre-registered ~0.002 band, it reproduces the cached numbers bit-for-bit (the gauge's Ray backend vs this probe's native backend included). The chunked reruns also reproduced the smoke run's credit-g values to the last digit.

## The recovery table (lower metric_error is better; recovery per the pre-registered formula)

| dataset | metric | rows | k1 (=bonsai_ts) | cat native | k4 | k8 | improve k4 | improve k8 | recovery k4 | recovery k8 |
|--|--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Amazon_employee_access | roc_auc | 32769 | 0.15224 | 0.12874 | 0.15504 | 0.15546 | **-0.00279** | **-0.00322** | -0.12 | -0.14 |
| kddcup09_appetency | roc_auc | 50000 | 0.16963 | 0.15808 | 0.17193 | 0.17030 | **-0.00230** | -0.00067 | -0.20 | -0.06 |
| in_vehicle_coupon_recommendation | roc_auc | 12684 | 0.19014 | 0.15334 | 0.18825 | 0.18776 | **+0.00190** | **+0.00239** | +0.05 | +0.06 |
| Marketing_Campaign | roc_auc | 2240 | 0.07432 | 0.06701 | 0.07187 | 0.07219 | **+0.00245** | **+0.00212** | +0.34 | +0.29 |
| credit-g | roc_auc | 1000 | 0.22466 | 0.21226 | 0.21380 | 0.23171 | **+0.01085** | **-0.00705** | +0.88 | -0.57 |
| qsar-biodeg | roc_auc | 1054 | 0.08887 | 0.07531 | 0.09060 | 0.08825 | **-0.00173** | +0.00061 | -0.13 | +0.05 |
| Bank_Customer_Churn | roc_auc | 10000 | 0.13009 | 0.12294 | 0.12936 | 0.12936 | +0.00074 | +0.00074 | +0.10 | +0.10 |
| customer_satisfaction_in_airline | roc_auc | 129880 | 0.00653 | 0.00520 | 0.00652 | 0.00659 | +0.00001 | -0.00007 | +0.01 | -0.05 |
| NATICUSdroid | roc_auc | 7491 | 0.01516 | 0.01404 | 0.01516 | 0.01516 | 0.00000 | 0.00000 | 0.00 | 0.00 |
| anneal | log_loss | 898 | 0.07005 | 0.05176 | 0.07005 | 0.07005 | 0.00000 | 0.00000 | 0.00 | 0.00 |
| splice | log_loss | 3190 | 0.10538 | 0.09051 | 0.10538 | 0.10538 | 0.00000 | 0.00000 | 0.00 | 0.00 |
| MIC | log_loss | 1699 | 0.49134 | 0.46338 | 0.49134 | 0.49134 | 0.00000 | 0.00000 | 0.00 | 0.00 |

Bold marks deltas beyond the 0.001 chance band. The four exact zeros are structural, not empirical: the gauge wrapper falls back to plain ordinal codes on multiclass (anneal, splice, MIC), and NATICUSdroid's 87 binary flags are typed int by AutoGluon so no column is category-typed; on all four the encoder never runs and K is a no-op by construction. The lever can only move the 8 TS-active binary datasets, and both pool views are reported below.

## The K curve

| aggregate | K=1 | K=4 | K=8 |
|--|--:|--:|--:|
| pool(12) mean recovery, mean of per-dataset ratios | 0 | +0.077 | -0.026 |
| pool(12) recovery, ratio of means (agg improve / agg gap) | 0 | +0.052 | -0.029 |
| TS-active(8) mean recovery | 0 | +0.116 | -0.039 |
| TS-active(8) ratio of means | 0 | +0.080 | -0.045 |
| pool(12) mean improve (metric_error) | 0 | +0.00076 | -0.00043 |

The curve is non-monotone: a small positive bump at K=4 (under an eighth of the share on every aggregate view) and negative at K=8. Per-dataset distribution at K=4: beyond-band helps on 3 (credit-g, Marketing_Campaign, in_vehicle_coupon), beyond-band hurts on 3 (Amazon, kddcup09, qsar-biodeg), inside the band or structurally zero on 6. At K=8: helps on 2 (in_vehicle_coupon, Marketing_Campaign), hurts on 2 (Amazon, credit-g), zero or noise on 8.

## Verdict: WEAK fired

Pre-registered criteria: STRONG if K=8 recovers at least half the categorical share on the pool mean; WEAK if under a quarter. K=8 recovers -0.03 on the pool mean (negative on every aggregate view), and even the best rung, K=4, recovers +0.05 to +0.12 depending on the view, all far under a quarter. The WEAK verdict fires with no interpretive strain: static K-permutation averaging is not the mechanism behind CatBoost's categorical share, and the dynamic per-split machinery (per-tree permutations, target statistics recomputed inside the tree, ctr feature combinations) remains the substance of the doc-17 price. The doc-17 decision stands, with sharper knowledge: rung 2 (productizing K-averaging) is dead on arrival, and any future reopening starts from the engine design, not the encoder.

The sharpest evidence is where the share lives: on the two datasets where decision 80 priced the machinery largest among TS-active tasks (Amazon -0.048, kddcup09 -0.019), static averaging HURTS or is flat at both K. If CatBoost's edge were permutation averaging, these are exactly where averaging had the most room to help; instead the single-ordering encoder is already the better static approximation there.

A mechanism consistent with all of it: as K grows, the average of causal prefix means converges toward leave-one-out target statistics, and decision 58 already measured that non-causal target encoding leaks (0.8462 vs 0.8590 ordered on amazon). The single ordering's prefix noise acts as implicit regularization; averaging it away walks the training encoding toward the leaky limit, worst where cardinalities are high and categories small (Amazon). CatBoost never materializes this average as a feature: it resamples a permutation per boosting step, which is a training-dynamics property, unreachable by any static preprocessing.

## Where K=8 hurts (the over-smoothing ledger)

Beyond the band: Amazon_employee_access (-0.0032 AUC-error, the pool's highest-cardinality task) and credit-g (-0.0071, its smallest task at 1000 rows). credit-g is also the loudest warning about reading single-fold small-data deltas: it swings from the pool's best K=4 result (+0.0109) to a beyond-band K=8 loss, an 0.018 swing between adjacent rungs on the same split, which is variance, not a dose-response. qsar-biodeg (-0.0017) and kddcup09 (-0.0023) sit beyond the band at K=4 only. The honest summary is that averaging trades a small median gain on mid-size tasks against real losses exactly where the categorical machinery matters most.

## Costs

Encode time scales as K by construction, measured x3.9 to x4.3 at K=4 and x7.7 to x8.6 at K=8 on pool-matched shapes (`OrderedTargetEncoder.fit_transform`, M2): the largest TS-active shape (113k rows, 17 cat cols, airline-like) costs 0.31s at K=1, 1.26s at K=4, 2.54s at K=8; kddcup09-like (44k rows, 39 of 213 cols) 0.34 / 1.44 / 2.90s; credit-g-like 2ms / 8ms / 15ms. Because bonsai's fit dominates, end-to-end 8-fold bagged train time moves much less: summed over the pool, 126.6s (K=1), 135.4s (K=4), 143.8s (K=8); per-dataset K=8/K=1 multipliers run 0.92x to 1.85x (worst kddcup09, 39 columns times 8 folds times 8 permutations), most near 1.0x. Cost is not the reason to decline; the recovery number is.

## Budget, protocol notes, deviations

Fresh compute: 12 datasets x 3 arms x 8 bag folds, all local M2 CPU, foreground in three chunks (147s + 163s + 195s, 505s total, plus a 13s single-dataset smoke). Deviations flagged: (1) the task brief described the gauge wrapper as using `cross=2`; the committed gauge wrapper (`tmp_scripts/bonsai_model.py`) actually constructs `OrderedTargetEncoder(cat_positions)` at defaults, i.e. `cross=1`, `keep_codes=True`, `prior_weight=10`, `seed=0`, and reproducing the cached bonsai_ts bit-for-bit confirms `cross=1` is what the gauge ran; this probe therefore K-averages the encoder as the gauge actually uses it. (2) The lever is structurally inert on 4 of 12 pool datasets (three multiclass fallbacks plus NATICUSdroid's int-typed flags), so pool(12) aggregates dilute toward zero; the TS-active(8) aggregates are reported alongside and the verdict fires on both. (3) The matrix ran as three sequential invocations of the same script over dataset subsets rather than one; bonsai's determinism makes this equivalent, verified by the smoke-vs-chunk bit-identical credit-g row. (4) Single fold-0 throughout, per the gauge's lite protocol: per-dataset deltas inside the 0.001 band are noise and only the aggregates are read.
