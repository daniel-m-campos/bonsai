# Learning-rate rule probe: how much of CatBoost's default-vs-default lead is the lr? (2026-07)

Decision 81 withdrew the ordered-boosting attribution and re-pinned CatBoost's small-data lead on defaults and protocol. This probe isolates the learning-rate slice of that defaults residual: bonsai ships a fixed default lr of 0.05, CatBoost resolves its default per dataset from a size/iterations heuristic, and the question is how much of the default-vs-default gap an lr rule alone could close. Five bonsai arms price it: the reproduction arm, a flat 0.1 control, a validation-selected per-dataset oracle (the ceiling of any rule), CatBoost's own auto-lr transplanted into bonsai, and a leave-one-out two-parameter size rule fit on the oracle's choices. CatBoost's numbers are the committed ordered-probe default arm, not fresh runs.

Probe script: `scripts/probe_lr_rule.py`. Raw rows: `benchmarks/results/lr-rule-probe-2026-07.jsonl`. Reference rows: `benchmarks/results/ordered-boosting-probe-2026-07.jsonl`. Everything ran locally (M2, CPU) in the TabArena-Lite gauge venv (CatBoost 1.2.10, scikit-learn 1.7.2), with bonsai from `build-tabarena/python`.

## Pool, protocol, and the fidelity gate

The pool and protocol are the ordered-boosting rung-0 probe's, imported from its script so split identity holds by construction: the same 12 pure-numeric small datasets (6 regression on rmse, 6 binary on 1 minus roc_auc), the same single split (gauge fold-0 where available, else the fixed stratified 75/25 holdout, seed 42), the same 20% validation slice driving early stopping (rounds 50), depth 6, 1000-iteration cap, single-model fits. Chance band per decision 55: about 2% relative of the metric for rmse, 0.001 absolute for 1 minus roc_auc. `catboost_lead` is the committed `bonsai - cat_ordered_def`, positive when CatBoost's default arm beats bonsai's.

Reproduction gate: the fresh `bonsai_default` arm reproduces the committed `bonsai` column EXACTLY on all 12 datasets (delta +0.000e+00 everywhere, bit-for-bit; fresh arm wall 15.3 s vs the cached 15.8 s). The gate ran before any new arm on each dataset.

From the committed rows, CatBoost-default leads bonsai-default beyond the band on exactly 2 of 12 datasets: QSAR_fish_toxicity (+0.0753, band 0.0207) and MagicTelescope (+0.00192, band 0.001). Those two are the datasets any lr story must move.

## The five arms, test error (lower better)

| dataset | n_train | cat def (ref) | cat lead (ref) | band | default (0.05) | flat 0.1 | oracle | cat rule | LOO rule |
|--|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| QSAR_fish_toxicity | 483 | 0.9582 | +0.0753 | 0.0207 | 1.0335 | 1.0167 | 1.0220 | 1.0290 | 1.0092 |
| concrete_compressive_strength | 548 | 5.0087 | -0.4213 | 0.0918 | 4.5874 | 4.6595 | 4.5857 | 4.7082 | 4.5534 |
| QSAR-TID-11 | 3062 | 0.9093 | -0.0264 | 0.0177 | 0.8829 | 0.8871 | 0.8895 | 0.8837 | 0.8879 |
| houses | 11008 | 0.2273 | +0.0009 | 0.0046 | 0.2282 | 0.2280 | 0.2277 | 0.2261 | 0.2265 |
| superconductivity | 11340 | 10.6266 | -0.7703 | 0.1971 | 9.8563 | 9.8584 | 9.8563 | 9.8576 | 9.9162 |
| wind | 3944 | 3.0006 | +0.0201 | 0.0604 | 3.0207 | 3.0476 | 3.0198 | 3.0261 | 3.0354 |
| breast_cancer | 340 | 0.00545 | +0.00063 | 0.00100 | 0.00608 | 0.00650 | 0.00671 | 0.00692 | 0.00650 |
| pima_diabetes | 460 | 0.17576 | -0.00824 | 0.00100 | 0.16752 | 0.17648 | 0.16609 | 0.16740 | 0.16358 |
| banknote | 823 | 0.00000 | +0.00003 | 0.00100 | 0.00003 | 0.00003 | 0.00007 | 0.00003 | 0.00003 |
| phoneme | 3242 | 0.06992 | -0.00230 | 0.00100 | 0.06763 | 0.06573 | 0.06513 | 0.06562 | 0.06440 |
| spambase | 2760 | 0.01081 | -0.00031 | 0.00100 | 0.01050 | 0.01040 | 0.01135 | 0.01071 | 0.01067 |
| MagicTelescope | 11412 | 0.05924 | +0.00192 | 0.00100 | 0.06116 | 0.06255 | 0.06420 | 0.06030 | 0.06081 |

Gain over the default in chance-band units (positive = better than lr 0.05, magnitude 1.0 = at the band edge):

| dataset | flat 0.1 | oracle | cat rule | LOO rule |
|--|--:|--:|--:|--:|
| QSAR_fish_toxicity | +0.81 | +0.56 | +0.22 | +1.18 |
| concrete_compressive_strength | -0.79 | +0.02 | -1.32 | +0.37 |
| QSAR-TID-11 | -0.24 | -0.37 | -0.05 | -0.29 |
| houses | +0.06 | +0.10 | +0.46 | +0.38 |
| superconductivity | -0.01 | +0.00 | -0.01 | -0.30 |
| wind | -0.45 | +0.01 | -0.09 | -0.24 |
| breast_cancer | -0.42 | -0.63 | -0.84 | -0.42 |
| pima_diabetes | -8.96 | +1.43 | +0.12 | +3.94 |
| banknote | +0.00 | -0.03 | -0.00 | +0.00 |
| phoneme | +1.89 | +2.50 | +2.01 | +3.23 |
| spambase | +0.10 | -0.85 | -0.21 | -0.17 |
| MagicTelescope | -1.39 | -3.04 | +0.87 | +0.35 |
| **pool mean** | **-0.78** | **-0.03** | **+0.10** | **+0.67** |

Raw-unit pool means: regression default 3.2682 vs flat10 3.2829, oracle 3.2668, cat rule 3.2885, LOO rule 3.2714; binary default 0.05215 vs flat10 0.05362, oracle 0.05226, cat rule 0.05183, LOO rule 0.05100.

## The chosen, transplanted, and rule lrs: do they even agree?

| dataset | n_train | oracle lr | transplanted lr (CatBoost) | LOO rule lr | LOO (a, b) |
|--|--:|--:|--:|--:|--|
| QSAR_fish_toxicity | 483 | 0.08 | 0.0454 | 0.0872 | (-1.226, -0.196) |
| concrete_compressive_strength | 548 | 0.12 | 0.0463 | 0.0778 | (-1.538, -0.161) |
| QSAR-TID-11 | 3062 | 0.08 | 0.0607 | 0.0587 | (-1.259, -0.196) |
| houses | 11008 | 0.03 | 0.0742 | 0.0542 | (-1.618, -0.139) |
| superconductivity | 11340 | 0.05 | 0.0746 | 0.0463 | (-1.237, -0.197) |
| wind | 3944 | 0.02 | 0.0632 | 0.0652 | (-1.484, -0.150) |
| breast_cancer | 340 | 0.30 | 0.0243 | 0.0623 | (-2.503, -0.047) |
| pima_diabetes | 460 | 0.12 | 0.0262 | 0.0800 | (-1.549, -0.159) |
| banknote | 823 | 0.01 | 0.0302 | 0.1040 | (-0.264, -0.298) |
| phoneme | 3242 | 0.03 | 0.0424 | 0.0642 | (-1.355, -0.172) |
| spambase | 2760 | 0.12 | 0.0407 | 0.0578 | (-1.258, -0.201) |
| MagicTelescope | 11412 | 0.20 | 0.0578 | 0.0303 | (-0.183, -0.355) |

They do not agree; they are mildly ANTI-correlated (corr of ln lrs: oracle vs transplanted -0.261). Three separate facts sit in this table.

CatBoost's own rule barely moves off 0.05 here. The transplanted lrs span 0.024 to 0.075 with the small binary sets pulled low and the 11k-row sets pulled high; a least-squares fit of the transplanted values gives ln(lr) = -4.882 + 0.236 ln(n_train) (corr +0.824), an INCREASING size rule. On this pool CatBoost's celebrated auto-lr is a gentle wiggle around exactly the value bonsai already ships, and transplanting it moves the pool mean +0.10 band units, a no-op.

The oracle's choices trend the OTHER way and are mostly validation noise. The oracle lr correlates with size at -0.255, and the pooled LOO fit on its choices is ln(lr) = -1.2818 - 0.1899 ln(n_train), a DECREASING rule (about 0.085 at 500 rows, 0.047 at 11k rows), opposite in sign to CatBoost's heuristic. The per-fold (a, b) swing from (-2.503, -0.047) to (-0.183, -0.355) depending on which single dataset is held out, which is what fitting 2 parameters to 11 noisy points looks like. The extreme oracle picks are tells: 0.3 on breast_cancer (chosen by a 0.0006 AUC-scale val edge on 86 validation rows, worse on test), 0.01 on banknote (an all-zero-val-error tie broken toward the low grid end), 0.2 on MagicTelescope (worse on test by 3 band units).

The validation split cannot carry the selection at these sizes. The oracle beats the default on the VALIDATION split on 10 of 12 datasets, but on TEST on only 6 of 12. What the sweep giveth on val, test taketh away.

## Verdict: LR IS NOT THE STORY fired

Pre-registered criteria and which fired:

LR IS THE STORY (oracle closes >= half the committed CatBoost-default lead on the beyond-band lead datasets AND improves the pool mean beyond band): DID NOT FIRE. On QSAR_fish_toxicity the oracle closes +15.3% of the +0.0753 lead; on MagicTelescope it closes -158.2% (the val-chosen lr 0.2 is worse than 0.05 on test). The oracle's pool-mean gain is -0.03 band units, dead inside the band.

RULE PRACTICAL (arm 4 or 5 captures >= ~70% of the oracle's pool-mean gain): NOT MEANINGFULLY EVALUABLE, because the oracle's pool-mean gain is zero-to-negative; there is nothing to capture 70% of. Both rules numerically EXCEED the oracle (cat rule +0.10, LOO rule +0.67 band units vs the oracle's -0.03), which is not evidence for a rule but the signature of selection noise: a smooth rule cannot overfit the validation slice, the per-dataset argmin can and does. And neither rule reaches the band on the pool mean either, so there is no productizable auto-default here regardless.

LR IS NOT THE STORY (oracle gain inside the band on the pool mean): FIRED. What the oracle DID move, for the record: beyond the band only on pima_diabetes (+1.4 band units at lr 0.12) and phoneme (+2.5 at lr 0.03), and it HURT beyond the band on MagicTelescope (-3.0 at lr 0.2); everything else is inside the band, including the entire regression pool (best oracle regression move: QSAR_fish_toxicity at +0.56 bands, still inside). The two beyond-band wins are exactly offset by the beyond-band loss plus the drizzle of small losses, hence the -0.03 pool mean.

The flat-0.1 control answers the is-0.05-simply-wrong question: no. Flat 0.1 is -0.78 band units on the pool mean, with a -8.96 band-unit catastrophe on pima_diabetes; 0.05 is on the flat part of the pool's lr response and 0.1 is off it.

The remaining defaults residual (decision 81's protocol-and-defaults attribution) therefore does not decompose onto the learning rate on this pool. The committed lead datasets stay unclosed: QSAR_fish_toxicity's +0.0753 lead is 3.6 bands wide and the best any lr arm does is +32% of it (LOO rule, still leaving a 2.5-band gap), and MagicTelescope's lead is only approached by the cat rule (+45%) and by less than half. Whatever CatBoost's default arm is doing on those two, it is not primarily its lr heuristic, which on this pool resolves to approximately bonsai's own 0.05 anyway.

## The ES interaction, honestly

Chosen-lr vs trees-used correlation at the oracle's choice: -0.79 (Spearman -0.69). Low lr rides early stopping longer, as it must: mean trees kept across the pool falls monotonically from 863 at lr 0.01 to 99 at lr 0.3.

| grid lr | 0.01 | 0.02 | 0.03 | 0.05 | 0.08 | 0.12 | 0.2 | 0.3 |
|--|--:|--:|--:|--:|--:|--:|--:|--:|
| mean trees kept | 863 | 745 | 658 | 465 | 383 | 208 | 157 | 99 |
| datasets at the 1000 cap | 9/12 | 6/12 | 4/12 | 2/12 | 1/12 | 0/12 | 0/12 | 0/12 |

The honest caveat inside that: at lr 0.01 the 1000-iteration cap binds on 9 of 12 datasets (and on 2 of 12 at the default 0.05: QSAR-TID-11 and superconductivity), so the low end of the grid is measured cap-truncated, not ES-terminated; "lr 0.01 with unlimited trees" is not what this probe priced. The cap is the ordered probe's own knob, kept identical by design. It does not rescue an lr story: lr 0.02 and 0.03 are cap-free on half the pool and still deliver no beyond-band pool gain, and where the cap binds at 0.05 the oracle re-chose 0.05 (superconductivity) or 0.08 (QSAR-TID-11) rather than lower, so more headroom for low lrs is not where the pool wanted to go.

## The full sweep, test error by grid lr (the oracle's distribution)

| dataset | 0.01 | 0.02 | 0.03 | 0.05 | 0.08 | 0.12 | 0.2 | 0.3 |
|--|--:|--:|--:|--:|--:|--:|--:|--:|
| QSAR_fish_toxicity | 1.0174 | 1.0192 | 1.0259 | 1.0335 | 1.0220 | 1.0127 | 1.0171 | 1.0397 |
| concrete_compressive_strength | 4.9404 | 4.6934 | 4.6781 | 4.5874 | 4.5944 | 4.5857 | 4.7402 | 4.7271 |
| QSAR-TID-11 | 0.9393 | 0.9030 | 0.8901 | 0.8829 | 0.8895 | 0.8884 | 0.8868 | 0.9074 |
| houses | 0.2338 | 0.2308 | 0.2277 | 0.2282 | 0.2271 | 0.2306 | 0.2297 | 0.2331 |
| superconductivity | 10.4581 | 10.1224 | 9.9088 | 9.8563 | 9.9099 | 10.0656 | 9.9170 | 10.2831 |
| wind | 3.0177 | 3.0198 | 3.0252 | 3.0207 | 3.0256 | 3.0225 | 3.0555 | 3.0824 |
| breast_cancer | 0.00608 | 0.00671 | 0.00650 | 0.00608 | 0.00629 | 0.00608 | 0.00734 | 0.00671 |
| pima_diabetes | 0.16501 | 0.16728 | 0.16693 | 0.16752 | 0.16358 | 0.16609 | 0.17075 | 0.17475 |
| banknote | 0.00007 | 0.00000 | 0.00003 | 0.00003 | 0.00003 | 0.00003 | 0.00003 | 0.00003 |
| phoneme | 0.06786 | 0.06545 | 0.06513 | 0.06763 | 0.06374 | 0.06426 | 0.06434 | 0.06673 |
| spambase | 0.01050 | 0.01069 | 0.01060 | 0.01050 | 0.01076 | 0.01135 | 0.01038 | 0.01104 |
| MagicTelescope | 0.06225 | 0.06125 | 0.06134 | 0.06116 | 0.06185 | 0.06103 | 0.06420 | 0.06460 |

Read across any row: the test response to lr is SHALLOW in the middle of the grid. Over 0.02 to 0.12 the test error moves by at most about 1.4 band units on 10 of 12 datasets, and the two exceptions (pima_diabetes at 3.9, phoneme at 3.9) are exactly the two datasets the oracle moved beyond band. A response that flat is why no selection scheme, oracle or rule, can mine a pool-level win from it. The matching validation-error grid, per-lr trees, and per-lr fit times are all in the jsonl (`sweep` field).

## Costs

Total probe wall 163.7 s on the M2 (plus a 4.6 s two-dataset smoke). Bonsai fit time by arm (pool totals): default 15.3 s, flat10 9.5 s, oracle sweep (8 lrs, default included) 118.0 s, cat-rule fit 15.2 s, LOO-rule fit 16.0 s. Resolving CatBoost's auto-lr via the one-tree callback stop cost 0.2 s for the whole pool, confirming the transplant is essentially free to compute at deploy time; it just does not buy anything here.

## Deviations, flagged loudly

1. The cat-rule lr is read from a CatBoost fit stopped after ONE TREE by a fit callback, not from a literal `iterations=1` fit as the brief sketched. The iteration count enters CatBoost's auto-lr formula: a literal `iterations=1` fit resolves learning_rate=0.5 (verified in-venv), which is the 1-iteration default, not the default the committed CatBoost arm ran at. The callback stop leaves `iterations` at the library default 1000, so `get_all_params()` reports the exact lr the committed default arm used, at one tree of compute. Same defaults-plus-eval_set context as the committed arm.
2. The low end of the lr grid is truncated by the ordered-probe's 1000-iteration cap (9 of 12 datasets at lr 0.01, 2 of 12 at 0.05), detailed in the ES section. Kept identical to the reference protocol by design; a deeper-cap sweep would be a different probe.
3. Oracle ties on the validation metric break toward the LOWER lr (grid ascending, first argmin). It fired once, on banknote's all-zero validation errors, and cost the oracle a within-band 0.00004.
4. The LOO rule lr is clamped to the grid range [0.01, 0.3] as a safety rail; no fold actually clamped (fitted rule lrs span 0.0303 to 0.1040).
