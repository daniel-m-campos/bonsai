# Per-node / per-level feature subsampling: declined by measurement (2026-07)

Issue #45 asks for per-level and per-node feature draws (xgboost's `colsample_bylevel` / `colsample_bynode`, lightgbm's `feature_fraction_bynode`) on top of the per-tree draw bonsai already has (`tree.feature_fraction` == `colsample_bytree`). The claim is that per-node sampling decorrelates deep trees better. Building it touches the grower's selected-features plumbing and the CUDA find staging (which assumes a per-tree selection), so the feature-admission gate applies: measure the benefit at zero core cost by toggling the reference libraries' own knobs, and only build if per-node/per-level BEATS the per-tree knob bonsai already has.

Probe: `scripts/probe_feature_subsample.py`, campaign knobs (200 iters, lr 0.05, depth 6, seed 42). Chance band ~2% rmse / ~0.001 auc (decision 55).

## Q1: in isolation, does bynode/bylevel beat bytree at a matched fraction?

xgboost, delta vs the no-subsample baseline (lower rmse / higher auc is better):

| dataset (feats) | bytree0.7 | bytree0.5 | bylevel0.7 | bynode0.7 | bynode0.5 |
|---|--:|--:|--:|--:|--:|
| california (8, rmse) | -0.0049 | +0.0068 | -0.0060 | -0.0034 | -0.0053 |
| higgs60k (28, auc) | -0.0004 | +0.0001 | -0.0011 | -0.0007 | -0.0008 |
| year_msd60k (90, rmse) | **-0.0210** | -0.0202 | -0.0155 | +0.0002 | -0.0117 |
| synth_wide200 (200, rmse) | -1.92 | **-4.73** | -1.24 | -0.71 | -1.18 |

On the three realistic datasets (<=90 features) every arm sits in the chance band or per-tree wins; per-node never leads. On the 200-feature synthetic, per-tree at 0.5 (-4.73) dominates the best per-node arm (-1.18) by ~4x.

## Q2/Q3: the wide best case: stack per-node on per-tree, then tune per-tree alone

`synth_wide200`, rmse (lower better):

| config | rmse |
|---|--:|
| bytree0.7 | 93.68 |
| bytree0.5 | 90.87 |
| bytree0.5 + bynode0.5 | 90.26 |
| **bytree0.4 (per-tree only)** | **89.34** |
| bytree0.3 | 89.59 |

Stacking `bynode0.5` on `bytree0.5` does add a real -0.6 rmse on 200-feature data (its one genuine signal). But **a well-tuned per-tree fraction alone beats it**: `bytree0.4` (89.34) is better than the best per-node stack `bytree0.5+bynode0.5` (90.26) by ~0.9 rmse. The regularization per-node offers is reachable, and exceeded, by tuning the single knob bonsai already ships.

## Verdict

**Declined by measurement (decision 75).** Per-node/per-level feature subsampling is dominated by bonsai's existing per-tree `feature_fraction` on every dataset tested, including the 200-feature synthetic where feature bagging matters most. It changes no standings and unlocks no workload bonsai cannot already serve by tuning `feature_fraction`. The cost (grower selected-features plumbing plus a CUDA find-staging change that today assumes a per-tree selection) buys a strictly weaker regularization lever.

**Reopener:** a real workload with many features where per-node/per-level beats a tuned `feature_fraction` beyond the chance band. None found across four datasets spanning 8 to 200 features. bonsai's wide-data strength is the regime to watch, but on synthetic 200-feature data the existing knob already wins.
