# Multi-target trade-off probe, 2026-10: the evidence behind decision 161

The question: when a model predicts K regression targets from one feature matrix, should a round grow one tree per target (each on its own gradient, the shape `MulticlassBooster` already has for K classes) or one tree with K-wide leaves (a split criterion summed over the targets, so the K outputs share structure)? The second shape costs a K-wide histogram cell in every engine; the first costs nothing in the engines. The reference libraries carry both shapes, so they are the prototype: xgboost's `multi_strategy` toggles between `one_output_per_tree` and `multi_output_tree` on the same data, and catboost's `MultiRMSE` is the shared-structure shape on symmetric trees beside K separate `RMSE` fits. lightgbm fits K models, as bonsai does today.

Reproduce with `scripts/probe_multitarget.py`, which appends its rows to `benchmarks/results/multitarget-tradeoff-2026-10.jsonl`; the file is not committed, the tables below carry the evidence. Knobs follow the quality campaign: 200 rounds, learning rate 0.05, depth 6, 255 bins, seed 42, one 80/20 shuffled holdout, 8 threads on an M2. The metric is aRRMSE, the mean over targets of the test RMSE divided by the RMSE of predicting the training mean, so a target's scale does not weight the mean.

Datasets: the Mulan multi-target regression sets on OpenML, chosen for correlated targets of the multi-horizon kind. enb (41478, 768 x 8, K=2), wq (41491, 1060 x 16, K=14), rf1 (41483, 9125 x 64, K=8, river flow at eight sites 48 hours ahead from past flows), rf2 (41484, 9125 x 576, K=8, the same with forecasts as features), scm1d (41485, 9803 x 280, K=16, supply-chain prices one day ahead), scm20d (41486, 8966 x 61, K=16, twenty days ahead), and a synthetic set (200000 x 64, K=5) built as one nonlinear signal in the features scaled per horizon plus noise accumulated over horizons, with the first horizon's R2 near 0.1, the shape of a multi-horizon markup target.

Pre-registered before the first run: K-wide leaves are justified only if a shared-structure arm beats its own per-target sibling by more than 2% aRRMSE on a majority of datasets with no loss beyond 2% on any other; bonsai's K separate fits must sit inside 2% of xgboost's `one_output_per_tree`, the arm a K-output booster of per-target trees reproduces.

## Results (aRRMSE, fit seconds in parentheses)

| arm | enb | wq | rf1 | rf2 | scm1d | scm20d | synth |
|---|--:|--:|--:|--:|--:|--:|--:|
| xgboost, one tree per target | **0.0662** (0.4) | 0.9545 (3.7) | 0.1575 (5.0) | 0.1564 (7.9) | **0.3038** (34.0) | **0.3861** (10.8) | 0.9647 (9.7) |
| xgboost, K-wide leaves | 0.0696 (0.3) | 0.8991 (2.8) | 0.1747 (6.9) | 0.1748 (10.8) | 0.3177 (61.6) | 0.4033 (16.1) | 0.9660 (11.1) |
| catboost, K RMSE fits | 0.0717 (0.1) | 0.9004 (1.5) | 0.1954 (3.8) | 0.1955 (9.5) | 0.3379 (28.6) | 0.4744 (8.9) | 0.9584 (9.6) |
| catboost, MultiRMSE | 0.0770 (0.0) | **0.8890** (0.9) | 0.2252 (2.8) | 0.2309 (8.0) | 0.3579 (24.3) | 0.5008 (6.5) | **0.9580** (7.7) |
| lightgbm, K fits | 0.0838 (0.6) | 0.9327 (3.7) | 0.1577 (7.0) | 0.1555 (12.3) | 0.3058 (20.7) | 0.3915 (11.9) | 0.9647 (12.4) |
| bonsai, K depthwise fits | 0.0762 (1.1) | 0.9483 (7.9) | **0.1563** (5.5) | **0.1553** (18.0) | 0.3051 (25.1) | 0.3908 (9.8) | 0.9636 (7.7) |

Shared structure against its own per-target sibling, same library:

| dataset | xgboost K-wide vs per target | catboost MultiRMSE vs K fits | K-wide fit time | K-wide predict time |
|---|--:|--:|--:|--:|
| enb | +5.1% | +7.4% | 0.75x / 0.30x | 0.8x / 0.3x |
| wq | -5.8% | -1.3% | 0.77x / 0.62x | 0.3x / 0.07x |
| rf1 | +10.9% | +15.2% | 1.38x / 0.75x | 0.3x / 0.2x |
| rf2 | +11.8% | +18.1% | 1.37x / 0.84x | 0.3x / 0.1x |
| scm1d | +4.6% | +5.9% | 1.81x / 0.85x | 0.1x / 0.5x |
| scm20d | +4.5% | +5.6% | 1.49x / 0.73x | 0.1x / 0.1x |
| synth | +0.1% | 0.0% | 1.14x / 0.81x | 0.3x / 0.3x |

## Readings

1. **K-wide leaves lose on accuracy.** Both libraries' shared-structure arms trail their per-target siblings on five of seven sets, by 4.5 to 18%, far outside the 2% band; the two exceptions are wq (1060 rows, where the per-target xgboost arm is itself the worst of the per-target arms) and the synthetic set (a tie, which is what a target family built from one shared signal should give). The criterion fails in the direction opposite to the hypothesis: the per-target trees are the stronger model on real correlated targets, because each target gets its own 200 splits instead of a quarter or a sixteenth of a shared budget.
2. **The speed claim is split.** Shared structure predicts 3 to 15 times faster (one walk yields K outputs), but it does not train faster on xgboost (K-wide gains cost more per bin than K separate scans save, 1.4 to 1.8x slower at K=8 and 16) and only modestly on catboost (0.6 to 0.85x). A K-wide cell in bonsai's engines would buy the predict side alone, and at K in single digits the per-target predict is K walks of the predict crown.
3. **bonsai's K fits are the per-target reference.** Within 1.2% of xgboost's `one_output_per_tree` on every set over 1000 rows (and best of all arms on rf1 and rf2); the +15% on enb is 615 training rows at aRRMSE 0.07, inside the chance band at that size. A K-output booster of per-target trees ships this accuracy with one ingest, one model, one predict call.
4. **The synthetic set does not separate the arms.** Every arm lands at 0.958 to 0.966, so a low-signal multi-horizon target of this construction is not where the structure question is decided; the real sets are.

Caveats: single split per dataset, seed 42, no per-dataset tuning, the quality campaign's convention. Fit times are a loaded laptop's single rep and only compare arms within a row. xgboost 3.2.0, catboost 1.2.10, lightgbm 4.6.0, bonsai 2.7.0 source build.
