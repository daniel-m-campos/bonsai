# CatBoost

CatBoost contributed two structural ideas to gradient boosting.

The first is that boosting leaks its own training targets in two places (naive target statistics and per-round gradients both let a row see its own label), and that one mechanism fixes both: impose an artificial order and let each row learn only from the rows before it. Applied to categorical encoding this gives ordered target statistics; applied to the boosting loop it gives ordered boosting.

The second is the oblivious tree: force every node at a given depth to split on the same feature and threshold, and the tree becomes a bit-indexable table: worse per-tree fit, but faster evaluation and stronger regularization at scale.

bonsai adopted one of these ideas directly, rebuilt one as preprocessing, and declined one after measuring it.

## Adopted: oblivious trees

The `oblivious` grower is bonsai's symmetric-tree strategy and its strongest configuration: on GPU at 16M rows it is the fastest and most accurate variant bonsai has.

Getting there took a debugging lesson. Through mid-campaign, bonsai's oblivious accuracy trailed catboost's at scale (test r² 0.864 vs 0.876 at 16M), and the gap looked algorithmic: their per-level scoring, perhaps, or their quantization.

It was a bonsai bug: the device level-find kernel let infeasible frontier nodes veto level candidates, a missing port of a fix the CPU grower already had ([decision 63](https://github.com/daniel-m-campos/bonsai/blob/main/docs/decisions.md)). With the kernel fixed, bonsai matched catboost's accuracy exactly. The gap had never been theirs to explain.

## Rebuilt as preprocessing: ordered target statistics

catboost's categorical machinery (ordered target statistics, per-tree permutations, crossed CTRs) leads everything we tested on categorical-heavy data (0.8894 AUC on amazon).

bonsai implements the leak-free core of that machinery as [`OrderedTargetEncoder`](https://github.com/daniel-m-campos/bonsai/blob/main/python/bonsai/encoding.py), a preprocessing step rather than an engine feature: segmented-cumsum ordered statistics with optional pair crossings, about 40 lines, applied per dataset only where it helps.

On the same split, bonsai plus the encoder reaches 0.8590, above lightgbm's native categorical splits at 0.8572, with no added complexity in the training engine ([decision 58](https://github.com/daniel-m-campos/bonsai/blob/main/docs/decisions.md), [trade-off study](https://github.com/daniel-m-campos/bonsai/blob/main/benchmarks/categorical-tradeoff-2026-07.md)).

The remaining gap to catboost-native comes from the parts that are engine-side by nature: per-tree permutations and crossed CTRs. Designs for them exist in [architecture doc 17](https://github.com/daniel-m-campos/bonsai/blob/main/docs/architecture/17-categorical-splits.md), gated on measurements that would justify the complexity.

## Declined: ordered boosting

We tested the hypothesis that catboost's per-round accuracy at scale came from ordered boosting by benchmarking catboost against itself: `boosting_type=Ordered` vs `Plain`, same data, same budget.

The result was a wash (identical r², sometimes slightly worse, always ~7× slower), and catboost itself defaults to Plain past ~50k rows, so at 16M it was never running ordered to begin with ([scale-edge study](https://github.com/daniel-m-campos/bonsai/blob/main/benchmarks/catboost-scale-edge-2026-07.md)).

Prediction shift, the problem ordered boosting solves, matters on small, noisy, categorical-heavy data. At the scales bonsai targets, the benefit did not reproduce, so bonsai does not implement it; the measurements are recorded if the question comes up again.

## The score today

Same pod, matched settings, 16M rows: bonsai `cuda_oblivious` 18.4s, catboost-GPU 18.5s, both at 0.876 test r², with bonsai using ~3× less host memory (7.0 vs 19.4 GB).

On wide data catboost keeps the lead (1024 columns: 9.7s vs bonsai's 10.6; 4096: 35.8 vs 41.9), recorded in the [performance table](https://github.com/daniel-m-campos/bonsai/blob/main/README.md#performance).

bonsai's strongest configuration is built on catboost's tree structure, and the project's most useful debugging lesson came from assuming their advantage was real before checking whether it was our defect.
