# CatBoost

CatBoost's core insight is that gradient boosting leaks its own training targets: naive target statistics and naive per-round gradients both let a row peek at its own label, and the fix — impose an artificial order, let each row learn only from rows before it — falls out twice, once as ordered target statistics and once as ordered boosting.

The other signature, oblivious trees, looks like a restriction and is actually a trade: force every node at a depth to ask the same question, and the tree becomes a bit-indexable table — worse per-tree fit, radically better evaluation speed and regularization at scale.

bonsai adopted one of these ideas whole, rebuilt one as preprocessing, and measured one into a respectful decline.

## Adopted: oblivious trees

The `oblivious` grower is bonsai's symmetric-tree strategy, and it earned the flagship slot: on GPU at 16M rows it is bonsai's fastest, most accurate configuration.

Getting there required believing catboost over our own output: through mid-campaign, bonsai's oblivious accuracy trailed catboost's at scale (r² 0.864 vs 0.876 at 16M), and the gap looked like algorithmic secret sauce — per-level candidate scoring, maybe, or their quantization.

It was our bug: the device level-find kernel let infeasible frontier nodes veto level candidates, a missing port of a fix the CPU grower already had ([decision 63](https://github.com/daniel-m-campos/bonsai/blob/main/docs/decisions.md)). One kernel change later, bonsai matched catboost's accuracy exactly — the "gap" had been a defect wearing their crown.

That episode is the reason this section exists: before concluding a reference library is better, prove the difference isn't your own defect.

## Rebuilt as preprocessing: ordered target statistics

catboost's categorical machinery — ordered target statistics with per-tree permutations and crossed CTRs — is genuinely excellent, and on categorical-heavy data (amazon: 0.8894 AUC) it leads everything we tested.

bonsai ships the leak-free core of the idea as [`OrderedTargetEncoder`](https://github.com/daniel-m-campos/bonsai/blob/main/python/bonsai/encoding.py), a preprocessing step rather than an engine feature: segmented-cumsum ordered statistics, optional pair crossings, ~40 lines a user can read and decline per dataset.

Measured on the same split, bonsai + the encoder reaches 0.8590 — above lightgbm's native categorical splits (0.8572) — while keeping the engine's core at zero added complexity ([decision 58](https://github.com/daniel-m-campos/bonsai/blob/main/docs/decisions.md), [the trade-off study](https://github.com/daniel-m-campos/bonsai/blob/main/benchmarks/categorical-tradeoff-2026-07.md)).

The remaining gap to catboost-native is real and documented: per-tree permutations and crossed CTRs are engine-side ideas by nature, and stage-2 designs exist in [architecture doc 17](https://github.com/daniel-m-campos/bonsai/blob/main/docs/architecture/17-categorical-splits.md) should the measurements ever justify them.

## Measured and declined: ordered boosting

The hypothesis that catboost's per-round accuracy edge at scale came from ordered boosting was tested directly: catboost against itself, `boosting_type=Ordered` vs `Plain`, same data, same budget.

Ordered vs Plain was a wash — identical r², sometimes worse, always ~7× slower — and catboost itself defaults to Plain past ~50k rows, so at 16M it was never ordered to begin with ([the scale-edge study](https://github.com/daniel-m-campos/bonsai/blob/main/benchmarks/catboost-scale-edge-2026-07.md)).

The idea is principled — prediction shift is real, and on small, noisy, categorical-heavy data ordered boosting defends against it — but at the scales bonsai serves, its benefit did not reproduce, and bonsai declines it with the evidence recorded.

## The score today

Same pod, matched settings, 16M rows: bonsai `cuda_oblivious` 18.4s, catboost-GPU 18.5s, both at 0.876 test r² — a photo finish bonsai wins while using ~3× less host memory (7.0 vs 19.4 GB).

On wide data catboost keeps the lead (1024 cols: 9.7s vs bonsai's 10.6; 4096: 35.8 vs 41.9) — recorded in the [performance table](https://github.com/daniel-m-campos/bonsai/blob/main/README.md#performance) rather than argued with.

What bonsai owes catboost is the shape of its strongest configuration and the sharpest lesson of the project: measure the reference before explaining it.
