# LightGBM

LightGBM made histogram training the field's default idiom and contributed the ideas that squeeze the most out of it: leaf-wise growth under a leaf budget instead of level-by-level, the histogram subtraction trick, gradient-based one-side sampling (GOSS), and exclusive feature bundling (EFB) for sparse data.

bonsai adopted most of that list outright; lightgbm is the library bonsai resembles most in training structure.

## Adopted: the training idioms

The `leafwise` grower — best-first growth with a `max_leaves` budget — is bonsai's default in the Python estimators, because on most tabular workloads it reaches a given accuracy in fewer nodes than level-wise growth ([guide chapter 4](../guide/4-growing-trees.md)).

The subtraction trick — build the histogram for the smaller child, derive the sibling by subtracting from the parent — is wired through every grower ([guide chapter 2](../guide/2-binning-and-histograms.md)).

GOSS is bonsai's `goss` sampler ([guide chapter 5](../guide/5-sampling.md)). Its A/B benchmark earned its keep twice: once as a feature, and once by exposing a latent stale-score bug that affected all subsampled training — the kind of defect that only surfaces when every feature ships with a measured comparison against the library that invented it.

The initial-score convention (boost from the average, log class priors for multiclass) also follows lightgbm's.

## Measured and rebuilt: native categorical splits

lightgbm's native categorical set splits were the reference point for bonsai's categorical decision, and the measurements cut both ways.

On amazon (categorical-heavy), lightgbm-native scores 0.8572 AUC — but bonsai plus a 40-line ordered-target-statistics preprocessing step scores 0.8590, above it. On kick, lightgbm's own categorical toggle *hurts* (−0.018 vs treating the codes as ordinals).

So bonsai kept set splits out of the engine and shipped the preprocessing instead ([decision 58](https://github.com/daniel-m-campos/bonsai/blob/main/docs/decisions.md), [trade-off study](https://github.com/daniel-m-campos/bonsai/blob/main/benchmarks/categorical-tradeoff-2026-07.md)) — a per-dataset tool rather than an always-on engine feature that can quietly cost accuracy.

## Not yet measured: EFB

Exclusive feature bundling is lightgbm's lever for sparse, one-hot-heavy data, and bonsai does not implement it.

The benchmark suite to date is dense, so the honest status is untested rather than declined — if a sparse workload ever puts bonsai meaningfully behind lightgbm, EFB is the first hypothesis to price.

## The score today

At 16M rows on CPU, bonsai's growers fit in 73–76s against lightgbm's 111.3s, and the gap widens on wide data (4096 columns: lightgbm 256.2s, bonsai's GPU growers 42–44s; the re-baseline runs lightgbm on its CPU path, which remains its strongest).

Accuracy is lightgbm's consolation: at 16M it reaches 0.879 r², tying bonsai's depthwise and edging its oblivious (0.876) — recorded in the [performance table](https://github.com/daniel-m-campos/bonsai/blob/main/README.md#performance) with everything else.

bonsai's default training configuration — leafwise growth over histograms with subtraction, GOSS available behind a flag — is essentially lightgbm's recipe, reimplemented small enough to read and measured until it kept up.
