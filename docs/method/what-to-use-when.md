# What to use when

This page answers one question: for a given shape of data, which library should you reach for? The table gives the short answer, and each row links the committed evidence behind it.

Several rows name a competitor. Those rows are stated as plainly as the bonsai rows, because an honest recommendation is the whole point of the page. Where bonsai loses, the loss is linked with the same prominence as the wins.

Read the table top to bottom as a decision on your data's shape. Row scale and hardware come first, then data character: categoricals, sparsity, numeric-only, ranking. The last two rows are about guarantees, not accuracy.

| your data | reach for | evidence |
|---|---|---|
| 1M+ rows, GPU training | bonsai | [frontier](results.md#gpu-accuracy-vs-time-frontier-at-16m), [airline AUC](../../benchmarks/airline-2026-07.md) |
| small tabular, heavy categoricals | CatBoost | [cat probe](../../benchmarks/tabarena-cat-probe-2026-07.md) |
| small pure-numeric tabular | split (see below) | [Grinsztajn](results.md#external-standings-the-grinsztajn-suite), [cat probe control](../../benchmarks/tabarena-cat-probe-2026-07.md) |
| memory-constrained hosts | bonsai | [perf ledger](results.md#perf-division) |
| sparse or high-dimensional-sparse | XGBoost | [sparse probe](../../benchmarks/sparse-tradeoff-2026-07.md) |
| learning-to-rank | XGBoost or LightGBM | [ranking probe](../../benchmarks/ranking-tradeoff-2026-07.md) |
| bit-reproducible artifacts across CPUs | bonsai only | [the contract](../design/determinism.md) |

## 1M+ rows on a GPU: bonsai

On the 16M accuracy-versus-time frontier, bonsai is first to every measured accuracy at every horizon ([decision 78](../decisions.md)). Its marginal round is 64 ms, below CatBoost's 78 on the same pod, and the last crossover is gone. On the real-data airline ladder, a bonsai variant has the best AUC in every cell from 1M rows up, under both protocols.

Two caveats travel with the speed. Identical GPUs measure up to 25% apart across the fleet, so only same-pod numbers compare. And XGBoost-GPU still owns raw wall clock on very narrow data, such as the airline set's 8 columns.

The unconditional frontier claim has one boundary. It holds for the eligible regime: MSE, no DART, no sample weights, and uniform or no row sampling. Ineligible fits keep the earlier frontier, whose only residue was plateau-depth.

## Small tabular data with heavy categoricals: CatBoost

On the cat-heavy TabArena subset, CatBoost's own toggle prices its native categorical machinery at 68% of its remaining lead over bonsai ([decision 80](../decisions.md)). That machinery is per-split ordered target statistics and feature combinations, which are engine-side by nature. It costs CatBoost 4.2x its own train time on these datasets, which is the price for the accuracy.

bonsai's `OrderedTargetEncoder` closes the other third and beats ablated CatBoost where the machinery matters most. What it cannot match is the native per-split statistics themselves.

This is the one row where a bonsai gap is measured as load-bearing, and the reopener predicate has fired. Whether to build native splits is a launch-strategy call, not a measurement gap.

## Small pure-numeric tabular data: it depends

This one splits by how you measure. At matched knobs on the Grinsztajn suite, bonsai has the best mean rank, 1.44 across 55 tasks, and leads both numeric subsuites.

On bagged aggregate leaderboards, CatBoost still leads small mixed and numeric data, and its pure-numeric control lead persists on 5 of 6 datasets with categoricals ablated. The mechanism is not ordered boosting: its own toggle prices that at zero or below on this regime ([the rung-0 probe](https://github.com/daniel-m-campos/bonsai/blob/main/benchmarks/ordered-boosting-probe-2026-07.md)); the lead points at the bagged-ensemble protocol and its defaults. So the honest answer is bonsai at matched knobs single-model, and CatBoost where bagged ensembles with its defaults are in play.

On the tuned TabArena-Lite gauge, CatBoost's Elo is 1340 against bonsai_ts's 1204. No categorical work closes that pure-numeric gap; it belongs to a future ordered-boosting campaign.

## Memory-constrained hosts: bonsai

bonsai bins each feature to one byte per cell, which holds its memory down. Peak host RSS at 16M rows is 7.0 GB, against XGBoost's 22.2 GB and CatBoost's 19.4 GB, roughly 3x less ([the perf overview](README.md), from [the re-baseline data](../../benchmarks/results/rebaseline-2026-07.jsonl)). Predict is about 3x faster on the same runs. That same u8 storage is what puts a 500M-row fit on one 80GB card in [case E5](../learn/engine/5-the-ceiling.md).

The footprint also compounds under fit-parallelism: more concurrent fits fit on one card. And a shared `bonsai.Dataset` bins once for a whole sweep.

## Sparse or high-dimensional-sparse data: XGBoost

bonsai declined sparse compute: no sparsity-aware split finding and no exclusive feature bundling ([the feature-gap record](https://github.com/daniel-m-campos/bonsai/blob/main/docs/feature_gap.md)). Its dense histograms pay every zero, so a 10k-column tf-idf fit ran 25x slower than the sparse libraries, at O(rows x columns) memory. Reach for a sparse-aware library above a few thousand genuinely sparse columns.

XGBoost pioneered sparsity-aware split finding; on the one sparse corpus measured, LightGBM's bundling path was the fastest. bonsai's claims live on dense and categorical data, not here.

One case looks sparse but is not. One-hot expansions have a better answer: feed the raw code columns to the encoder instead. That beat LightGBM's bundled one-hot on the control, 0.8708 against 0.8647.

## Learning-to-rank: XGBoost or LightGBM

bonsai's ranking probe declined the pairwise LambdaRank objective by measurement: it matched plain regression on relevance labels, which bonsai already does. The one stable gap is about 0.015 NDCG@10 to a listwise loss, on MQ2008 at both tree counts.

So for learning-to-rank today, reach for XGBoost's `rank:ndcg`, the listwise loss that showed the gap, or LightGBM's `lambdarank`. bonsai's issue #58 is reframed listwise-first and gated on confirming the margin across more folds.

## Bit-reproducible artifacts across CPUs: bonsai only

bonsai's model bytes are identical across runs, thread counts, and CPU architectures, and CI enforces it per commit. It trains a reference model on an arm64 Mac and an x86-64 Linux box and compares file hashes. No reference library makes this claim. If you need a bit-exact artifact to compare two builds by file hash, this is the one row where bonsai stands alone.

The contract also caught real bugs: an FMA contraction that diverged bytes across architectures, and a silent OpenMP downgrade. Both became hard errors, so the guarantee is enforced, not aspirational.

## Statuses change when decisions change

Every row here is a snapshot of the current record, not a permanent verdict. A declined feature reopens when a workload makes its gap load-bearing, exactly as native categoricals did in [decision 80](../decisions.md). When a decision moves, this page moves with it. The dated trail of what changed and why is [the project timeline](../learn/timeline.md).
