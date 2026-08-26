# What to use when

This page answers one question: for a given shape of data, which library should you reach for? The table gives the short answer, and each row links the committed evidence behind it.

Several rows name a competitor. Those rows are stated as plainly as the bonsai rows, because an honest recommendation is the whole point of the page. Where bonsai loses, the loss is linked with the same prominence as the wins.

Read the table top to bottom as a decision on your data's shape. Row scale and hardware come first, then data character: categoricals, sparsity, numeric-only, ranking. The last two rows are about guarantees, not accuracy.

| your data | reach for | evidence |
|---|---|---|
| 1M+ rows, GPU training | bonsai | [perf ledger](results.md#perf-division), [airline AUC](../../benchmarks/airline-2026-07.md) |
| small tabular, heavy categoricals | CatBoost | [cat probe](../../benchmarks/tabarena-cat-probe-2026-07.md) |
| small pure-numeric tabular | depends on the protocol | [Grinsztajn](results/quality-grinsztajn.md#external-standings-the-grinsztajn-suite), [cat probe control](../../benchmarks/tabarena-cat-probe-2026-07.md) |
| memory-constrained hosts | bonsai | [perf ledger](results.md#perf-division) |
| wide or extreme-aspect data (thousands of features) | bonsai | [the archive](results/archive.md), decisions 90 and 91 |
| sparse or high-dimensional-sparse | XGBoost | [sparse probe](../../benchmarks/sparse-tradeoff-2026-07.md) |
| learning-to-rank | XGBoost or LightGBM | [ranking probe](../../benchmarks/ranking-tradeoff-2026-07.md) |
| bit-reproducible artifacts across CPUs | bonsai only | [the contract](../learn/determinism-as-a-contract.md) |

## Where each row is argued

Numbers stay in one place: [the results ledger](results.md) and the [scenario panels](results/perf.md) carry every figure, under the rules in [the benchmark protocol](benchmark-protocol.md). This page never restates a digit, so the ledger cannot go stale behind it.

The reasoning sits with whoever owns it: [how we decide](how-we-decide.md) for the pricing discipline that produced the declines, [the decisions log](../decisions.md) for the dated verdict on each one, [determinism as a contract](../learn/determinism-as-a-contract.md) and [the HPC tension](../learn/the-hpc-tension.md) for the last two rows, and [the categorical chapter](../guide/13-categorical-features.md) for what bonsai's encoders do and do not match.

Every row is a snapshot of the current record, not a permanent verdict. A declined feature reopens when a workload makes its gap load-bearing, exactly as native categoricals did.
