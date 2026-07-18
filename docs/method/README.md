# Results

Every performance and accuracy claim on this site comes from a committed benchmark run, on named hardware, at matched settings. This page states what we measure and gives the headline numbers. [The results ledger](results.md) holds the full evidence, generated from every committed results file.

## What we measure

Two divisions, kept apart. Quality rows make accuracy claims, where the metric is primary and timing is never citable. Perf rows make speed and memory claims, where accuracy is recorded only as a sanity guard. The [benchmark protocol](benchmark-protocol.md) is the normative charter: divisions, suites, metrics, timing modes, and the row schema.

The only citable standings table is the Grinsztajn suite: 55 tasks selected by third parties. That removes the selection-bias objection a self-picked suite cannot answer.

## Three headline numbers

All three are GPU, same-pod, at 16M rows.

- **Speed.** On the 16M GPU frontier bonsai reaches every measured accuracy first, at every horizon: the marginal boosting round costs 64ms against catboost's 78 on the same pod, and the fixed cost is 3.8s against catboost's 11.7 ([the frontier](results.md#gpu-accuracy-vs-time-frontier-at-16m)).
- **Memory.** Peak host RSS at 16M is 7.0GB against xgboost's 22.2GB and catboost's 19.4GB, roughly 3x less. Predict is about 3x faster ([the ledger](results.md#perf-division)).
- **Determinism.** Models are bit-identical across CPU architectures and thread counts, enforced per commit in CI. No reference library offers this ([the contract](../design/determinism.md)).

Losses are recorded with the wins. On wide data catboost keeps the lead, with bonsai second. xgboost holds the last 0.001 r² of cut quality on some tasks. Both are in [the ledger](results.md).

## How we decide

Two rules govern every change: price a change before you build it, and admit a feature only when measurement earns its place. Both are on [how we decide](how-we-decide.md), with the episodes that earned them. The raw feed behind every number is the [decisions log](../decisions.md).
