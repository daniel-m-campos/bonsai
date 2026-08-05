# Results

Every performance and accuracy claim on this site comes from a committed benchmark run, on named hardware, at matched settings. This page states what we measure and lists the claims with their proofs.

<div class="grid cards" markdown>

-   **[The results ledger](results.md)**

    ---

    Division summaries first, then one generated page per study.

-   **[The benchmark protocol](benchmark-protocol.md)**

    ---

    The normative charter: divisions, suites, metrics, timing modes.

-   **[How we decide](how-we-decide.md)**

    ---

    Price before building; admit features only by measurement.

-   **[What to use when](what-to-use-when.md)**

    ---

    The honest decision table, competitors named where they win.

</div>

## What we measure

Two divisions, kept apart. Quality rows make accuracy claims, where the metric is primary and timing is never citable. Perf rows make speed and memory claims, where accuracy is recorded only as a sanity guard. The [benchmark protocol](benchmark-protocol.md) is the normative charter: divisions, suites, metrics, timing modes, and the row schema.

The only citable standings table is the Grinsztajn suite: 55 tasks selected by third parties. That removes the selection-bias objection a self-picked suite cannot answer.

## Claims and proofs

Every claim links a reproducible run and the decision that records it; the point of a small, measured library is that you can check it.

| Claim | Evidence |
|---|---|
| **Bit-identical models across CPU architectures** (arm64 == x86-64) at a fixed thread count; no reference library offers this | decisions [59/60](../decisions.md); asserted per-commit by [`cross-arch.yml`](../../.github/workflows/cross-arch.yml) via [`scripts/model_hash.py`](../../scripts/model_hash.py); [the contract](../design/determinism.md) |
| **Best mean rank on the 55-task Grinsztajn benchmark under either min_child_weight convention** (36 outright wins; second-or-better on 50/55, never last) | [the standings page](results/quality-grinsztajn.md), [grinsztajn-2026-07](../../benchmarks/grinsztajn-2026-07.md), decision 68 |
| **Fastest GPU slot at every row scale**; at 16M `levelwise` edges CatBoost and beats XGBoost-GPU at matched accuracy | [fit at scale](results/perf-scale.md), [scale-edge](../../benchmarks/catboost-scale-edge-2026-07.md), decisions 62 to 64 |
| **Fastest at every measured width and aspect ratio**, with measured device memory that sizes to the problem | [width and shape](results/perf-shape.md), decisions 90 and 91 |
| **The only GBT whose GPU path ships in a 2.3MB pip install, validated on live GPU hardware per release** | decision 70; [`wheels.yml`](../../.github/workflows/wheels.yml) |
| **Within ~8% of XGBoost-hist at 16M rows on CPU, host-dependent**: a dead tie on one pod, XGBoost ahead on another | decision 61; [fit at scale](results/perf-scale.md) |
| **Best library on 9 of 10 datasets of the internal quality campaign** | [campaign smoke](results/quality-campaign.md), decisions 56 to 57 |
| **Categorical parity with CatBoost within the chance band**, via preprocessing not an engine feature | decision 58; [categorical-tradeoff](../../benchmarks/categorical-tradeoff-2026-07.md); [`encoding.py`](../../python/bonsai/encoding.py) |
| **~3x less host memory than XGBoost at 16M** (7.0 vs 22.2GB) and ~3x faster predict | [fit at scale](results/perf-scale.md) |
| **Ranking is a measured, scoped gap**: ~+0.015 NDCG@10 to a listwise loss, not pairwise LambdaRank | [ranking-tradeoff](../../benchmarks/ranking-tradeoff-2026-07.md); [`probe_ranking.py`](../../scripts/probe_ranking.py) |
| **Every feature earns its place by measurement**; refutations are recorded too | [how we decide](how-we-decide.md); the declined probes on [the probes page](results/quality-probes.md) |

Losses are recorded with the wins, and so are their reversals: the wide-data GPU lead CatBoost held in the July 8 study flipped to bonsai by the July 30 recheck (decision 90). XGBoost holds the last 0.001 r² of cut quality on some tasks. Both are in [the ledger](results.md).

## How we decide

Two rules govern every change: price a change before you build it, and admit a feature only when measurement earns its place. Both are on [how we decide](how-we-decide.md), with the episodes that earned them. The raw feed behind every number is the [decisions log](../decisions.md).
