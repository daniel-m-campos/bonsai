# The benchmark protocol

This page is the normative charter behind every published bonsai number; [Benchmarks you can trust](benchmarking.md) explains why the rules exist, this page states what they are. Reproduce the headline table yourself:

```
pip install bonsai-gbt[bench]
python -m bonsai.bench.grinsztajn out.jsonl
python -m bonsai.bench.grinsztajn out.jsonl --report
```

## Divisions

Every result row belongs to exactly one division, and both run what MLPerf would call closed: matched knobs across libraries, no per-model tuning.

**quality**: accuracy claims. The metric is primary; timing may be recorded but is never citable from a quality row. **perf**: latency, throughput, and memory claims. Every row declares its `timing_mode`; accuracy is recorded only as a sanity guard.

## Suites

| suite | division | datasets (tier) | primary metric | results file | runner | decision |
|---|---|---|---|---|---|---|
| grinsztajn | quality | 55 external tasks (quality-external) | r² / AUC | `results/grinsztajn-2026-07.jsonl` | `python -m bonsai.bench.grinsztajn` | 68 |
| campaign | quality | internal ten (quality-smoke) | per-task | `results/quality-campaign-2026-07.jsonl` | `scripts/compare.py` per config | 56 |
| probes | quality | per-study | per-study | `results/<probe>-<date>.*` | `scripts/probe_*.py` | 57, 58, 67, ... |
| scaling | perf | friedman1 (perf-synthetic) | fit_s, predict_s, RSS | `results/scaling.jsonl` | `python -m bonsai.bench.scaling` | 46 |
| gpu_msd | perf | year_msd (perf-scale) | fit_s | `results/gpu_msd.jsonl` | `scripts/bench_gpu.py` | 41 |
| rebaseline | perf | friedman1 | fit_s, r² guard | `results/rebaseline-2026-07.jsonl` | scaling runner, rows axis | 62 to 64 |

The Grinsztajn suite is the only citable standings table: its 55 tasks were selected by third parties (Grinsztajn, Oyallon, Varoquaux 2022), which removes the selection-bias objection a self-picked suite can never answer. The internal campaign remains the fast local regression check.

## Datasets

Tiers and provenance live in the machine-readable registry, `bonsai.bench.datasets` (`python -m bonsai.bench.datasets --list`): test-pin (CI-load-bearing, immovable), quality-external, quality-smoke, perf-scale, perf-synthetic. `tests/data/README.md` documents the on-disk layout.

## The synthetic dataset

Perf-division data is a generalized form of Friedman #1 (Friedman, "Multivariate Adaptive Regression Splines", 1991), the standard synthetic regression for tree methods: the classic five-feature target 10 sin(pi x0 x1) + 20 (x2 - 0.5)^2 + 10 x3 + 5 x4 repeated over the informative features in blocks of five with 0.6^b decaying block weights, uniform [0, 1) float32 features, and Gaussian noise with sigma = y.std() / 3, which places the best-achievable test r² near 0.9. Synthetic is kept for perf deliberately: rows, columns, and bins become free experimental axes (decision 46), and generation is deterministic in (seed, rows, cols) via SeedSequence, so bin and thread sweeps reuse byte-identical data. The implementation and full provenance: `bonsai.bench.synth`. A deliberately frozen linear variant lives in `scripts/model_hash.py`; its output feeds the cross-architecture byte-identity CI gate and must never be edited.

## Metrics

One implementation, `bonsai.bench.metrics`. Primary metric per task, the only one a claim may headline: r² (regression), AUC (binary), accuracy (multiclass), NDCG@10 (ranking). rmse and mae are recorded as secondaries.

## Timing

Two modes, declared per row. `in_memory`: fit timed from in-memory arrays, including each library's own ingest (bonsai binning, xgboost QuantileDMatrix, lgb.Dataset, catboost Pool); the scaling and rebaseline convention. `pipeline`: fit timed end to end including CSV read; the gpu_msd and CLI-compare convention. Numbers from different modes are never compared against each other. predict_s always times prediction from a raw test matrix.

## Knobs

Two named sets in `bonsai.bench.params`: CAMPAIGN (200 iters, lr 0.05, depth 6, 255 bins) for quality, SCALING (100 iters, lr 0.1, depth 8) for perf. Two lightgbm leaf conventions exist by declaration, not drift: `num_leaves_campaign(depth)` = (1 << depth) - 1 and `num_leaves_full(depth)` = 1 << depth; each row records which. Reference mappings (including catboost's GPU border cap and xgboost's hessian-weighted min_child_weight, whose two readings bracket xgboost per decision 68's correction) live only in `params.py`; re-deriving them by hand caused a published correction once and is the one prohibited act.

## Hardware rules

Comparative perf numbers come only from the same machine in the same session (rental-fleet variance reaches ~25%). Rented hosts must pass the 30-second sync-latency probe (round-trips over 50µs reject the pod, decision 48). Quality rows are hardware-independent by construction: references run their CPU paths, bonsai models are bit-identical across architectures by contract.

## The row schema

Schema v1 (`bonsai.bench.runlog`): every row carries `schema, ts, git_sha, division, suite, script, cmd, dataset, task, variant, seed, knobs, knobs_hash, metric, value, timing_mode, host` (with library versions), and `status`. Rows are append-only; files may mix schema generations; readers tolerate extra keys. Every published table must name its results file and the command that regenerates it.

## Amendments

History is append-only: committed rows are never edited or regenerated. Corrections are banner annotations on the evidence document plus a decisions-log entry (the pattern of decisions 48, 63, and 68). Superseded artifacts are deleted from the tree; git history is the archive.
