# The benchmark protocol

This page is the normative charter behind every published bonsai number. The rules exist because cloud hardware is not a controlled instrument, and a benchmark without its raw data is an anecdote. Reproduce the headline table yourself:

```
pip install bonsai-gbt[bench]
python -m bonsai.bench.grinsztajn out.jsonl
python -m bonsai.bench.grinsztajn out.jsonl --report
```

## Divisions

Every result row belongs to exactly one division, and both run what MLPerf would call closed: matched knobs across libraries, no per-model tuning.

**quality**: accuracy claims. The metric is primary; timing may be recorded but is never citable from a quality row. **perf**: latency, throughput, and memory claims. Every row declares its `timing_mode`; accuracy is recorded only as a sanity guard. A third division, [code](#the-code-division), measures the bonsai tree itself and is self-only by construction.

The current evidence, rendered whole from every committed results file, is [the results ledger](results.md); this page is the rules it follows.

## Suites

| suite | division | datasets (tier) | primary metric | results file | runner | decision |
|---|---|---|---|---|---|---|
| grinsztajn | quality | 55 external tasks (quality-external) | r² / AUC | `results/grinsztajn-2026-07.jsonl` | `python -m bonsai.bench.grinsztajn` | 68 |
| campaign | quality | internal ten (quality-smoke) | per-task | `results/quality-campaign-2026-07.jsonl` | `scripts/compare.py` per config | 56 |
| probes | quality | per-study | per-study | `results/<probe>-<date>.*` | `scripts/probe_*.py` | 57, 58, 67, ... |
| scaling | perf | friedman1 (perf-synthetic) | fit_s, predict_s, RSS | per-run `--out` | `python -m bonsai.bench.scaling` | 46 |
| airline | perf | benchm-ml airline 0.1m/1m/10m (perf-external) | fit_s, AUC | `results/airline-2026-07.jsonl` | `python -m bonsai.bench.airline` | issue #154 |
| rebaseline | perf | friedman1 | fit_s, r² guard | `results/rebaseline-2026-07.jsonl` | scaling runner, rows axis | 62 to 64 |
| iso-volume | perf | friedman1 (perf-synthetic) | fit_s, dev_mem, r² guard | `results/iso-volume-2026-08.jsonl` | `python -m bonsai.bench run --spec iso-volume-2026-08` | 91 |

The Grinsztajn suite is the only citable standings table: its 55 tasks were selected by third parties (Grinsztajn, Oyallon, Varoquaux 2022), which removes the selection-bias objection a self-picked suite can never answer. The internal campaign remains the fast local regression check.

## Datasets

Tiers and provenance live in the machine-readable registry, `bonsai.bench.datasets` (`python -m bonsai.bench.datasets --list`): test-pin (CI-load-bearing, immovable), quality-external, quality-smoke, perf-scale, perf-synthetic. `tests/data/README.md` documents the on-disk layout.

## The synthetic dataset

Perf-division data is a generalized form of Friedman #1 (Friedman, "Multivariate Adaptive Regression Splines", 1991), the standard synthetic regression for tree methods: the classic five-feature target 10 sin(pi x0 x1) + 20 (x2 - 0.5)^2 + 10 x3 + 5 x4 repeated over the informative features in blocks of five with 0.6^b decaying block weights, uniform [0, 1) float32 features, and Gaussian noise with sigma = y.std() / 3, which places the best-achievable test r² near 0.9. Synthetic is kept for perf deliberately: rows, columns, and bins become free experimental axes (decision 46), and generation is deterministic in (seed, rows, cols) via SeedSequence, so bin and thread sweeps reuse byte-identical data. The implementation and full provenance: `bonsai.bench.synth`. A deliberately frozen linear variant lives in `scripts/model_hash.py`; its output feeds the cross-architecture byte-identity CI gate and must never be edited.

## Metrics

One implementation, `bonsai.bench.metrics`. Primary metric per task, the only one a claim may headline: r² (regression), AUC (binary), accuracy (multiclass), NDCG@10 (ranking). rmse and mae are recorded as secondaries.

## Timing

Two modes, declared per row. `in_memory`: fit timed from in-memory arrays, including each library's own ingest (bonsai binning, XGBoost QuantileDMatrix, lgb.Dataset, CatBoost Pool); the scaling and rebaseline convention. `pipeline`: fit timed end to end including CSV read; the CLI-compare convention. Numbers from different modes are never compared against each other. predict_s always times prediction from a raw test matrix.

## Knobs

Two named sets in `bonsai.bench.params`: CAMPAIGN (200 iters, lr 0.05, depth 6, 255 bins) for quality, SCALING (100 iters, lr 0.1, depth 8) for perf. Two LightGBM leaf conventions exist by declaration, not drift: `num_leaves_campaign(depth)` = (1 << depth) - 1 and `num_leaves_full(depth)` = 1 << depth; each row records which. Reference mappings (including CatBoost's GPU border cap and XGBoost's hessian-weighted min_child_weight, whose two readings bracket XGBoost per decision 68's correction) live only in `params.py`; re-deriving them by hand caused a published correction once and is the one prohibited act.

Two translation caveats the matched-knobs design cannot remove, stated rather than hidden. On classification tasks the leaf-size floor is not equivalent across libraries: XGBoost's `min_child_weight` counts hessian mass (so 20 demands far more than 20 rows under logloss, and 1 demands fewer; the two published runs bracket the equivalent point without hitting it), while CatBoost's symmetric-tree policy supports no per-leaf floor at all and runs without one. And CatBoost's `border_count` counts splits where every other library's `max_bin` counts bins; the bins-1 fencepost is applied inside `catboost_core` (2026-07-30 fairness review, which found call-site translations had drifted three ways, including a one-bin shortfall for XGBoost and LightGBM in the airline suite).

## Hardware rules

Comparative perf numbers come only from the same machine in the same session (rental-fleet variance reaches ~25%). Rented hosts must pass the 30-second sync-latency probe (round-trips over 50µs reject the pod, decision 48). Quality rows are hardware-independent by construction: references run their CPU paths, bonsai models are bit-identical across architectures by contract.

Same-machine control is also what makes a competitor gap debuggable. Two apparent CatBoost advantages localized to bonsai bugs precisely because everything else was held equal: an accuracy gap that traced to a GPU kernel veto (decision 63), and a binning-cost gap that was a per-feature sampling pass CatBoost does not pay (decision 64, a 24x mapper speedup after the fix).

## The row schema

Schema v1 (`bonsai.bench.runlog`): every row carries `schema, ts, git_sha, division, suite, script, cmd, timing_mode, host` (with library versions), plus `knobs`/`knobs_hash` when a knob set applies; suite-specific fields (`cell`, `dataset`, `task`, `variant`, `seed`, `metric`, `value`, `status`, ...) ride alongside. Rows are append-only; files may mix schema generations; readers tolerate extra keys. Every published table must name its results file and the command that regenerates it.

## Standings and evidence (decision 92)

Every results file is one of two classes, and the class sets its lifecycle.

Evidence: the dated record behind a decision. Frozen forever; its claim carries its date and sha, so it cannot go stale. The probe files, the recheck files, and the campaign smoke are evidence.

Standings: the current claim on one published axis. The registry [`benchmarks/standings.json`](../../benchmarks/standings.json) lists one file per axis (rows, width, shape, frontier, airline, quality-grinsztajn, code) with the single sha its rows were measured at. Standings supersede in place by re-measurement, generalizing the code division's rule (decision 69): the tree has exactly one current state per axis. The ledger stamps every standings caption with the measured sha computed from the rows, so the reader always sees the vintage.

Two gates hard-fail, enforced by `scripts/check_standings.py`. Claim time: a decisions-log entry that claims a perf change on an axis carries a `Standings: <axis>` line, and `make docs-check` fails while any tagged entry is newer than the axis's registered state. Release time: the wheels publish job fails unless every axis was refreshed for exactly the version being released, bounding staleness at one release even for untagged changes.

A related rule for the round that produces a claim, not the standings axis it lands on: an accept gate for a per-round or fixed-cost lever must be specified as absolute time, or evaluated at more than one scale, because a fixed saving shrinks as a fraction of the largest cell and a single-scale percentage gate systematically under-credits it.

Reader-facing prose never restates standings digits; only generated tables carry them, so every digit surface is behind a `--check`.

The refresh is one rented pod session, driven locally by `scripts/standings_refresh.py` (decision 96; section 10 of the RunPod runbook): a same-pod A/B of the previous release wheel against HEAD on anchor cells detects whether perf moved, then the standings specs re-measure the axes and the supersession lands as one reviewed PR.

## Amendments

Evidence files are append-only: committed rows are never edited or regenerated, and corrections are banner annotations plus a decisions-log entry (the pattern of decisions 48, 63, and 68). Standings files supersede in place instead (the standings section above). Superseded artifacts of either class are deleted from the tree; git history is the archive.

## The code division

The code division measures bonsai itself, self-only: no comparative claim against any other library is made or implied. It exists so the readable-core statement is falsifiable; a claim about code you can read must come with counts you can check.

The tool is [lizard](https://github.com/terryyin/lizard), pinned as `uvx lizard@1.23.0` and run by `scripts/measure_complexity.py`. Per plane it records file count, LOC (`wc -l`), lizard NLOC, function count, and mean and max cyclomatic complexity (CCN). Results land in `results/code-metrics-2026-07.jsonl` and render into [the code division page](results/code-metrics.md); the meta row carries the tool version and the measured git SHA.

| plane | contents |
|---|---|
| core_headers | `include/bonsai/**` except `cuda/` and `cli/` |
| engine_impl | `src/**` except `cuda/`, `cli/`, `python/` |
| cuda_plane | `src/cuda/**`, `include/bonsai/cuda/**` |
| bindings_cli | `src/python/**`, `src/cli/**`, `include/bonsai/cli/**`, `python/bonsai/**` except `bench/` |
| bench_tooling | `python/bonsai/bench/**`, `scripts/*.py`, `benchmarks/*.cpp` |
| tests | `tests/**`, `python/tests/**` |

Non-claims, stated once: LOC alone is not quality, and a small number is not an argument. The numbers describe this tree at one SHA and nothing else.

The five highest-CCN functions across the core planes are published by name. Naming our own worst functions is deliberate; an offender list its author curates away is marketing, not measurement.

Re-measurement supersedes the results file in place (decision 69, generalized to all standings by decision 92): a new run at a new SHA replaces it, and git history is the archive. The append-only rule of the other divisions does not apply; the tree has exactly one current state, so there is nothing to append.
