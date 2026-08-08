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
| scaling | perf | friedman1 (perf-synthetic) | fit_s, predict_s, RSS | per-run `--out` | `python -m bonsai.bench.scaling` | 46 |
| gpu-tall / gpu-wide / gpu-extreme | perf | friedman1 (perf-synthetic) | ingest_s, train_s, RSS, VRAM, r² | `results/<axis>-<date>.jsonl` | `python -m bonsai.bench run --spec <axis>` | 103 |
| cpu-tall / cpu-wide | perf | friedman1 (perf-synthetic) | ingest_s, train_s, RSS, r² | `results/<axis>-<date>.jsonl` | `python -m bonsai.bench run --spec <axis>` | 103 |
| gpu-early-stop | perf | friedman1 (perf-synthetic) | fit_s ratio, time to stop | `results/gpu-early-stop-<date>.jsonl` | `python -m bonsai.bench run --spec gpu-early-stop` | 103 |

The Grinsztajn suite is the only citable standings table: its 55 tasks were selected by third parties (Grinsztajn, Oyallon, Varoquaux 2022), which removes the selection-bias objection a self-picked suite can never answer. The internal campaign (`scripts/compare.py`) and the one-off probes (`scripts/probe_*.py`) still run, but they publish nothing standing: their closed rounds are listed in [the archive](results/archive.md).

## Scenarios

The perf division measures five scenarios and one behavior, not a grid. A scenario is a single (rows x cols) cell chosen to make one claim sharply, and every scenario publishes the same four dimensions so a reader never re-orients: `ingest_s`, the fixed cost of turning raw floats into the library's training structure, paid once per dataset however many fits follow; `train_s`, the variable cost, paid per fit and scaling with trees and depth; peak host RSS with its headroom over the shared input array; and on GPU scenarios peak device memory, per process via NVML, never the device-wide total (a co-tenant's allocation is not a measurement). One fused `fit(X, y)` wall clock is published at the tall GPU scenario for completeness, sourced from the refresh's parity arm rather than measured separately, because fixed-plus-variable is the workflow the two-step API serves and the fused call is the one-shot convenience.

The scenarios: **gpu-tall** (16,777,216 x 128) and **gpu-wide** (131,072 x 16,384) are an iso-volume pair, the same 2^31 cells arranged to stress row-parallel and column-parallel machinery respectively, so their contrast isolates shape from size. **gpu-extreme** (16,777,216 x 1,024, 2^34 cells) is the largest cell the standings pod can generate: host float32 input costs four bytes per cell against bonsai's one-byte bins, and the pod class caps container memory at 188GB, so host generation binds long before bonsai's device footprint does. The scenario therefore measures capacity separation rather than bonsai's own ceiling; an arm that cannot fit publishes its OOM as a result, because capacity is a claim. **cpu-tall** (2,097,152 x 128) and **cpu-wide** (16,384 x 16,384) repeat the pair at 2^28 cells: the CPU story is an advantage demonstration at sizes a workstation actually runs, not a race at sizes only a GPU should run. **gpu-early-stop** measures the behavior latency-sensitive users pay for: the per-round cost of carrying an eval set at fixed iterations, and the wall clock to a stopped model under patience.

GPU is the primary division because it is the peak of the hardware, and the standings card is the RTX PRO 6000 Blackwell (96GB). Arms pair by growth strategy, the only fair pairing: depthwise against XGBoost, leafwise against LightGBM, levelwise against CatBoost, each on both planes. The panels page presents one grower at a time with its rival beside it and hardware never mixed within a table.

**Plane gating.** Every axis declares its plane, and the release gate accepts an axis without a fresh refresh when the sources that could move it are byte-unchanged since its last one: the cpu digest covers the tree outside `src/cuda`, the gpu digest covers everything, and a bumped reference-library major invalidates both. A CPU-only change re-measures no GPU axis and vice versa; the refresh driver's `--only-stale` runs exactly what the digests demand. This is decision 40's bit-identity discipline applied to scheduling: what provably did not change is not re-measured.

## Datasets

Tiers and provenance live in the machine-readable registry, `bonsai.bench.datasets` (`python -m bonsai.bench.datasets --list`): test-pin (CI-load-bearing, immovable), quality-external, quality-smoke, perf-scale, perf-synthetic. `tests/data/README.md` documents the on-disk layout.

## The synthetic dataset

Perf-division data is a generalized form of Friedman #1 (Friedman, "Multivariate Adaptive Regression Splines", 1991), the standard synthetic regression for tree methods: the classic five-feature target 10 sin(pi x0 x1) + 20 (x2 - 0.5)^2 + 10 x3 + 5 x4 repeated over the informative features in blocks of five with 0.6^b decaying block weights, uniform [0, 1) float32 features, and Gaussian noise with sigma = y.std() / 3, which places the best-achievable test r² near 0.9. Synthetic is kept for perf deliberately: rows, columns, and bins become free experimental axes (decision 46), and generation is deterministic in (seed, rows, cols) via SeedSequence, so bin and thread sweeps reuse byte-identical data. The implementation and full provenance: `bonsai.bench.synth`. A deliberately frozen linear variant lives in `scripts/model_hash.py`; its output feeds the cross-architecture byte-identity CI gate and must never be edited.

## Metrics

One implementation, `bonsai.bench.metrics`. Primary metric per task, the only one a claim may headline: r² (regression), AUC (binary), accuracy (multiclass), NDCG@10 (ranking). rmse and mae are recorded as secondaries.

## Timing

Two modes, declared per row. `in_memory`: fit timed from in-memory arrays, including each library's own ingest (bonsai binning, XGBoost QuantileDMatrix, lgb.Dataset, CatBoost Pool); the scaling and standings convention. `pipeline`: fit timed end to end including CSV read; the CLI-compare convention. Numbers from different modes are never compared against each other. predict_s always times prediction from a raw test matrix.

Every arm receives the same host array and is measured through the ingest API its own documentation recommends for that input: bonsai's `Dataset` then `train(pairs, dataset)`, XGBoost's `QuantileDMatrix`, LightGBM's `Dataset` built with its params, CatBoost's `Pool`. fit_s spans ingest through the trained model for every arm; ingest_s and train_s are reported underneath it rather than measured on their own, and fit_s is never redefined as their sum. Every arm reports the split, bonsai included. Each measured row builds its own ingest structure: reusing one across repeats or variants would amortize a cost each row must be charged in full.

bonsai's split is honest only because its `Dataset` takes a device hint (decision 99): built with `device="cuda"` it bins on the GPU, which is where the fused `train(pairs, X, y)` call bins for a cuda grower, so the two forms measure one pipeline and the seam between them is free to report. An unhinted `Dataset` bins on the host whatever grower follows it, and its ingest number then describes a pipeline no GPU arm runs; that is not hypothetical, it reached a published refresh once and was withdrawn. The standing evidence is a parity arm on the refresh pod: the anchor cell fit both ways, interleaved, banded at 5% on fit_s and on peak RSS, with a failure stopping the supersession rather than annotating it.

Because that shared array dominates every peak-host-RSS figure, the perf panels also carry headroom above it (rows plus the held-out test rows, both resident since `bonsai.bench.synth.gen_data` returns numpy views into one buffer, times columns, times 4 bytes float32), quoting bonsai's device memory (`dev_mem`) alongside since that headroom is host-input only and paid on the device instead, and collapsing under device-resident input (issue #289); the Grinsztajn suite reads real files through each library's own ingest path, so no single input-array size is shared across arms and no headroom column is printed for it.

Where a library sketches or bins, host or device, is a property of that library and is reported, not equalized: routing every arm through the same intermediate structure would erase the difference the perf division exists to measure. Changing any runner's call form is therefore a protocol change, not a refactor, and needs a same-pod A/B before it lands: a runner was once quietly moved off its documented ingest path, and the loss was caught only by a human comparing two refreshes and noticing every other arm got faster.

## Knobs

Two named sets in `bonsai.bench.params`: CAMPAIGN (200 iters, lr 0.05, depth 6, 255 bins) for quality, SCALING (100 iters, lr 0.1, depth 8) for perf. Two LightGBM leaf conventions exist by declaration, not drift: `num_leaves_campaign(depth)` = (1 << depth) - 1 and `num_leaves_full(depth)` = 1 << depth; each row records which. Reference mappings (including CatBoost's GPU border cap and XGBoost's hessian-weighted min_child_weight, whose two readings bracket XGBoost per decision 68's correction) live only in `params.py`; re-deriving them by hand caused a published correction once and is the one prohibited act.

Two translation caveats the matched-knobs design cannot remove, stated rather than hidden. On classification tasks the leaf-size floor is not equivalent across libraries: XGBoost's `min_child_weight` counts hessian mass (so 20 demands far more than 20 rows under logloss, and 1 demands fewer; the two published runs bracket the equivalent point without hitting it), while CatBoost's symmetric-tree policy supports no per-leaf floor at all and runs without one. And CatBoost's `border_count` counts splits where every other library's `max_bin` counts bins; the bins-1 fencepost is applied inside `catboost_core` (2026-07-30 fairness review, which found call-site translations had drifted three ways, including a one-bin shortfall for XGBoost and LightGBM in the airline suite).

## Early stopping

The early-stop suite measures two separate quantities on one cell, 4M rows by 100 columns at the SCALING knobs, and never mixes them into one number.

The first is per-round eval overhead. The same cell is fit twice at the same fixed iteration count, once with a validation set attached and once without, with no patience armed either time, and the ratio of the two `fit_s` values is what it costs a library to score a validation set every round. Both arms fit the reduced train side, so the ratio prices the eval work and not a row-count difference: the eval rows are carved off the train side (10 percent, the split the early-stopping guide already benchmarks) in the "off" arm too, and then thrown away.

The second is time to stop. Patience 50, an iteration cap of 2000 that the valid set rather than the cap is meant to end, and a learning rate of 0.05. The row carries `fit_s` (wall clock to a stopped model), `stopped_at`, and the test metric of the model that stopping actually produced. This is a latency in its own right: it is how long a production retrain takes when nobody has hand-tuned the round count.

The eval rows always come off the train side, so the test split stays untouched and the reported metric still lands on rows no library saw.

Five translation caveats the matched-knobs design cannot remove, stated rather than hidden.

`stopped_at` is the retained round count, the size of the model the reported metric came from, because the four libraries number their best iteration two different ways (bonsai and LightGBM report a count, XGBoost and CatBoost a zero-based index). The wall clock covers roughly `stopped_at` plus the patience, and those trailing rounds are counted deliberately: they are what the stop costs.

XGBoost is the only one of the four that does not truncate on a stop, and its default prediction range is every tree it grew, so the runner passes `iteration_range` explicitly. Read without it, the metric would describe the overshot model instead of the stopped one.

CatBoost shrinks the model to its best iteration whenever an eval set is present, detector armed or not: 200 requested iterations came back as a 140-tree model with no detector at all (CatBoost 1.2.10). The fixed-iteration arm therefore sets `use_best_model=False`, or it would report the metric of a model shorter than the one whose fit was timed.

Building the validation structure lands in `ingest_s` for the three reference libraries and in `train_s` for bonsai, which bins its eval set inside `train` with the Dataset's own mappers. The overhead ratio is quoted on `fit_s`, the outer wall clock, which has no such seam.

Eval cadence and loss definition stay each library's own. All four score every round at these settings, but each scores with its own objective's eval metric, so the per-round validation numbers are not comparable across libraries; the stopping point and the final test metric are.

## Hardware rules

Comparative perf numbers come only from the same machine in the same session (rental-fleet variance reaches ~25%). Rented hosts must pass the 30-second sync-latency probe (round-trips over 50µs reject the pod, decision 48). Quality rows are hardware-independent by construction: references run their CPU paths, bonsai models are bit-identical across architectures by contract.

A reference library that requests a cuda device and quietly trains on CPU instead is a measurement failure, not a slow row: on the same pod it poisons the comparison in whichever direction the fallback lands, and nothing about the resulting number looks wrong on its own (a rental once posted 90.6s at a cell that reads 6.9s elsewhere, entirely because XGBoost warned and kept going instead of raising). `run_xgb` asserts placement after the fit by reading `booster.save_config()`, which reports the device XGBoost actually trained on rather than the one requested, and raises instead of returning a row when the two disagree. LightGBM and CatBoost carry no comparable post-fit signal (their params echo the request, not the outcome) and both raise immediately, rather than falling back, when their GPU device is missing, so no equivalent guard exists for them.

Same-machine control is also what makes a competitor gap debuggable. Two apparent CatBoost advantages localized to bonsai bugs precisely because everything else was held equal: an accuracy gap that traced to a GPU kernel veto (decision 63), and a binning-cost gap that was a per-feature sampling pass CatBoost does not pay (decision 64, a 24x mapper speedup after the fix).

## The row schema

Schema v1 (`bonsai.bench.runlog`): every row carries `schema, ts, git_sha, division, suite, script, cmd, timing_mode, host` (with library versions), plus `knobs`/`knobs_hash` when a knob set applies; suite-specific fields (`cell`, `dataset`, `task`, `variant`, `seed`, `metric`, `value`, `status`, ...) ride alongside. Rows are append-only; files may mix schema generations; readers tolerate extra keys. Every published table must name its results file and the command that regenerates it.

## Standings and evidence (decision 92)

Every results file is one of two classes, and the class sets its lifecycle.

Evidence: the dated record behind a decision. Frozen forever; its claim carries its date and sha, so it cannot go stale. The probe files, the recheck files, and the campaign smoke are evidence.

Standings: the current claim on one published axis. The registry [`benchmarks/standings.json`](../../benchmarks/standings.json) lists one file per axis (gpu-tall, gpu-wide, gpu-extreme, cpu-tall, cpu-wide, gpu-early-stop, quality-grinsztajn, code) with the single sha its rows were measured at. Standings supersede in place by re-measurement, generalizing the code division's rule (decision 69): the tree has exactly one current state per axis. The ledger stamps every standings caption with the measured sha computed from the rows, so the reader always sees the vintage.

Two gates hard-fail, enforced by `scripts/check_standings.py`. Claim time: a decisions-log entry that claims a perf change on an axis carries a `Standings: <axis>` line, and `make docs-check` fails while any tagged entry is newer than the axis's registered state. Release time: the wheels publish job fails unless every axis was refreshed for exactly the version being released, bounding staleness at one release even for untagged changes.

The release gate has one mechanical exception, scoped to quality axes (a registry key starting with `quality`). A quality axis's slow clock, "refresh on engine quality changes or reference-library majors," used to be enforced by memory alone. It is now a check: the axis's committed `hash_set` (a digest over `scripts/model_hash.py` and the whole `src/`/`include/` implementation it drives) must still equal a fresh digest, and none of the axis's `refs` reference libraries may have bumped a major version since. Bit-identical CPU model-hash bytes cannot move quality standings by construction (decision 40's contract: CPU-only builds stay bit-identical), so an unmoved guard proves the axis current without spending pod time on it. Perf axes carry no such proof, since their numbers are wall-clock rather than bytes, and still require an exact `refreshed_for` match. `scripts/update_standings.py` stamps `hash_set` and `refs` whenever it supersedes a quality axis; the reference-library majors it records come from the installed environment when available and otherwise from the hand-maintained `benchmarks/reference_versions.json` ledger, since the bench extras are unpinned in `pyproject.toml` and there is no lockfile to read statically.

A related rule for the round that produces a claim, not the standings axis it lands on: an accept gate for a per-round or fixed-cost lever must be specified as absolute time, or evaluated at more than one scale, because a fixed saving shrinks as a fraction of the largest cell and a single-scale percentage gate systematically under-credits it.

Reader-facing prose never restates standings digits; only generated tables carry them, so every digit surface is behind a `--check`.

The refresh is one rented pod session, driven locally by `scripts/standings_refresh.py` (decision 96; section 10 of the RunPod runbook): a same-pod A/B of the previous release wheel against HEAD on anchor cells detects whether perf moved, then the standings specs re-measure the axes and the supersession lands as one reviewed PR.

Every axis carries a `plane` (cpu or gpu) and the digest of that plane's sources at its last refresh. A routine refresh runs `measure --only-stale`, which measures the axes whose plane digest has moved and leaves the rest alone, so a device-only change never pays for a CPU-plane sweep; a release refresh measures every axis on one host.

## Carry-forward stamps

Plane gating is mechanical, and mechanical rules are coarse. The gpu digest covers every host source too, so a change to host-side fill code invalidates a GPU axis whose device sources are byte-identical and whose thread count its spec pins. The digest says "something that could matter moved"; sometimes the tree says it could not have. A carry-forward stamp is the recorded way to say so. `python3 scripts/update_standings.py --axis <axis> --restamp-verified` moves the axis's `hash_set` to the current digest with no new measurement, leaving the results file, measured sha, host, date, and `refreshed_for` exactly as the last real run set them.

It is legitimate under two conditions together, and neither is a matter of taste. The changed sources must be unable to reach the axis: not "unlikely to move it", unable, by an argument you can state in one sentence about which code the axis executes. And the equivalence must be verified rather than assumed: run the check that would catch you being wrong, and record what it printed.

The entry records the argument as structured fields, in a `carried_forward` block: `measured_at` (the sha the data still comes from), `stamped_at` (the commit whose digest was carried onto it), `reason` (why the change cannot reach the axis), and `evidence` (the kind of proof and its two values, which must be equal; `model-hash` is the usual one). The tool refuses without all of them, refuses when the two evidence values differ, refuses on an axis that is already current, and refuses when the staleness comes from a reference-library major bump, which no argument about our own sources can answer.

The distinction is permanent by construction. The measured sha and results file never move, so the published caption still shows the real vintage; the release gate prints `carried-forward stamp` rather than `hash-unchanged skip`; and `--stale` prints a note naming the axis, its stamped commit, and its reason. A carry-forward is also spent: the next time the plane digest moves, the axis is stale again like any other, because the argument was about one specific change and not about the axis.

The rule this exists to protect: a carry-forward must never be used to dodge a refresh that is merely expensive. Cost is not evidence. If the honest answer is "the change probably did not move this number and the pod costs money", the axis is stale and gets measured.

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
