# Project retrospective: bonsai

> **Companions:** [proposal](proposal.md), [architecture/](architecture/), [decisions log](decisions.md).
>
> **This document is a dated snapshot** of the project at MVP completion
> (git tag `mpcs-submission`). The claims about missing parallelism, two
> growers, and the speed gap were true then and are preserved as the
> point-in-time record — every performance number below is historical.
> For what changed since, see the addenda at the bottom; for current
> numbers, the [README performance section](../README.md#performance) and
> the committed runs in [`benchmarks/results/`](../benchmarks/results/)
> are the living source of truth.

## Summary

`bonsai` is a histogram gradient-boosted tree library in C++23 with a CLI front-end (`fit` / `predict` / `eval` / `bench` / `info` / `params`), TOML configuration, and a single-source-of-truth dispatch table over the cartesian product of objectives × growers × samplers. The MVP ships 8 dispatch combos (2 × 2 × 2), 204 unit tests, microbenchmarks, and a reference-library comparison harness. RMSE parity with xgboost / lightgbm on the regression workload is the strongest correctness signal; the fit / predict speed gap is the obvious next milestone and has one named cause (single-threaded).

## What was built

- **Spine:** `Dataset`, `BinMapper(s)`, `Histogram`, `Booster<O,G,Sa>`, `DenseTree` / `ObliviousTree`, `DepthwiseGrower` / `ObliviousGrower`, `HistogramNode/LevelSplitFinder`, `MSEObjective` / `LogLossObjective`, `AllRowsSampler` / `BernoulliSampler`.
- **Dispatch:** [`include/bonsai/registry/typelists.hpp`](../include/bonsai/registry/typelists.hpp) enumerates `Objectives`, `Growers`, `Samplers`. `cartesian_product_t` gives `Configurations`. `make_booster()` and `io::save/load_booster` share a `with_combo_matching(disp, cb)` helper that walks the product once and dispatches to the matching `Booster<O,G,Sa>`.
- **Config / IO:** declarative TOML sections in [`config/sections/`](../include/bonsai/config/sections/) (each new section is one row); model envelope is nlohmann/json + msgpack with a `k_format_version` discriminator. Config and model round-trip through `bonsai params --dump-config` and `bonsai fit --model out.msgpack`.
- **Benchmark harness:** Catch2 microbenches in [`benchmarks/`](../benchmarks/) (split finders, samplers, tree predict); a Python sidecar in [`scripts/compare.py`](../scripts/compare.py) trains all four bonsai variants and xgboost / lightgbm / catboost on the same TOML, emitting markdown + JSON reports.

## Performance

### Reference comparison: Year Prediction MSD

463,715 train / 51,630 test rows, 90 timbre features, regression on release year (1922–2011). 200 boosting iterations, max_depth = 8, learning_rate = 0.05, lambda_l2 = 1.0, max_bin = 255. Single thread for all libraries.

| library                       |   rmse |  fit (s) | predict (s) |
|-------------------------------|-------:|---------:|------------:|
| bonsai (depthwise, all_rows)  | 8.9911 |     73.4 |       0.561 |
| bonsai (depthwise, bernoulli) | 9.1873 |     66.2 |       0.560 |
| bonsai (oblivious, all_rows)  | 9.1745 |     58.2 |       0.313 |
| bonsai (oblivious, bernoulli) | 9.3247 |     52.0 |       0.316 |
| xgboost                       | 8.9849 |      8.6 |       0.027 |
| lightgbm                      | 8.9891 |     13.9 |       0.098 |
| catboost                      | 9.1441 |      7.6 |       0.009 |

**Headline:** bonsai depthwise matches xgboost / lightgbm RMSE to 0.07% (8.9911 vs 8.9849–8.9891). bonsai oblivious lands in catboost's neighborhood (9.17 vs 9.14), both oblivious-tree implementations. The RMSE alignment by algorithm family, not just by library, is the single piece of evidence that the implementation does what it claims.

**Fit speed:** bonsai is 6–10× slower than the externals. The single named cause is that bonsai is single-threaded; all three reference libraries run multi-threaded histogram builds by default. The speed gap is not algorithmic (depthwise + all_rows at 73 s on one core is in the expected range), and `ParallelBackend` (deferred, see below) is the lever that closes it.

**Predict speed:** 10–20× slower than xgboost. Two effects: same single-thread story, plus xgboost has a binned-uint8 predict path bonsai doesn't. bonsai oblivious is ~2× faster than depthwise on predict (0.31 vs 0.56 s); branch-free traversal pays off on a 51k × 200-tree workload.

### Reference comparison: California Housing (smaller dataset)

16,512 train / 4,128 test rows, 8 features. Same hyperparameters as above except max_depth = 6, min_data_in_leaf = 20.

| library                       |   rmse | fit (s) | predict (s) |
|-------------------------------|-------:|--------:|------------:|
| bonsai (depthwise, all_rows)  | 0.4738 |   0.224 |       0.045 |
| bonsai (depthwise, bernoulli) | 0.5384 |   0.235 |       0.045 |
| bonsai (oblivious, all_rows)  | 0.5364 |   0.159 |       0.016 |
| bonsai (oblivious, bernoulli) | 0.5793 |   0.168 |       0.016 |
| xgboost                       | 0.4739 |   0.283 |       0.002 |
| lightgbm                      | 0.4735 |   0.843 |       0.006 |
| catboost                      | 0.5147 |   0.278 |       0.001 |

Same RMSE-by-family alignment as on MSD. At this size, the wall-clock predict numbers are dominated by CSV parsing and process startup; the microbenchmark (next section) measures the actual traversal cost.

### Predict-path refactor: variant to flat Node

Effect of [`b7fb149`](../include/bonsai/tree.hpp): replacing `std::variant<InternalNode, LeafNode>` (24-byte node, tag check per step) with a flat tagged struct (20-byte node, sentinel `feature_id == k_leaf_flag`).

**Microbenchmark** (`bonsai_bench [tree]`, 30 samples, mean per call, synthetic random trees grown by `DepthwiseGrower`):

| workload                  |   variant |     flat |     Δ |
|---------------------------|----------:|---------:|------:|
| 1k rows × 4f × depth 4    |    9.4 µs |   7.8 µs | −17% |
| 10k rows × 8f × depth 6   |    164 µs |   145 µs | −12% |
| 100k rows × 16f × depth 8 |   2.56 ms |  2.28 ms | −11% |
| 10k rows × 8f × depth 10  |    401 µs |   360 µs | −10% |

**End-to-end on Year MSD** (only DenseTree-predict columns; ObliviousTree is the unchanged-code control):

| library                       | variant predict | flat predict |     Δ |
|-------------------------------|----------------:|-------------:|------:|
| bonsai (depthwise, all_rows)  |         0.583 s |      0.561 s | −3.8% |
| bonsai (depthwise, bernoulli) |         0.593 s |      0.560 s | −5.6% |
| bonsai (oblivious, all_rows)  |         0.326 s |      0.313 s | (noise) |
| bonsai (oblivious, bernoulli) |         0.314 s |      0.316 s | (noise) |

End-to-end gain (4–6%) is smaller than the microbench (10–17%) because CSV parse, model load, and CSV write dominate the wall-clock predict step on a 51k-row dataset. The oblivious rows confirm the run-to-run noise floor (~4%) since `ObliviousTree`'s code path is unchanged. Full reproduction recipe in [`benchmarks/results/year_prediction_msd_predict_perf.md`](../benchmarks/results/year_prediction_msd_predict_perf.md).

The assembly difference: node stride 24 → 20 bytes (line `mul *, #0x14` vs `#0x18`) and the variant tag load (`ldr w15, [x14, #0x14]; cbz w15`) is gone, one fewer load on the critical path per node visited.

### Split finder microbenchmarks

`HistogramNodeSplitFinder` (per-node, used by depthwise growing) and `HistogramLevelSplitFinder` (per-level, used by oblivious growing). Synthetic histogram workloads:

| bench                                  | samples | mean    |
|----------------------------------------|--------:|--------:|
| find: 8 features × 64 bins             |      30 |  1.79 µs |
| find: 64 features × 128 bins           |      30 |  29.4 µs |
| find: 256 features × 256 bins          |      30 |   235 µs |
| find: 1024 features × 512 bins         |      30 |  1.90 ms |
| level find: 4 parents × 8 × 64 bins    |      30 |  7.33 µs |
| level find: 16 parents × 64 × 128 bins |      30 |   486 µs |
| level find: 64 parents × 256 × 256 bins |     30 |  22.1 ms |

The split finder is the hot path of fit. The per-level finder costs roughly the per-node finder × parents; the savings of oblivious growing come from the smaller frontier overall (one cut shared per level vs one cut per node), not from a cheaper per-call.

### Sampler microbenchmarks

`AllRowsSampler` (`std::iota` over row indices) vs `BernoulliSampler` (`std::bernoulli_distribution` per row at probability `p`):

| bench                       | samples | mean    | rows/s   |
|-----------------------------|--------:|--------:|---------:|
| all_rows: 1k                |      20 | 315 ns  |   3.2 G/s |
| all_rows: 100k              |      20 | 419 ns  |   239 G/s (memcpy-bound) |
| all_rows: 1M                |      20 | 1.15 µs |   870 M/s |
| bernoulli p=0.5: 1k         |      20 | 408 ns  |   2.5 M/s |
| bernoulli p=0.5: 100k       |      20 | 26.7 ms |   3.7 M/s |
| bernoulli p=0.5: 1M         |      20 | 273 ms  |   3.7 M/s |
| bernoulli p=0.8: 100k       |      20 | 16.5 ms |   6.1 M/s |

Bernoulli is ~150× slower per row than iota because `std::bernoulli_distribution` is RNG-heavy (internally calls `uniform_real_distribution`). A future optimization could replace it with a `uniform_int_distribution<uint32_t>` threshold compare, but sampling is called once per boosting iteration over the full row set, not per-row in the hot path, so it isn't currently a fit-time bottleneck (the per-iter cost is dominated by histogram building across all kept rows, not by selecting them).

## Design highlights

**Closed type system + typelist dispatch.** The biggest payoff in the design. Each new objective / grower / sampler is two edits in the common case: append to the matching `TypeList` and specialize `impl_name<T>`. The cartesian-product dispatch table, the `bonsai info` listing, and the `TEMPLATE_LIST_TEST_CASE` parity tests all expand automatically; adding the second sampler grew the test count from 191 to 204 without any test file edits. The `all_named_v<TypeList>` gate fires at the typelist edit site if `impl_name<T>` is missing, so consumers don't need their own assertions.

**External name trait.** Keeping `impl_name<T>` outside the impls (vs. as a static member) keeps the impls dispatch-agnostic and lets the same type carry different names in different dispatch tables should that ever be needed.

**Concepts as contracts.** `Objective`, `Sampler`, `TreeGrower`, `Tree`, `NodeSplitFinder`, `LevelSplitFinder`: each spells out the extension requirement in the type system. The `BernoulliSampler` work required loosening `Sampler` from static-only to instance-based; the change was a few lines and the rest of the system caught up automatically through the concept.

## What I underestimated

**Config and I/O.** The architectural thought required for the config section descriptors and the model JSON envelope is substantial, easily the largest single chunk of design work after the GBT algorithm itself. The current design (declarative per-section tuples, NLOHMANN macros behind a typelist-driven dispatch helper) is clean enough that adding a sampler with a runtime parameter required exactly: one new struct, one section descriptor, one NLOHMANN line, and a format-version bump. But getting to that point ate more design iterations than the typelist machinery did. **JSON glue still couples to specific tree types** (`tree_to_json(DenseTree const&)` vs `(ObliviousTree const&)`); a reflection-aware design would push that into a per-type trait.

## What I deferred and why

**Parallelism.** The proposal called for OpenMP + `std::execution` backends with measurable speedup. Shipped: zero. The choice was deliberate: adding parallelism early would have locked the algorithm loops into a threading model before I understood the data dependencies in (a) per-node histogram building, (b) per-level frontier expansion in the oblivious grower, and (c) the prediction loop. I preferred to land feature breadth first (second grower, second sampler, predict-path refactor) so the eventual `ParallelBackend` design can be informed by what's actually in the codebase rather than what was imagined at proposal time. That said: the fit-speed gap to the reference libraries is essentially entirely this deferral. It is the single biggest lever remaining.

## C++23 over C++26 trunk

The proposal flagged a C++26 reflection branch for the registry (P2996). Skipped on purpose: I prefer a stable compiler over chasing trunk gcc / clang. The reflection seams are documented in [`sections/all.hpp`](../include/bonsai/config/sections/all.hpp) and [`internal/field_name.hpp`](../include/bonsai/config/internal/field_name.hpp) so future work can drop in `nonstatic_data_members_of(^Cfg)` without restructuring the surrounding layers.

## What's next

- `ParallelBackend` concept + first impl (OpenMP). Closes the fit / predict gap; the proposal target of "measurable speedup on multi-core hardware."
- Binned-uint8 predict path. Closes the remaining predict gap; model-format change so it needs a `k_format_version` bump.
- `GOSSSampler`. The interesting sampler: needs gradient / Hessian access during sampling, which the `Sampler` concept already passes through.
- Categorical features. The canonical demonstration of the extension API per the proposal.

---

## Addendum (2026-07)

Everything the retrospective named as "the next milestone" has since
landed, in four tagged pushes (see `git tag -n`):

- **`v0.2.0` — performance parity** (decisions 31–33): leafwise grower,
  OpenMP parallelism behind a one-function seam (bit-identical to serial
  at any thread count), parallel CSV/binning, ordered gradients, stable
  scatter, node-totals hoisting. YearPredictionMSD fit went 73s → 12–28s.
- **`v0.3.0` — feature parity** (decisions 34–35): feature subsampling,
  GOSS (whose benchmark exposed and fixed a latent stale-score bug in all
  subsampled training), early stopping, L1, MAE/Huber/quantile objectives,
  monotone + interaction constraints, DART. Each feature was A/B'd with
  the same knob enabled in xgboost/lightgbm/catboost; tables live in
  [feature_gap.md](https://github.com/daniel-m-campos/bonsai/blob/main/docs/feature_gap.md).
- **`v0.4.0` — Python bindings** (decision 36): nanobind module,
  sklearn-style `BonsaiRegressor`, models interchangeable with the CLI,
  static libomp so bonsai coexists with xgboost/lightgbm in one process.
- **Feature importance** (decision 37): split-count + gain, CLI and
  Python surfaces.

Standing on YearPredictionMSD as of that addendum (200 iters, in-process
"native" timing like the references; `benchmarks/results/msd_native.*` —
historical; see the second addendum below for where things stand now):

| library | rmse | fit_s | predict_s |
|---|---|---|---|
| bonsai (depthwise) | 8.9911 | 28.3 | 0.063 |
| bonsai (leafwise) | 9.0871 | 12.3 | 0.081 |
| xgboost | 9.1357 | 5.7 | 0.017 |
| lightgbm | 9.0826 | 7.7 | 0.105 |
| catboost | 9.1441 | 8.4 | 0.014 |

With early stopping enabled for everyone, all five libraries converge to
RMSE 8.96–9.00 and bonsai leafwise lands between xgboost and lightgbm.
The dispatch grid is 5 objectives × 3 growers × 3 samplers = 45 combos,
316 C++ tests + a Python binding suite. Remaining backlog:
[feature_gap.md](https://github.com/daniel-m-campos/bonsai/blob/main/docs/feature_gap.md) rows 10–17.

## Addendum (2026-07-13, v1.2.0)

The month after the first addendum closed the speed gap entirely and then
some; decisions 42–66 in the [decisions log](decisions.md) narrate it.

- **CUDA backend, device-resident training** (decisions 42–54): two GPU
  growers (`cuda_depthwise`, `cuda_oblivious`) with histograms, rows, and
  split finding on the device; kernels compiled by the project's own clang
  (`-x cuda`, same C++23, same libc++, no nvcc). Models trained on GPU
  predict everywhere.
- **The current score, same pod, matched settings** (decisions 62–64;
  [`benchmarks/results/rebaseline-2026-07.jsonl`](../benchmarks/results/rebaseline-2026-07.jsonl)):
  at 16M rows bonsai `cuda_oblivious` fits in 18.4s vs catboost-GPU 18.5s
  (both 0.876 test r²) and xgboost-GPU 19.9s, at ~3× less host memory
  (7.0 vs 19–22GB); bonsai holds the fastest GPU slot at every row scale
  tested. On CPU, bonsai ties xgboost-hist at 16M (75.8 vs 75.7s) and
  beats lightgbm at scale and on wide data. Catboost keeps the wide-data
  GPU lead (1024–4096 columns); xgboost keeps a +0.001 r² cut-quality
  edge (decision 55) — both recorded, not argued with.
- **Quality campaign** (decisions 56–57): best library on 9 of 10 real
  datasets against all three references at matched settings.
- **Bit-exact determinism as a contract** (decisions 59–60): models are
  byte-identical across CPU architectures (arm64 == x86-64) and thread
  counts, enforced per-commit by a cross-arch CI gate. No reference
  library offers this.
- **Python surface grown to parity where it matters**: `BonsaiClassifier`
  (binary + multiclass with `predict_proba`), `sample_weight`,
  sklearn-compatible estimators with no sklearn runtime dependency,
  xgboost-style constructor aliases, a reusable pre-binned `Dataset` for
  hyperparameter sweeps (decision 65), leak-free ordered target statistics
  as preprocessing (decision 58).
- **Distribution**: prebuilt wheels on GitHub Releases for Linux
  x86_64/aarch64 and macOS arm64, Python 3.9–3.13, no toolchain required
  (decision 66); every wheel smoke-tested in a clean venv before upload.
- The dispatch grid is now 7 objectives × 5 growers × 3 samplers = 105
  combos; 498 C++ tests plus the Python binding suite.

Remaining measured gaps, tracked openly: listwise ranking (~+0.015
NDCG@10, scoped in the ranking study), catboost's native categorical
machinery on categorical-heavy data (stage-2 designs exist, admission
gated on measurement), and the wide-data GPU lead above.
