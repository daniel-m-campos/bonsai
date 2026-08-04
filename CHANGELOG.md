# Changelog

All notable changes to bonsai. Format loosely follows [Keep a Changelog](https://keepachangelog.com/); versions are git tags. Design rationale for anything below lives in [`docs/decisions.md`](docs/decisions.md).

## [Unreleased]

## [1.6.1] - 2026-08-03

The two-step workflow reaches the device, and the leafwise record is corrected.


### Added
- **Device-resident input** (issue #289, decision 102): CUDA arrays supporting DLPack (cupy, torch, jax; numba through `cupy.asarray`) are binned where they already live, so a GPU-native caller reaches a trained model without a host round trip. The bin mapper still cuts on a host sample, gathered on the device and downloaded as a compact block, so cuts stay bit-identical to the host path. Measured at 4M x 100 with data already resident: 1.30s to download and fit, 1.04s binning in place. CPU-grower models are byte-identical to the host path; `cuda_*` growers keep their tolerance-equal contract.
- **`Dataset` takes a device hint** (issue #288, decision 99): `bonsai.Dataset(X, y, ..., device="cuda")` bins on the device, so the two-step form (build once, train many times) reaches the ingest path the fused call has always used. Previously a prebuilt Dataset always binned on the host whatever grower followed it, costing 46% of fit time and 35% of peak host memory at 4M x 100. Parity with the fused call is measured at 4M and 16M, and a five-fit sweep over one device Dataset bins once instead of five times. The constructor also gained `n_threads`, which its binning never honored before.

## [1.6.0] - 2026-08-02

Leaf-wise growth moves to the GPU, and the device engine is symmetric at last.

### Fixed
- **`device="cuda"` honors the growth strategy.** The compat mapping substituted `cuda_depthwise` for `leafwise` (once "the nearest CUDA grower"; stale since `cuda_leafwise` landed), so a GPU fit silently trained depth-wise trees where the same config on CPU trained best-first ones, including the estimator default. The mapping is now a pure prefix: `device=` moves compute and never changes the model class. GPU users who want the previous behavior say `grower="depthwise"` explicitly (it is faster at matched knobs: 12.1s vs 13.8s at 16M x 100 in the 1.6.0 standings; the 22.6s against 30.8s this entry first quoted came from the handicapped harness decision 100 corrects).

### Added
- **The device-resident objective arms for `cuda_leafwise`** (issue #268, [`docs/architecture/20-cuda-leafwise.md`](docs/architecture/20-cuda-leafwise.md)): eligible leafwise GPU fits (MSE, LogLoss, or Poisson; no DART; row sampling off or Bernoulli) now keep labels and scores on the device for the whole fit, as depthwise and oblivious have since 1.4. Per tree that removes the gradient upload, the values and leaf-id downloads, and the host objective and score loops. Measured same-pod at 16M x 100 on an L40S: fit 30.2s to 24.6s, grow 20.0s to 13.1s, identical r2; it is the larger half of the closing ladder below. Both arms of that A/B ran on the harness decision 100 corrects, so the saving is the measured one and only the absolutes it starts from are handicapped. Arming is decided once per fit against the conservative bound (every feature, the full leaf budget), because a tree that declined to the host plane mid-fit would have no host gradients to fall back on; configs that clear only the per-tree bound run non-resident as before, and `BONSAI_HOST_OBJECTIVE=1` still forces the host path. CPU models are byte-identical.
- **The `cuda_leafwise` round stages once, and asynchronously** ([`docs/architecture/20-cuda-leafwise.md`](docs/architecture/20-cuda-leafwise.md)): best-first growth expands one leaf per round, so the round's fixed cost is the whole overhead, and it staged eight small pageable uploads. The fit-constant monotone vector now uploads once per tree instead of once per split find, the histogram kernels' (offset, count, slot) triple is one packed upload, and the rest moves to pinned host memory with asynchronous copies: four per round, none of them syncing. Worth a fixed 0.37s of grow per fit at these knobs, the same absolute saving at every scale because it is per round and not per row. Same-pod L40S, identical r2: 250k x 100 fit 2.27s to 1.88s (-17%), 1M 3.14s to 2.79s (-11%), 16M 20.69s to 20.39s (-1.8%, where 20s of compute dilutes it). Depthwise and oblivious are untouched. A third lever, overlapping the partition chain's copy-back and scan, was built, measured at no win outside the noise, and reverted; doc 20 and decision 98 carry the refutation.
- **`cuda_leafwise`** (issue #268, decision 97, [`docs/architecture/20-cuda-leafwise.md`](docs/architecture/20-cuda-leafwise.md)): best-first growth with histograms, partition, and split finding on the device. A per-tree histogram slot pool replaces the level plane's ping-pong, so a frontier of one is served without wasted memset: the root takes slot 0, each split builds the smaller child into the next free slot and derives the larger by in-place subtraction in the parent's, and the partition rewrites one segment in place. The grow loop keeps the gain heap, the depth cap, and every other decision, reaching the plane through a new `LeafStep` seam. Tolerance-equal to CPU `leafwise`; the dispatch grid goes from 105 to 126 cells. Admitted by same-pod ladder at 30.8s at 16M x 100 against LightGBM-CUDA's 32.4s, and shipping at 13.8s against 31.8s in the 1.6.0 standings once the two levers above and the harness correction of decision 100 are both in (see Measured).

### Measured
- **The closing device-leafwise ladder** (decision 98, [`leafwise-stage3-2026-08.jsonl`](benchmarks/results/leafwise-stage3-2026-08.jsonl)), **corrected 2026-08-03** (decision 100, [`leafwise-correction-2026-08.jsonl`](benchmarks/results/leafwise-correction-2026-08.jsonl)): the campaign's ladders were measured through a benchmark harness that binned on the host, so the numbers this entry first carried handicapped every bonsai CUDA arm. Re-measured on the fixed path, one L40S, three device arms, best of two reps, interleaved: `cuda_leafwise` fits 250k/1M/4M/16M x 100 in 3.3/4.2/7.2/18.7s against LightGBM-CUDA's 8.6/9.7/14.8/37.0s and bonsai's own resident `cuda_depthwise` at 0.8/1.5/4.4/15.8s, on the slowest of the three L40S rentals the campaign used (its LightGBM arm runs 16% to 29% above the campaign's). The 1.6.0 standings measure the 16M cell on a rental that reproduces those reference times, at 13.8s against 31.8s. Either way the plane is about 2x LightGBM's CUDA leaf-wise at 16M, where this entry first claimed 24.4s against 31.9s and a 1.3x margin. Uncapped at 16M (256 leaves, no depth cap, the regime where best-first differs) it reads 27.5s at r2 .8862 against LightGBM's 36.8s at .8858, where the campaign published 32.3s against 31.7s: the one cell the campaign recorded against bonsai inverts, and the plane leads by 34% at matched accuracy.

### Docs
- **[`docs/architecture/20-cuda-leafwise.md`](docs/architecture/20-cuda-leafwise.md)** is the plane's design and as-built record, from the slot-pool sketch through the stage 3 measurements, including two refuted levers (the partition chain, and small-node occupancy on the preserved `perf/leafwise-occupancy` branch) and the profiler-attribution correction that reordered the whole lever list. The GPU chapter introduces the three CUDA growers together and walks the leaf step beside the level transaction; `device="cuda"` is documented as moving compute, never the growth strategy.

## [1.5.4] - 2026-07-30

The benchmark harness becomes a tool, and its first campaign ships in the ledger.

### Added
- **The unified bench CLI**: `python -m bonsai.bench run --spec <name-or-path>` executes declarative cell specs (explicit cells or generators, variants, threads, per-device repeats, gates, timeout caps) over the existing worker protocol; `plan` prints the expansion without fitting, `variants` prints the registry, and spec-mode runs resume by default (finished rows skip, failures re-attempt). The `iso_volume` generator holds rows x cols constant and sweeps aspect ratio. Custom cell ladders no longer require source edits.
- **Campaign specs ship in the wheel**: the committed specs are package data under `bench/specs/`, `--spec` resolves a bare name against them (`python -m bonsai.bench run --spec iso-volume-2026-08` works from any install; a filesystem path still wins), and `python -m bonsai.bench specs` lists what ships.
- **Measured peak device memory**: a sampler thread (NVML, `nvidia-smi` fallback) records per-process and whole-device VRAM peaks into a `dev_mem` field on every GPU row, on error and OOM rows too, with the sampling interval recorded so rows never claim more precision than sampled. `--data-cache DIR` memoizes the synthetic generator as memmapped `.npy` files (bit-identical fits, verified).
- **One variant registry** (`bonsai.bench.variants`): canonical names, per-suite membership, and the historical alias spellings in one table; `params.bonsai_core` closes the last hand-built config path. Worker children report the reference-library versions they imported, so rows finally carry them.
- **A committed pod campaign driver** (`scripts/pod_bench_driver.sh`): clone-or-fetch, Blackwell-aware CUDA 12.8 side-install, build, run a committed spec, resume-safe.

### Measured
- **The iso-volume shape frontier** (decision 91, RTX PRO 6000 Blackwell 96GB): bonsai's CUDA growers fastest at every cell of the 2^31 and 2^33 constant-volume ladders, near-flat across the tall half where the references vary 1.5-2x. The instrument's first catches: XGBoost-GPU dies at 32k x 65536 having allocated 33.4GB of a 96GB card, and CatBoost-GPU reserves 90.2GB at every shape. At p ~ 2x n the oblivious grower holds test r2 .873 where depthwise-family growers fall to .815.

### Docs
- **The results ledger is per-suite pages**: the old URL is now a landing with perf-first division summaries linking one generated page per study; the README states each division in one table; navigation gains section landings and door cards.

## [1.5.3] - 2026-07-30

The wide-data wall falls on both processors, and the standings prove it.

### Changed
- **One tiled fill replaces the row-wise/feature-parallel pair** (decision 89): the binned mirror moved to a column-block-tiled layout (2048-feature blocks) and the histogram fill runs tiles outer, rows inner, so the live scatter target is one block's histograms at any width. In an interleaved same-pod A/B the tiled fill beat the row path at its best cell (326 vs 369s at 1M x 4096), beat feature-parallel at its best cell (442 vs 514s at 131k x 16384), and tied at 16M x 100. Models are bit-identical to 1.5.2 at every width, so the 24MB routing threshold and its cache-size follow-up are retired outright. Evidence: `benchmarks/wide-cpu-hist-2026-07.md`.
- **The bin mapper radix-sorts its cut subsample**: an LSD byte-radix on the order-preserving key transform replaces `std::sort` above 2k samples, cutting mapper fit from 2.85 to 2.12s at 131k x 4096 on an M2 (the mapper cost 11.5s of a 54.9s GPU fit at 16k features, decision 90's price list). Cuts and models are byte-identical; small columns keep `std::sort`.

### Measured
- **The CUDA wide wall was stale data** (decision 90): the recorded ~5x wide-GPU deficit dated to 2026-07-08 code, before the device-resident objective landed. On current main, same pod, bonsai's CUDA growers lead every wide cell against both CatBoost-GPU and XGBoost-GPU at 3-4x less host memory.
- **Six-variant cols re-baseline** (decision 90 follow-up): one pod, four cells from 1M x 100 to 131k x 16384; bonsai CUDA fastest at every width (50.5s vs 71.3/77.3s at the widest) and the wide standings now carry their chart in the ledger.

### Docs
- **Engine chapter E6 epilogue**: how the tiled mirror dissolves the wall the chapter builds, with the A/B that proved it.

## [1.5.2] - 2026-07-30

Wide data on CPU gets 2.6-5.8x faster, from the first production field report.

### Changed
- **Ultra-wide CPU fits route through the feature-parallel fill** (decision 88, issue #217): u8 levels whose selected-histogram footprint exceeds 24MB (more than ~12k features at 255 bins) stop using the row-wise fill, whose per-row scatter breaks every cache at that width. Same-pod at 131k x 16384, t=16: depthwise 1019s to 379s (LightGBM-CPU parity), leafwise 2591s to 445s (a 7x deficit collapses to 1.2x), at 2.7x less peak memory than LightGBM (18.8 vs 50.1GB). Narrow and mid-width fits are code-identical (fixed-input models byte-identical); above the threshold, models change bytes at identical accuracy and become bit-identical at any thread count. The threshold was set by an interleaved same-pod A/B; the cache-size-aware refinement is recorded on issue #217. Evidence: `benchmarks/wide-cpu-hist-2026-07.md`.

### Docs
- **Engine track chapter E6, the wide-data wall**: the two fills as one arithmetic with the scatter on opposite sides, the footprint table behind the cache cliff, and the two refuted theories on the way to the shipped constant.

## [1.5.1] - 2026-07-30

Package hygiene and benchmark fairness, driven by the first field reports from production use.

### Changed
- **The Python package is properly moduled**: `__init__.py` (849 lines of implementation) is now a ~50-line public surface over `estimators.py`, `_compat.py` (the XGBoost translation tables), and `_coerce.py`. Public API unchanged (same nine names); pickles from earlier versions load bit-identically. The lite design review behind it: `docs/reviews/2026-07-30-design-review-python.md`.
- **Benchmark harness fairness fixes** (adversarial review, `docs/reviews/2026-07-30-review-benchmark-fairness.md`): the CatBoost borders-vs-bins fencepost now lives in one place (`catboost_core`, guard-tested) after call-site translations drifted three ways; XGBoost and LightGBM no longer run one bin short of bonsai in the airline and gpu-pareto suites; grinsztajn pins bonsai to the references' thread count. A rented-GPU re-check of both airline knob shapes confirmed no published cell depended on the drift (deltas within ±0.0022 of zero, all cited leads stand). The protocol page now states the classification leaf-floor caveats.

### Docs
- **Running the benchmarks** (`docs/use/benchmarks.md`): the installed-wheel and source-tree paths side by side, all four suites documented (the airline ladder for the first time), and where result rows land.

## [1.5.0] - 2026-07-29

On PyPI as `bonsai-gbt`, and the XGBoost drop-in surface.

### Added
- **PyPI via trusted publishing**: `pip install bonsai-gbt` with no `--find-links`. Releases upload through OIDC (no stored tokens); the CUDA-enabled linux x86_64 wheel reaches the index only after the rented-GPU release gate passes, and a TestPyPI rehearsal path (workflow dispatch) proved the flow end to end before the first real upload.
- **The XGBoost drop-in surface**: the canonical `XGBRegressor`/`XGBClassifier` script runs with only the class name swapped. New aliases `min_child_weight` (`tree.min_child_hess`, the same minimum hessian mass per child), `gamma`, `subsample` (switches on the bernoulli sampler), and `device="cuda"` (picks the matching CUDA grower); XGBoost objective strings on both estimators (`reg:squarederror`, `reg:quantileerror` with `quantile_alpha`, `binary:logistic`, `multi:softprob`, ...); `eval_set` in the list-of-tuples form; `evals_result()`, `best_iteration`, and `best_score` backed by a per-round eval history (in-memory only, the model format is untouched; squared error is presented as rmse, the exact root); `save_model`/`load_model`/`apply`/`iteration_range`; `n_features_in_`. The full swap table and the deliberate differences: [Switching from XGBoost](docs/use/from-xgboost.md).

### Measured
- **XGBoost 3.3 recheck** (decision 87): every published standing survives against the 2026-07-21 release on a same-pod three-arm ladder; the one real competitor gain (wide-CPU histogram tiling, 2x at 1M x 4096) flips no published cell. The 16M host-memory ratio (22.1 vs 6.9GB) reproduced on a second host.

## [1.4.0] - 2026-07-19

GPU fits 15 to 25% faster through the device-resident objective, and the documentation rebuilt around results.

### Added
- **Device-resident objective for CUDA fits** (decision 77, issue #171): eligible fits (MSE, LogLoss, or Poisson; sample weights supported; no DART; row sampling off or Bernoulli) keep labels and scores on the GPU, derive each tree's gradients there, and fuse the score update into the tree epilogue, so per tree nothing crosses the bus in either direction. Measured same-pod: 20-25% faster fits at 16M rows, 15-16% at 64M, with identical accuracy (full-data resident models are bit-identical to the host-objective GPU models). Ineligible fits and `BONSAI_HOST_OBJECTIVE=1` take the unchanged host path; CPU models are byte-identical.
- **CUDA device selection** (issue #158): `parallel.device_id` places a `cuda_*` grower's whole fit (ingest and training) on a specific GPU, so multi-GPU hosts can pin concurrent fits to different cards without `CUDA_VISIBLE_DEVICES`. `0` is the default device and behavior-preserving; an out-of-range id, or a nonzero id on a CUDA-less build, raises `ConfigError`. Placement only: model bits are unaffected and the knob is not persisted in the model artifact. CPU growers ignore it (the `sampler.subsample` convention for regime-scoped knobs). First step of the single-node multi-GPU track (architecture doc 19).
- **Explicit bin edges on `Dataset`** (decision 73, architecture doc 18): `bonsai.Dataset(X, y, bin_edges={col: edges})` bins listed columns at user-supplied cut points (regulatory bands, clinical thresholds, incumbent-scheme parity) instead of fitted quantiles. The edges travel inside the model artifact like fitted cuts, so `predict`, `save`, and `load` work on raw values with no external transform; k edges give k+1 splittable bands plus the NaN-only missing bin. Unlisted columns fit exactly as before (byte-identical models when the argument is omitted).

### Docs
- **The site rebuilt around five doors** (issue #179): results-first standings with three drift-gated generated references (parameters, make targets, the GBT timeline), the Engine track teaching five real perf campaigns case-method, an honest What-to-use-when including competitor recommendations, CI-enforced style and CI-executed examples, and the Design core from system map to determinism contract.

### Fixed
- **Weighted fits no longer pay a serial per-tree weight multiply**: the host gradient weighting loop is now parallel (elementwise, bitwise-identical models); at 16M rows it was costing roughly 80ms per boosting round of single-threaded work on every weighted fit.
- **The missing bin is NaN-only on every fitting path** (decision 74, issue #155): fitted cuts now end with a `FLT_MAX` top-band closer, so finite values above the last cut (a stride path's top tail, a capped column's heavy maximum, rows beyond the bin sample) get a real splittable bin instead of training as missing, which also removes a train/predict routing skew for those rows. Models change only where a leak existed; capped-column synthetics recover up to half the lost variance, and on the re-validated Grinsztajn standings bonsai's mean rank improves 1.73 to 1.44 with 36 of 55 outright wins and no last-place finishes (evidence: `benchmarks/missing-bin-closer-2026-07.md`).


## [1.3.0] - 2026-07-14

GPU training from a 2.3MB pip install, and the benchmark harness ships in the package.

### Added
- **CUDA in the linux x86_64 wheel** (decision 70): GPU training out of the box on any NVIDIA driver R525+, SASS for sm_70 through sm_120 plus a compute_90 PTX forward-JIT floor, cudart statically linked. The whole backend costs 2.33MB of wheel (vs ~300MB for xgboost's GPU wheel) and behaves exactly like a CPU wheel on GPU-less machines. Every release's CUDA wheel is validated on rented GPU hardware before it attaches; the byte-identity model-hash gate now runs across all three wheel platforms on every build.
- **Runtime docker image**: `ghcr.io/daniel-m-campos/bonsai:cuda` with the CUDA wheel preinstalled, RunPod-ready (sshd entrypoint); the release gate boots this exact image, so the image and the wheel are validated together.
- **`bonsai.bench` in the wheel** (decision 69): `pip install bonsai-gbt[bench]` reproduces the published benchmark tables; `python -m bonsai.bench.grinsztajn out.jsonl --report` re-runs the external standings suite. Normative rules in the [benchmark charter](https://daniel-m-campos.github.io/bonsai/method/benchmark-protocol/).

### Fixed
- `Model.n_classes` reports 0 unless the model was trained with the softmax objective (binary classifiers no longer masquerade as multiclass).
- Linux wheels vendored a dynamic libomp while claiming static linkage; the workflow now documents the vendoring honestly (issue #134).

### Changed
- CLI `fit` no longer runs the binning pass on validation sets (issue #119): per-iteration eval reads features and labels only, so the binned data had no readers and the pass was pure waste in every early-stopping run.

## [1.2.0] - 2026-07-13

Install without a toolchain, reuse binning across fits, and a round of classifier correctness fixes surfaced by an adversarial post-release review.

### Added
- **Python 3.9 support**: `requires-python` lowered to 3.9 (full binding suite verified on CPython 3.9.25); ruff now pins `target-version = py39` so newer-only syntax can't creep in.
- **Prebuilt wheels on GitHub Releases**: Linux x86_64/aarch64 (`manylinux_2_35`, Ubuntu 22.04+/Debian 12+) and macOS arm64, Python 3.9–3.13. `pip install` the wheel with no LLVM/CMake; libc++ is vendored into the wheel, OpenMP statically linked. Every wheel is smoke-tested in a clean venv (fit, `predict_proba`, `Dataset`, `save`/`from_file`) before it ships. CPU-only; GPU training remains a source build.
- **Reusable pre-binned `bonsai.Dataset`**: bin once, train many: `ds = bonsai.Dataset(X, y); bonsai.train(params, ds)` skips the per-fit bin pass across a hyperparameter search or CV loop, bit-identical to fitting from `(X, y)`. On GPU the resident-matrix upload-skip cache now fires across fits (decision 54). Bin settings are sealed at construction: `bin_mapper.*` overrides are rejected whether they arrive as params or inside a config file (decision 65).
- **Multiclass `predict_proba`**: `(n, K)` row-wise softmax probabilities; completes `BonsaiClassifier` (the 1.1.0 follow-up).
- **xgboost/lightgbm-style constructor aliases** on both estimators: `n_estimators`, `num_leaves`, `random_state`, `n_jobs`, `reg_lambda`, `reg_alpha`, `max_bin`, `min_child_samples`, `colsample_bytree`.
- `Model.objective_name` / `Model.n_classes` read-only properties.

### Fixed
- **Multiclass `sample_weight` was silently ignored**: the softmax gradient/hessian loop never applied per-row weights (the single-output booster did); a weighted 3-class fit was bit-identical to unweighted. Weights now scale grad/hess; unweighted fits are unchanged bit-for-bit.
- **`BonsaiClassifier.from_file` crashed `predict`/`predict_proba`**: class metadata was only set by `fit`. Restored from the saved model as encoded ids `0..K-1` (xgboost's `load_model` convention; pickle preserves original label values), and non-classifier models are rejected instead of mislabeled. The first fix also imported `tomllib` (3.11+ stdlib) and broke on Python 3.10; now dependency-free via the new `Model` properties.
- **`eval_set` labels absent from the training classes now raise** instead of being silently mis-encoded; previously they corrupted the validation metric and fired early stopping tens of iterations early with no error. NaN labels are rejected like sklearn.

### Docs
- README: superseded RTX-5090 benchmark table and duplicated scaling history removed; stale facts corrected (test count, dispatch-combo count).

## [1.1.0] - 2026-07-13

The crown-week release: measured parity-or-better against xgboost, lightgbm, and catboost, plus a scikit-learn-shaped Python surface.

### Added
- **`BonsaiClassifier`**: sklearn-style classifier over the engine's `logloss` (binary) and `softmax` (multiclass) objectives. Binary `predict_proba`; multiclass `predict` returns labels (multiclass `predict_proba` is a tracked follow-up). Arbitrary label types are encoded/decoded via `classes_`.
- **scikit-learn estimator compatibility** for `BonsaiRegressor`/`BonsaiClassifier`: `get_params`/`set_params`/`score`, and drop-in use in `clone`, `Pipeline`, `GridSearchCV`, `cross_val_score`, and `pickle`, implemented **without a scikit-learn runtime dependency** (`import bonsai` never imports sklearn).
- **`sample_weight`** on `BonsaiRegressor.fit` / `bonsai.train` (sklearn convention): per-row weighting of gradients and hessians.
- **`OrderedTargetEncoder`**: leak-free ordered target statistics for categorical features, including `cross=2` pair encodings (decision 58; guide chapter 13).
- **Poisson** regression objective (closes #44).

### Changed / Performance
- **Binning: one shared row sample for the whole matrix** instead of a per-feature reservoir pass: 24× faster mapper-fit at 16M, quality-neutral (decision 64).
- **CPU fill loop software prefetch**: the 16M-row fit now ties xgboost-hist (decision 61).
- **Fresh same-pod re-baseline** (decision 62–64): bonsai's GPU `oblivious` grower now **edges catboost and beats xgboost-GPU at 16M** at matched accuracy and ~3× less host memory, and holds the fastest slot at every row scale.

### Fixed
- **GPU `oblivious` grower** carried a split-selection defect (a missing port of the issue-#60 fix) that silently cost ~0.011 test r² at depth ≥ 5; now matches its CPU twin exactly (decision 63).
- **Cross-architecture bit determinism**: models are byte-identical across arm64/x86-64 at a fixed thread count, with no floating-point contraction on the host plane (decisions 59–60), enforced by a cross-arch CI gate.
- **OpenMP build variance can no longer be silent**: a missing OpenMP is now a hard configure error, not a quiet serial fallback that changed model bits (decision 60).
- Quality-campaign correctness fixes: count-weighted cuts for heavy-value columns (#63/decision 57), one cut per distinct value on duplicate-heavy columns (#61), infeasible frontier nodes contribute zero gain rather than vetoing the split (#60), and the true diagonal softmax hessian p(1−p) (#62).

### Docs
- New guide material (categoricals, structure-vs-scheme) and a **claims-and-proofs** table in the README linking every performance/quality claim to a reproducible run and the decision that records it.
- Note: `pyproject.toml` had drifted to `0.6.0` after the `v1.0.0` tag; corrected to `1.1.0` here.

## [1.0.0] - 2026-07-11

First 1.0 release: histogram gradient-boosted trees with a C++23 core, three growers (depthwise / leafwise / oblivious), CPU and CUDA backends, a concept-checked component API, a CLI, and Python bindings.
