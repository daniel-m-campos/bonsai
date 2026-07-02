# Proposal — bonsai: a modern C++ gradient boosted trees library

> **Historical document** — the original proposal, kept as written. Several
> non-goals (Python bindings, DART, competitive speed) were later delivered;
> the audit trail is [decisions.md](decisions.md) and the living tracker is
> [feature_gap.md](feature_gap.md).

> **Status:** Draft proposal. Serves as a stable design reference for future-me / collaborating agents. Companion docs: `docs/context.md` (briefing), `docs/architecture/` (per-component design docs), `docs/decisions.md` (decisions log).

## Goals and non-goals

**Goals**

- A working, from-scratch histogram-based gradient boosted trees (GBT) library in C++23, supporting regression on numeric features as the MVP path. A logloss objective and binary-classification path land alongside MSE as far as they can be exercised by unit tests and stress the architecture without pulling in a second reference dataset; full parity work for classification is Phase 4.
- Predictive parity (within tolerance) with xgboost / LightGBM / CatBoost on matched hyperparameters on one regression dataset for the MVP. Additional datasets — including a classification dataset for AUC parity — are added in Phase 4. Only one reference dataset is debugged at a time.
- Two parallel backends — OpenMP and `std::execution` — both implemented and benchmarked, with measurable speedup on multi-core hardware.
- An extension API broad enough to express the conceptual surface of all three reference libraries: pluggable growers (depth-wise / leaf-wise / oblivious), splitters (histogram / exact), samplers (uniform / GOSS / bernoulli), objectives (MSE / logloss / ...).
- A modern-C++ showcase: concepts as component contracts, parameter-pack + fold-expression dispatch in place of macros, static polymorphism in hot paths, and an optional C++26 reflection branch for the registry.
- A CLI-first interface (`bonsai fit`, `bonsai predict`, ...) with TOML configuration, modeled on CatBoost's argument conventions.

**Non-goals**

- Beating xgboost / LightGBM on training speed. They have years of low-level tuning; this project will not match them.
- Production readiness, ABI stability, or distributable binaries.
- Python or R bindings (CLI is the only interface).
- GPU support, distributed training, DART, Random Forest, ranking objectives.
- Full parity / golden / benchmark coverage of binary classification in the MVP. The objective and the prediction path may be implemented as far as unit tests can carry them, but a *second* live reference dataset is not debugged in parallel with the regression path.
- Categorical features in the MVP. Deferred to Phase 4 and treated as the canonical demonstration of the extension API.
- Multiclass classification in the main path.

---

## 1. Application idea and justification

bonsai is a histogram-GBT library written ground-up in C++23, optimized for clarity and architectural cleanness rather than raw throughput. The MVP is a regression library benchmarked end-to-end against one reference dataset. A logloss objective and binary-classification prediction path are written in parallel where they cost little and exercise the architecture (a second `Objective` keeps the concept honest, and exposes any places where the design accidentally hard-codes regression). Full classification parity — matching AUC against reference libraries on a real dataset — is held until Phase 4 so that only one reference dataset is being debugged at a time. The CLI follows CatBoost conventions (`bonsai fit --data train.csv --objective mse --rounds 500`).

The honest case beyond pedagogy:

- **Reference value.** The production GBT libraries are large (tens of thousands of lines), heavily macro-driven, and difficult to read straight through. A small, deliberately-shaped implementation (~3–5 KLOC) with thorough documentation has standalone value as a teaching artifact for anyone learning how histogram GBTs work end-to-end.
- **Architectural improvement.** The reference libraries dispatch through virtual functions in their inner training loops because their plugin sets are open at runtime. bonsai closes the plugin set at compile time and dispatches statically inside the training loop, with one runtime decision at the configuration boundary. A microbenchmark with-and-without virtualization on the hot path is a real result, not just a stylistic preference.
- **Maintainability via concepts.** Component contract violations ("you forgot `compute_gradients`") are caught at compile time with readable error messages, rather than at runtime through a registry lookup or via a pure-virtual abstract base. This is a measurable maintainability win over the reference libraries' approach, and a direct showcase of C++20/23 concepts.

What this proposal explicitly does *not* claim: training-speed parity with the reference libraries, or production suitability. The contribution is pedagogical clarity plus a clean architectural treatment of static-vs-dynamic dispatch in a domain where production libraries default to dynamic everywhere.

---

## 2. Survey of reference libraries

| Aspect | XGBoost | LightGBM | CatBoost |
|---|---|---|---|
| Public C++ API | Yes, broad (`include/xgboost/`) | Yes, broad (`include/LightGBM/`) | No — C API + CLI only |
| Top-level entity | `Learner`, `DMatrix` | `Boosting`, `Dataset` | (not exposed) |
| Configuration | JSON / `map<string,string>` | Typed `Config` struct | CLI flags / JSON |
| Plugin registry | `DMLC_REGISTER_*` macros (string-keyed factories) | `Create*Function(name, cfg)` factories | Not externally extensible |
| Tree growth | Depth-wise + loss-guided | Leaf-wise (best-first) | Oblivious (symmetric) |
| Split finding | Exact, approximate, histogram | Histogram (with GOSS / EFB) | Histogram on oblivious trees |
| Categorical handling | External encoding required | Native partition-based | Native ordered target stats |
| Iteration model | `UpdateOneIter` | `TrainOneIter` | (internal) |
| Determinism | Configurable | Configurable | Default |
| CLI style | `xgboost train.conf key=val` | `lightgbm config=... task=train` | `catboost fit -f train.tsv ...` |
| Bindings | Python, R, JVM, Julia | Python, R, JVM | Python, R |

**What's common across all three.** A `Dataset`-like type owns binning and storage layout (the user does not poke raw arrays during training); plugins (objectives, metrics, growers) are looked up by string at runtime; config flows as data, not as constructor arguments; train / predict / save / load is the universal verb set; iteration-level granularity (`UpdateOneIter`) is exposed so callbacks can live outside the core; the C API is the stable ABI, and the C++ classes are "internal-ish".

**Where they diverge.** Tree-growth strategy is the most visible axis, but the deeper divergence is in the *extension model*. XGBoost and LightGBM both expose registries with macro-based self-registration; CatBoost does not expose extension at all. None of them exploit static polymorphism in the inner loop, even though the choice of objective and grower is fixed for the duration of training.

bonsai borrows from all three: the typed `Config` struct from LightGBM, the CLI convention from CatBoost, the iteration-level training loop from XGBoost. The architectural twist is the static-dispatch inner loop, which none of them do.

---

## 3. Core architecture and design approach

### 3.1 Key entities

| Entity | Role | Extension surface |
|---|---|---|
| `Dataset` | Owns column-major, pre-binned feature storage; labels; weights. Constructed once from a reader. | Closed. Layout and invariants are load-bearing for the rest of the library. |
| `BinMapper` | Per-feature bin boundaries; maps raw values to `uint8`/`uint16` bins. | Closed. |
| `Histogram` | Per-node, per-feature gradient/hessian sums per bin. Supports the parent − sibling subtraction trick. | Closed. |
| `Objective` | `compute_gradients(scores, labels) → (grad, hess)`, plus link function and metric defaults. | **Open** (concept). MSE is the MVP parity target. Logloss lands alongside MSE to keep the concept honest under unit tests; full classification parity is Phase 4. Quantile / huber later. |
| `TreeGrower` | Builds one tree given gradients, splitter, sampler. | **Open** (concept). Depth-wise first; leaf-wise + oblivious in Phase 4. |
| `SplitFinder` | Given a node's histograms, returns the best `(feature, threshold, gain)`. | **Open** (concept). Histogram-based first; exact in Phase 4. |
| `Sampler` | Picks rows / columns for the current tree. | **Open** (concept). Uniform first; GOSS, bernoulli later. |
| `ParallelBackend` | Wraps `parallel_for` / `parallel_reduce` so the same algorithm runs against OpenMP, `std::execution`, or serial. | **Open** (concept). Three impls. |
| `Booster<Obj, Gr, Sp, Sa, Backend>` | Class template. Holds trees, scores, gradients. `update_one_iter()` is the unit of progress. | Single concrete class for now; abstract base only if DART/RF are added (non-goals). |
| `Registry<Base, Config, Impls...>` | Compile-time typelist with fold-expression dispatch from a string to a typed factory call. | Add a new impl = list it in the typelist. |

### 3.2 Performance-sensitive surfaces

What needs to be right from day one (refactors here cascade):

- **Column-major, pre-binned storage in `Dataset`.** Histograms scan one feature across all rows; row-major storage destroys cache. Per-feature bin width (`uint8` if `max_bin <= 255`, else `uint16`) via `std::variant` saves ~50% memory and bandwidth in the common case.
- **Float storage, double accumulators.** Labels, scores, gradients in `float`; histogram sums and split-gain scoring in `double`. Matches xgb / lgbm; halves bandwidth with no measurable accuracy loss.
- **`std::span` returns from `Dataset`.** No vector copies in the hot path.
- **Histogram subtraction.** `sibling = parent − sibling` halves histogram build cost. Implemented from the start.
- **Deterministic parallel reductions at fixed thread count.** Per-thread local histograms, *not* atomic adds. Atomic FP adds are bit-unstable even at fixed `n_threads`. Cross-thread-count bit-exactness is not promised — predictions match within tolerance, model bytes may differ. Matches XGBoost / LightGBM / CatBoost field consensus; see decisions §7.

### 3.3 What must be iron-clad from the beginning

- `Dataset` layout and invariants. Cannot be re-architected mid-project without rewriting the histogram, split, and grower code.
- The `Objective` / `TreeGrower` / `SplitFinder` / `Sampler` concept signatures. Once the static dispatch boundary is wired through `Booster`, changing a concept ripples through every implementation.
- The `Config` schema and the runtime → static dispatch boundary. Adding a parameter is cheap; restructuring the section layout is not.
- The determinism contract. Same seed + same data + **same thread count** must produce the same model bytes; cross-thread-count runs match within numerical tolerance. Designed for from day one; retrofitting is impractical. See decisions §7.

### 3.4 Static vs dynamic polymorphism

The architectural centerpiece of the project. The rule:

> **Dynamic at the configuration boundary, static everywhere inside.**

The CLI parses TOML and decides — at runtime, exactly once per process — which `Objective`, `TreeGrower`, `SplitFinder`, and `Sampler` to use. From that point on, the chosen types are baked into a `Booster<Obj, Gr, Sp, Sa, Backend>` template instantiation, and every "virtual" call inside the training loop is a direct, inlinable call.

The components are expressed as C++20/23 concepts, not abstract base classes:

```cpp
template <typename T>
concept Objective = requires(T obj, std::span<const float> scores,
                             std::span<const float> labels,
                             std::span<float> grad, std::span<float> hess) {
    { obj.compute_gradients(scores, labels, grad, hess) } -> std::same_as<void>;
    { T::name } -> std::convertible_to<std::string_view>;
};
```

Concepts give better diagnostic messages than abstract base classes ("type X does not satisfy concept `Objective` because member `compute_gradients` is missing"), and they enforce the contract without forcing a vtable.

**Cost.** The cartesian product of concrete component types instantiates into ~15–90 `Booster` variants depending on how many growers / objectives / samplers exist at the time. At the scale of this project that is acceptable; the compiler dedupes substantial portions, and binary size is not a constraint. xgboost / LightGBM cannot do this because their plugin sets are open across third-party binaries; bonsai's are not, by design.

**Open question, resolved in `docs/architecture/6-dispatch.md`.** The exact mechanism for turning four runtime strings into one statically-typed `Booster<...>` without writing a four-deep nested-lambda tower. Candidates: a flat dispatch table keyed on a tuple of names, or a builder that resolves one component at a time and threads typed state through. This is deliberately deferred — it is a mechanical question that should not block proposal sign-off.

### 3.5 Metaprogramming approach

Three places where the project leans on template metaprogramming, each chosen because the alternatives are visibly worse:

1. **Concepts as component contracts.** Every extensible component is a concept, not an abstract class. Compile-time enforcement, no vtable in the hot path.
2. **Typelist registries with fold-expression dispatch.** Each registry is `Registry<Base, Config, Impls...>`; the `create(name, cfg)` method is a fold expression over the impl list that compares against each impl's `static constexpr std::string_view name`. No macros, no static-init, no linker dead-code-stripping problems. Adding an impl = adding a name to one typelist.
3. **`Booster` as a class template.** Once dispatch picks the four concrete component types, `Booster<Obj, Gr, Sp, Sa, Backend>` is instantiated and the training loop is fully monomorphized.

**Optional reflection branch.** As a stretch, the typelist registry will be re-implemented using C++26 static reflection (P2996) on a separate branch with gcc-trunk. The reflection version "discovers" implementations via reflection over the namespace rather than by listing them in a typelist. This is a showcase only; the typelist version is the production path. The reflection path is contingent on compiler stability and is not on the critical path.

### 3.6 Iteration model

`Booster::update_one_iter()` is the unit of progress: build one tree, update the running scores, compute one set of metrics. The top-level training loop is then trivially:

```cpp
for (int i = 0; i < cfg.booster.rounds && !stop.should_stop(); ++i) {
    booster.update_one_iter();
    if (i % cfg.metric.eval_freq == 0) on_eval(booster);
}
```

This puts logging, snapshotting, early stopping, and progress-bar updates *outside* the core, where they belong. It also makes unit testing trivial ("step the booster one round, assert this invariant").

### 3.7 Configuration

TOML, with sections that map 1-to-1 to component instances:

```toml
[objective]
name = "mse"

[tree]
grower = "depthwise"
max_depth = 6
max_leaves = 31
lambda_l2 = 1.0

[sampler]
name = "uniform"
row_fraction = 1.0

[split]
finder = "histogram"

[parallel]
backend = "openmp"
```

CLI flags override TOML using dotted keys (`--tree.max-leaves 63`). Resolution order: struct defaults → TOML file → CLI flags. Last write wins. Strict parsing — unknown keys are an error, not silently ignored.

Config is parsed into nested strongly-typed structs (`TreeConfig`, `ObjectiveConfig`, ...) with defaults at the struct level. Component-specific parameters live with the component (`top_rate` is in `[sampler]`, not global). Validation happens in component constructors and throws `ConfigError` with a key path. There is no `extra` map for "future-proofing".

---

## 4. Testing approach

Catch2 v3 (`Catch2WithMain`), four layers:

| Layer | Examples | What it catches |
|---|---|---|
| Unit | `test_bin_mapper`, `test_histogram`, `test_objective`, `test_tree_grower`, `test_registry`, `test_config_parse` | Per-component correctness, edge cases, registry lookup behavior, TOML round-tripping. `test_objective` covers both MSE and logloss gradients/hessians against analytical values from the MVP — the second objective is what catches regression-only assumptions in the `Objective` concept. |
| Integration | `test_train_predict`, `test_save_load`, `test_determinism` | End-to-end training on tiny datasets; serialization round-trip; same seed + fixed thread count → identical model bytes; same seed across thread counts → predictions within tolerance. |
| Numerical parity | `test_parity_xgb`, `test_parity_lgbm` | bonsai's RMSE matches reference libraries within tolerance on matched hyperparameters on the MVP regression dataset. AUC parity on a classification dataset is added in Phase 4. |
| Golden-file | `test_golden_predictions` | Predictions on a small committed regression dataset do not drift across commits. |

Conventions:

- **Tolerances, not equality.** Parallel reductions on floats are non-associative. A `WithinAbs(expected, tol)` helper is the default comparator for any cross-thread / cross-backend comparison. Tolerance values are picked after observing actual variance, not guessed up front.
- **Property-based tests for `BinMapper`.** Random data → bins → assert monotonicity of bin assignment, count invariants, edge handling. Catch2 `GENERATE` is sufficient; rapidcheck is overkill for this project's needs.
- **Determinism test from day one.** Two flavors: (a) at fixed `n_threads`, same seed + same data → identical model bytes; (b) across thread counts, same seed + same data → predictions within numerical tolerance. The (a) test will fail the first time parallelism is added if any reduction uses atomic FP adds or otherwise loses reproducibility at fixed thread count — which is the point. Cross-thread bit-exactness is *not* promised (see decisions §7) — XGBoost and CatBoost don't guarantee it either, and LightGBM only does so behind a flag that its own maintainers describe as fragile.
- **Golden files for one small regression dataset.** Reference predictions committed to the repo as small files; the test fails if predictions drift. High signal for catching subtle regressions during the parallel phase.

Test data lives under `tests/data/` (small CSVs and golden files only — no large datasets in git).

---

## 5. Benchmarking approach

Benchmarking is treated as a first-class deliverable, not an afterthought. The harness exists in Phase 2, before parallelism, so that every later change can be measured against a stable baseline.

**Tooling.** google-benchmark for microbenchmarks (separate binary, not mixed with Catch2). The CLI's `bonsai bench` subcommand drives end-to-end timing on real datasets.

**Datasets.** YearPredictionMSD (regression, ~500K rows) is the MVP reference dataset — it drives parity, golden-file, and speedup measurements through Phases 1–3. A small subset of YearPredictionMSD (~10K rows) is committed for golden-file tests; the full dataset is downloaded on demand by a helper script. Additional regression datasets, and Higgs (binary classification) once logloss lands in Phase 4, are added as scope allows. Loaders normalize to a common CSV form.

**What gets measured.** All numbers reported as median of N runs with percentile ranges:

- **Predictive parity.** RMSE on the MVP regression dataset versus xgboost, LightGBM, and CatBoost at matched hyperparameters (rounds, learning rate, depth, min-child constraints). Reported as a table. AUC columns added alongside logloss in Phase 4.
- **Training speedup.** Wall-clock training time at 1, 2, 4, 8, 16 threads, for both OpenMP and `std::execution` backends. Reported as a curve.
- **Inference throughput.** Rows per second for batch predict, single-tree versus full-model.
- **Static vs dynamic dispatch microbenchmark.** A separate hot-path microbenchmark (split scoring inner loop) compiled with virtual dispatch and with the static-dispatch path. Direct measurement of the architectural claim in section 1.
- **Ablation.** Histogram subtraction on / off; bin width 255 vs 65535; GOSS sampling on / off. Each is one column in a results table.
- **Memory.** Peak resident set size during training; serialized model size on disk.

Results land in `docs/benchmarking.md` (created in Phase 2). Reproduction recipe — datasets, command lines, hardware notes — committed alongside.

---

## 6. External dependencies

All dependencies are header-only or vendored via CMake `FetchContent`. No system package dependencies. A clean checkout builds with `cmake -B build && cmake --build build` and everything resolves.

| Concern | Library | Notes |
|---|---|---|
| Test framework | Catch2 v3 | Link `Catch2WithMain`. Property-based via `GENERATE`. |
| Microbenchmarks | google-benchmark | Separate binary from Catch2. |
| CLI parsing | CLI11 | Header-only. Native dotted-key support for nested TOML overrides. |
| Configuration | toml++ | Strict mode (reject unknown keys). |
| Logging | spdlog | FetchContent. Saves a week of yak-shaving on a custom logger. |
| Progress bars | indicators (p-ranav) | Header-only, thread-safe, has `MultiProgress` for showing per-validation eval alongside training. |
| Parallelism (a) | OpenMP | Compiler-provided. Primary backend. |
| Parallelism (b) | `std::execution` + TBB | TBB vendored via FetchContent — libstdc++ requires it for `par` to actually parallelize. Runtime check confirms. |

The reflection branch additionally requires gcc-trunk with C++26 reflection support enabled. That branch is not on the critical path; the main branch builds with any C++23-capable compiler.

---

## 7. Implementation phases

### 7.1 Phase plan

Phases are ordered, not time-boxed. Each phase's exit criteria gate the next.

| Phase | Deliverable | Definition of done |
|---|---|---|
| **1. Serial MVP** | End-to-end serial train + predict on numeric features. Regression (MSE) is the parity / golden / benchmark target on YearPredictionMSD. The `Objective` concept is implemented for both MSE and logloss (logloss has unit tests against analytical gradients and a smoke-level integration test on synthetic 2-class data, but no live reference dataset — that comes in Phase 4). Depth-wise grower. Histogram split finder. Uniform sampler. CLI: `bonsai fit / predict / eval`. TOML config + CLI overrides. | YearPredictionMSD trains, predictions match ballpark accuracy of reference libs (no parity checks yet), unit tests green for both MSE and logloss objectives, save/load round-trips. |
| **2. Benchmark harness** | Dataset loaders, parity tests vs xgboost / LightGBM / CatBoost, golden-file tests, microbenchmark scaffold for the dispatch microbenchmark. | RMSE parity passes within initial tolerance bands on YearPredictionMSD. `bonsai bench` produces reproducible numbers. |
| **3. Parallelism** | OpenMP backend (feature-parallel histogram construction, row-parallel within feature, parallel predict, SIMD bin scan). Then `std::execution` backend behind the same `ParallelBackend` concept. Determinism tests (fixed-N bytes, cross-N tolerance) continue to pass. | Speedup curve is positive and monotone up to 8 threads on both backends. Fixed-thread-count determinism passes at every supported `n_threads`; cross-thread-count predictions agree within tolerance. Numerical parity tolerances unchanged. |
| **4. Extensions** | Binary classification parity lands first: a classification reference dataset (Higgs subset), AUC parity vs reference libs, golden file. The logloss objective is already in the codebase from Phase 1, so this phase is dataset wiring + parity tuning, not new core code. Then leaf-wise grower; oblivious grower; GOSS sampler; exact splitter; categorical handlers. Each demonstrates the extension API by being added without touching the core. | Each extension is one new file plus one entry in the registry typelist (or, for classification, one new dataset + parity test). Unit tests + a parity check against the corresponding reference library on at least one dataset. |
| **Stretch** | — | C++26 reflection branch for the registry. Monotonic constraints. Snapshot / checkpoint format (currently a binary dump for v1). CV runner. Python bindings via pybind11 (probably skip). | Each is independently scoped. Stretch items are pop-off-able if time runs short. |

### 7.2 Open questions / risks

- **Dispatch mechanism shape** at the runtime → static boundary. Resolved in `docs/architecture/6-dispatch.md` before code lands. The cartesian-product instantiation is locked in; the syntactic shape is not.
- **`std::execution::par` quality.** libstdc++ silently falls back to serial if TBB is not linked. Mitigation: vendor TBB via FetchContent; add a runtime check at startup that confirms `par` actually parallelizes.
- **Numerical parity tolerances.** Picking tolerance bands up front is a guess. Plan: measure baseline variance on the parity tests in Phase 2, set tolerances 2–3x the observed variance, document the methodology.
- **C++26 reflection branch viability.** Depends on gcc-trunk stability through the term. If the reflection branch becomes a time sink it is dropped; the typelist version is the production path either way.
- **Phase 4 scope.** Each extension demonstrates the API, but adding all of them is not the goal. If three extensions land cleanly the architectural claim is supported; pushing for five is diminishing returns versus polishing benchmarks and the final report.
- **Dataset acquisition / size.** YearPredictionMSD (~500K rows, ~90 features) is small enough for fast iteration through Phase 1–3 and big enough that speedup curves are meaningful. Higgs is only loaded once binary classification lands in Phase 4, and even then a subset is used for the inner loop.
- **One reference dataset at a time.** Phase 1 deliberately benchmarks against one dataset (YearPredictionMSD) so debugging effort is not split across two reference loops simultaneously. The mitigation against accidentally hard-coding regression assumptions into the architecture is to land logloss as a *second* `Objective` implementation in Phase 1 with unit-test and synthetic-data coverage — enough to flush out any regression-only shortcuts in concepts and dispatch — while deferring its live parity dataset to Phase 4.

### 7.3 Evaluation criteria for the final report

The figures and tables that will appear in the writeup, decided up front so that instrumentation is built into the code rather than bolted on:

- **Predictive parity table.** RMSE on the MVP regression dataset (and AUC once logloss lands), columns for bonsai / xgboost / LightGBM / CatBoost, matched hyperparameters.
- **Speedup curves.** Wall-clock training time vs thread count (1, 2, 4, 8, 16), one line per backend (OpenMP, `std::execution`).
- **Static vs dynamic dispatch microbenchmark.** Hot-path runtime with and without virtualization. The architectural claim from section 1 stands or falls on this number.
- **Ablation.** Histogram subtraction on/off, GOSS on/off, bin width 255 vs 65535.
- **Memory footprint.** Peak RSS during training and serialized model size on disk.
- **Extension API audit.** A short table showing, for each Phase 4 extension, the diff size required to add it (lines added, files touched, registry entries). Demonstrates the open/closed claim concretely.

---

## 8. Physical design

### 8.1 Repository layout

```
bonsai/
  CMakeLists.txt              top-level build
  Makefile                    one-time setup (e.g., `make skills`)
  README.md                   short orientation, build, link to proposal
  docs/
    proposal.md               this document
    context.md                handoff briefing for agents / future-self
    decisions.md              decisions log, append-only
    benchmarking.md           (Phase 2+) results, methodology, reproduction
    architecture/             per-component design docs
      README.md               index + cross-cutting concerns
      1-dataset.md            Dataset, BinMapper, readers
      ...                     2-histogram, 3-tree, ... (added as designed)
    conversations/            preserved design transcripts
  include/bonsai/             public-ish headers
    core/                       Dataset, BinMapper, Histogram, Gradient
    tree/                       Tree, Node, TreeGrower concept
    split/                      SplitFinder concept
    objective/                  Objective concept
    sampler/                    Sampler concept
    parallel/                   ParallelBackend concept
    booster/                    Booster<Obj, Gr, Sp, Sa, Backend>
    config/                    Config struct, parser, CLI merger
    registry/                  Registry<Base, Config, Impls...>
    io/                        readers, model serialization
  src/
    CMakeLists.txt
    core/  tree/  split/  objective/  sampler/
    parallel/  booster/  config/  io/
    cli/                        subcommand handlers (fit, predict, eval, bench, info)
                                main.cpp lives here
  tests/
    CMakeLists.txt
    unit/  integration/  numerical/  golden/
    data/                       small CSVs and golden prediction files
  bench/
    CMakeLists.txt
    micro/                      google-benchmark microbenchmarks
    end_to_end/                 driver scripts for `bonsai bench`
  third_party/                  any vendored deps not via FetchContent (ideally none)
  scripts/                      dataset download helpers, parity-run scripts
```

### 8.2 Module boundaries

The directory structure mirrors the component concepts. Each component directory contains the concept definition, the configuration struct, and the concrete implementations as separate translation units. CMake `OBJECT` libraries group implementations per component family (e.g., `bonsai_objectives` contains `mse.cpp`, `logloss.cpp`, ...) — `OBJECT` libs sidestep the static- archive dead-code stripping that would otherwise be a problem if any self-registration were ever added.

The CLI binary depends on the component object libraries, the config parser, and the booster header. There is exactly one `int main()` in `src/cli/main.cpp`; everything else is library code, so `tests/` and `bench/` link directly against the component libraries without recompiling.

### 8.3 Build invariants

- Single `cmake --build` from a clean checkout produces the CLI, tests, and benchmarks. No system packages assumed.
- All third-party dependencies vendored via `FetchContent` in the top-level `CMakeLists.txt`. Pinned to specific tags / commits.
- C++23 is the floor. The reflection branch additionally requires `-std=c++26` and gcc-trunk; behind a CMake option (`BONSAI_REFLECTION`) that is OFF by default.
- `-Wall -Wextra -Wpedantic -Werror` in the default build profile; sanitizer-enabled debug profile (`-fsanitize=address,undefined`) for test runs in CI.
