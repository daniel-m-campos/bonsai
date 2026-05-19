# Project Context — GBT Library

> **Purpose of this doc**: Single-source briefing for any agent (or future-you)
> picking up this project. Captures decisions made, rationale, what's deferred,
> and what's out of scope. Keep this updated as decisions evolve; it's the
> first thing any new session should read.

## 1. Project in two sentences

A from-scratch C++23 implementation of a histogram-based gradient boosted
trees library, built as a final project for MPCS 51045 (Advanced C++).
Goal is to demonstrate modern C++ techniques (concepts, parameter packs, fold
expressions, optional reflection) and parallelism (OpenMP + std::execution)
while producing a working, benchmarked GBT that can be compared against
xgboost / lightgbm / catboost on one regression dataset for the MVP.

## 2. Goals and non-goals

**Goals:**
- Working serial histogram-GBT with regression as the live MVP path
  (parity, golden, benchmarks all run against one regression dataset).
- Logloss objective + binary-classification prediction path implemented
  alongside MSE in MVP, but only as far as unit tests + synthetic-data
  smoke tests carry it. Live classification parity is Phase 4. The goal
  is to keep the `Objective` concept and dispatch honest under a second
  implementation, not to debug two reference datasets at once.
- Parity (within tolerance) with reference libraries on RMSE on the MVP
  regression dataset. AUC parity on a classification dataset is added in
  Phase 4.
- Two parallel backends (OpenMP, std::execution) with measurable speedup.
- Clean extension API spanning the conceptual surface of all three
  reference libraries (different growers, splitters, samplers).
- Modern C++ showcase: concepts, static dispatch in hot paths, optional
  C++26 reflection branch.
- CLI-first (no Python bindings), CatBoost-style argument conventions.

**Non-goals:**
- Beating xgboost/lgbm on training speed (they have years of tuning).
- Production readiness, ABI stability, or distribution.
- Python/R bindings.
- GPU support.
- DART, Random Forest, or ranking objectives in the main path.
- Distributed training.
- Multiclass classification in the main path.
- Full classification parity / golden / benchmark coverage in the MVP.
  Logloss code lives in MVP under unit tests, but a second live reference
  dataset is not debugged in parallel with regression.
- Categorical features in MVP (deferred to Phase 4 as extension demo).

## 3. Justification beyond pedagogy

The honest pitch (use these in proposal section 1):
- **Reference value**: Production GBT libraries are large and hard to read.
  A clean ~3-5K LOC version with thorough docs is a teaching artifact.
- **Architectural improvement**: Compile-time dispatch in the inner loop is
  a measurable refinement over the uniform-virtual approach in xgb/lgbm.
  Microbenchmark with/without virtualization is a real result.
- **Maintainability via concepts**: Component contract violations caught at
  compile time, not runtime — improvement over reference libraries.

Avoid claiming: speed parity with reference libraries, production suitability.

## 3.5. Spine implementation status

The bonsai-policy gate (`.claude/skills/bonsai-policy`) restricts AI from
writing initial versions of spine components. Status as of 2026-05-18
(CLI, CSV reader, TOML loader, model save/load, Python comparison
sidecar, and the typed-section config / metric registry / fit-time
metric tick logging features are landed):

| Spine component | User-authored | File(s) |
|---|---|---|
| `Dataset`, `BinMapper`, `BinMappers` | yes | `include/bonsai/{dataset,bin_mapper,bin_mappers}.hpp` |
| `Histogram` | yes | `include/bonsai/histogram.hpp` |
| Depth-wise `TreeGrower`, `Tree` | yes | `include/bonsai/{grower,tree}.hpp` |
| Histogram `SplitFinder` | yes | `include/bonsai/split.hpp` |
| `Objective` concept + MSE + LogLoss | yes | `include/bonsai/objective.hpp` |
| `Sampler` concept | yes | `include/bonsai/sampler.hpp` |
| `Booster<O,G,Sp,Sa>` + `IBooster` | yes | `include/bonsai/booster.hpp` |
| Registry / dispatch (flat table) | yes | `include/bonsai/registry/`, `include/bonsai/typelist.hpp` |
| Dispatch resolution doc | yes | `docs/architecture/6-dispatch.md` |
| `ParallelBackend` concept | **no — gated** | (doc 7 TBD) |
| `SerialBackend`, `OpenMPBackend` | **no — gated** | (depends on doc 7) |

**Spine complete as of 2026-05-18.** The two remaining gated rows
(`ParallelBackend` concept + first impl) are reclassified as part of
Phase 3 work rather than spine-gated work — the spine-authorship
gate is satisfied for everything required to ship the Phase 1 MVP
and Phase 2 benchmark harness. AI assistance is now open across the
codebase (see §11 for the relaxed rules).

## 4. Decisions made (with rationale)

### Architecture
- **Static polymorphism in hot paths, dynamic at config boundary.**
  Components (Objective, TreeGrower, SplitFinder, Sampler) are concepts.
  `Booster<Obj, Gr, Sp, Sa>` is a class template. Runtime dispatch from TOML
  string happens once at construction; everything inside the booster is
  statically dispatched. Cost: cartesian product instantiations (~15-90
  variants), acceptable at this scale. Resolved as a flat table over
  `cartesian_product_t<...>` (decision 26); see
  `architecture/6-dispatch.md`.

- **`Booster` as single concrete class for now**, not abstract base.
  Virtualize when adding DART/RF (probably never, given non-goals).

- **Typelist registry for runtime → type dispatch** at the config boundary.
  Each impl declares `static constexpr std::string_view name`. Registry is
  `Registry<Base, Config, Impls...>` with fold-expression dispatch. No
  macros, no static-init, no linker-stripping issues. Concepts enforce
  contract at compile time.

- **No registry aliases.** Single canonical name per component. xgb/lgbm
  have many aliases per concept; not worth the maintenance.

- **Optional reflection branch** for the registry, using gcc-trunk
  (P2996 / static reflection in C++26). User has this working in another
  project. Showcase only; typelist version is the production path.

- **Skip the course's abstract-factory profile pattern.** Earlier sketch
  proposed using mpcs::abstract_factory for "lgbm-style profile" bundles.
  Rejected as abstraction-for-its-own-sake; TOML defaults handle presets
  fine.

### Data layout (lock in early, expensive to change)
- **Column-major storage.** Histograms read one feature across all rows;
  row-major destroys cache. Non-negotiable.
- **Per-feature uint8 vs uint16 bins** depending on `max_bin`. Saves 50%
  memory for the common case (max_bin <= 255). `std::variant` per feature
  column.
- **`std::span` returns** from Dataset, not vector copies.
- **Float storage, double accumulators.** Float for labels, scores,
  gradients in storage. Double for histogram accumulation, split scoring.
  Matches xgb/lgbm. Half the bandwidth, no measurable accuracy loss.

### Iteration model
- **`Booster::update_one_iter()`** as the unit of progress. Top-level
  `train()` is a loop over this. Callbacks (logging, snapshotting,
  early stopping) live outside the core. Mirrors xgb/lgbm.

### Parallelism
- **Two backends, both implemented**: OpenMP (primary, standard for this
  domain) and std::execution (showcase). Selected by CLI flag. Same
  benchmarks run against both.
- **Known risk**: libstdc++ `std::execution::par` requires linking TBB or
  it falls back silently to serial. Vendor TBB via FetchContent. Add a
  runtime check.
- **Determinism is a hard requirement at fixed thread count.** Same seed
  + same `n_threads` → same model bytes. Cross-thread-count: predictions
  within tolerance, bytes may differ. Forces per-thread local hists from
  day one (no atomic FP adds — bit-unstable even at fixed N). See
  decisions §7.

### Testing
- **Catch2 v3** (link `Catch2WithMain`).
- **Layered**: unit / integration / numerical-parity / benchmarks.
- **Tolerance-based** comparisons via `WithinAbs` (parallel reductions are
  non-associative for floats).
- **Property-based tests** for bin mapper using Catch2 `GENERATE`.
- **Determinism test** from the start (will fail first time you parallelize
  if you didn't design for it — that's the point).
- **Golden-file tests** for 1-2 small datasets (commit reference predictions,
  fail on drift).

### CLI
- **CatBoost-style subcommands**: `bonsai fit`, `bonsai predict`,
  `bonsai eval`, `bonsai bench`, `bonsai info`, `bonsai params`.
- **CLI11** for parsing — header-only, dotted/nested keys, no deps.
- **Fit-time output** is text via `std::print` from a tick callback
  gated on `booster.log_intervals`. Streaming progress-bar UI is
  deferred (not currently wired).
- **TOML config + CLI overrides** via dotted `--set` keys
  (`--set tree.max_depth=8`, repeatable). Resolution: defaults → file
  → CLI. Last write wins.
- **toml++** for parsing (strict mode — reject unknown keys).

### Config schema
- **Sections = component instances.** `[objective]`, `[tree]`, `[sampler]`,
  `[split]` each map to one factory call. `name` field selects impl, rest
  is impl-specific config.
- **Strongly-typed nested structs** in C++. Defaults at struct level, not
  in parser.
- **Component-specific params live with the component** (`top_rate` is in
  `[sampler]`, not global).
- **Validation in component constructors**, not centralized. Throw
  `ConfigError` with key path on bad input.
- **No `extra` map for "future-proofing."** Strict typing throughout.

### Logging
- **spdlog** is fetched but not yet wired into source. CLI fit-time
  output is `std::print` for now (decision deferred until a real
  logging surface needs structured records).

### Build
- **CMake with FetchContent** for all deps (Catch2, CLI11, toml++,
  spdlog, nlohmann/json). google-benchmark and TBB will be added when
  the microbench and parallel backends land.
- **OBJECT libraries** for component groups — avoids static-archive
  dead-code stripping issues.
- **No system package dependencies** — single `cmake --build` for grading.

## 5. Decisions deferred / open questions

- **Numerical parity tolerance values** — pick after seeing actual variance.
- **Whether to implement the C++26 reflection branch** or just sketch it.
  Depends on time remaining after main path is solid.
- **Snapshot/checkpoint format** — defer until needed; simple binary dump
  for v1.
- **CV runner** — nice-to-have, defer.

## 6. Architecture sketch

Current layout: `include/bonsai/` is mostly flat with header-per-component;
subdirectories exist for `cli/`, `config/`, `io/`, `registry/`, and
`detail/`. Source mirrors `src/`. Aspirational shape, mapping component
to file or directory (future impls noted where they don't exist yet):

```
include/bonsai/
  dataset.hpp, bin_mapper.hpp, bin_mappers.hpp, histogram.hpp
                — Dataset, BinMapper, Histogram, Gradient
  tree.hpp, grower.hpp
                — Tree, Node, TreeGrower concept;
                  impls: depthwise (now); leafwise, oblivious (future)
  split.hpp     — SplitFinder concept; impls: histogram (now), exact (future)
  objective.hpp, objective_traits.hpp, task.hpp, metric.hpp
                — Objective concept; impls: mse, logloss (now);
                  softmax, quantile, huber (future)
  sampler.hpp   — Sampler concept; impls: all_rows (now);
                  uniform, goss, bernoulli (future)
  cat/          — (Phase 4) CategoricalHandler;
                  impls: onehot, partition, target_stat
  parallel/     — (Phase 3) ParallelBackend concept;
                  impls: serial, openmp, stdexec
  booster.hpp   — Booster<Obj, Gr, Sa>, IBooster, training loop
  io/           — CSV reader; model save/load (MessagePack);
                  libsvm, parquet readers (future)
  config/       — Config struct, typed sections, FieldCodec,
                  TOML parser + dumper, CLI override merger
  registry/     — flat table over cartesian_product_t<...>,
                  IBooster at boundary
  cli/          — subcommand handlers
                  (fit, predict, eval, bench, info, params)
```

## 7. Implementation phases (high level)

1. **Phase 1: Serial MVP.** Numeric features only. MSE is the live parity
   target on one regression dataset (California Housing). Logloss
   objective ships alongside MSE — analytical-gradient unit tests + a
   synthetic 2-class smoke test, no live reference dataset. Depth-wise
   grower, histogram splitter, all-rows sampler. CLI: fit / predict /
   eval / bench / info / params. TOML config + CLI overrides.
2. **Phase 2: Benchmark harness.** California Housing parity script
   vs xgboost / lightgbm / catboost (RMSE), golden files,
   microbenchmark scaffold.

**Phase 2.5: CLI design + cleanup** (inserted 2026-05-18, decision
28). Before turning on parallelism, take a pass on CLI usability, the
typed-config surface, and the small things glossed over during the
Phase 1/2 sprints. Items captured as concrete tasks during the work,
not pre-listed here. No new spine, no parallel backends; refactors
are fair game now that AI assistance is open.

3. **Phase 3: Parallelism.** OpenMP first (feature-parallel hist
   construction, row-parallel within feature, parallel predict, SIMD bin
   scan). Then std::execution variant. Determinism + tolerance regression
   throughout. Adds YearPredictionMSD as the perf benchmark (California
   Housing is too small to expose scaling); speedup curves are measured
   here, not on the integration dataset.
4. **Phase 4: Extensions.** First item is binary classification parity:
   wire in a classification reference dataset (Higgs subset), AUC parity
   tests, golden file. Logloss code is already in the codebase from Phase
   1, so this is dataset wiring + tuning, not new core code. Then
   leaf-wise grower, oblivious grower, GOSS sampler, exact splitter,
   categorical handlers — one per "library style" demonstrates the
   extension API claim.
5. **Stretch:** C++26 reflection branch for registry, Python via pybind11
   (probably skip), monotonic constraints.

Phases are ordered, not time-boxed. No week assignments.

## 8. Doc structure

```
README.md             — short orientation, build, link to proposal
docs/
  proposal.md         — the proposal: rationale, plan, scope, phases,
                         evaluation criteria
  context.md          — this file (handoff briefing)
  ai-usage.md         — AI policy + audit trail
  decisions.md        — append-only decisions log
  benchmarking.md     — (later) results, methodology, reproduction
  architecture/
    README.md         — index + cross-cutting concerns (dispatch,
                         threading, error handling, determinism)
    1-dataset.md      — Dataset, BinMapper, readers
    2-histogram.md    — Histogram, subtraction, parallel reduce
    3-tree.md         — Tree, Node, TreeGrower
    4-objective.md    — Objective, MSE, logloss
    5-booster.md      — Booster, training loop
    6-dispatch.md     — registry, runtime → static boundary
    7-parallel.md     — (TBD) ParallelBackend, OpenMP, std::execution
    8-config.md       — Config, TOML, CLI overrides
    9-cli.md          — subcommand handlers
  conversations/
    2026-05-02-initial-design.md  — full transcript of design chat
```

`README.md` stays short — points readers at `docs/proposal.md`.

`docs/proposal.md` sections (locked):
1. Application idea + justification beyond pedagogy
2. Survey of xgboost / lgbm / catboost (table)
3. Core architecture (entities, extension points, perf-sensitive,
   data invariants, metaprogramming approach, static vs dynamic
   polymorphism, dispatch boundary)
4. Testing approach
5. Benchmarking approach
6. External dependencies
7. Implementation phases (table)
8. Physical design / directory structure
+ Goals/non-goals near top
+ Open questions / risks before phase plan
+ Evaluation criteria for final report (figures/tables planned)

`architecture/` holds per-component design docs. Each numbered file
covers one component (Dataset, Histogram, Tree, etc.) in enough depth
to implement it. `architecture/README.md` is the index plus
cross-cutting concerns (dispatch architecture, threading model, error
handling, determinism contract).

`decisions.md` is the append-only decisions log. Plain ordered list,
caveman style, one entry per non-trivial choice. New entries at the
bottom; numbering never reused.

## 9. Tech stack (locked)

| Concern | Choice | Notes |
|---|---|---|
| Language | C++23 | gcc-trunk for optional reflection branch (C++26) |
| Build | CMake + FetchContent | No system deps |
| CLI | CLI11 | Dotted nested keys |
| Config | toml++ | Strict mode, reject unknown keys |
| Progress | `std::print` tick callback | indicators (p-ranav) deferred |
| Logging | spdlog (fetched, not yet wired) | structured logging deferred |
| Tests | Catch2 v3 | Link `Catch2WithMain` |
| Bench | google-benchmark (planned) | not yet fetched |
| Parallel (planned) | OpenMP + std::execution | TBB to be vendored for stdexec |

## 10. Evaluation criteria (figures planned for final report)

- **Predictive parity table**: RMSE on the MVP regression dataset (and
  AUC once the Phase 4 classification dataset lands) vs xgb/lgbm/catboost
  at matched hyperparameters
- **Speedup curves**: 1, 2, 4, 8, 16 threads; OpenMP vs std::execution
- **Static vs dynamic dispatch microbenchmark**: hot path with/without
  virtualization
- **Ablation studies**: histogram subtraction on/off, GOSS on/off,
  bin width (255 vs 65535)
- **Memory footprint**: peak RSS during training, model size on disk

These figures should drive instrumentation decisions in the code.

## 11. AI usage policy (binding for all agents)

The professor has set a course policy on AI use. Full text and rationale
in `docs/ai-usage.md`; the operational summary for agents is below.
**This section is load-bearing — read it before writing code.**

The professor's two asks, paraphrased:
1. **Disclosure.** Document exactly how AI was used.
2. **Author the core yourself.** "Write the core functionality and
   architecture yourself, but once you have done so, judicious use of
   AI to add features, tests, produce architecture diagrams, etc. is
   ok." Initial implementation must be by the user.

**Status (2026-05-18).** Spine is complete. The user has hand-authored
every spine component required to ship the Phase 1 MVP and Phase 2
benchmark harness end-to-end (Dataset, BinMapper, BinMappers,
Histogram, depth-wise + oblivious TreeGrower, DenseTree, ObliviousTree,
histogram SplitFinder, Objective concept + MSE + LogLoss, Sampler
concept + AllRowsSampler, Booster + IBooster, registry / dispatch flat
table). The `ParallelBackend` concept and its first impl are reclassified
as Phase 3 work, not spine-gated. The course rule "write the core
functionality and architecture yourself" is satisfied.

What this means for any agent (Claude Code, subagent, etc.) working in
this repo from 2026-05-18 onward:

**ALLOWED — broadly**
- Refactoring anywhere in the codebase, including spine components,
  CLI, config, I/O, registry, and tests.
- Writing new features: additional objectives, samplers, growers,
  split finders, the parallel backends, categorical handlers.
- Authoring tests, build, benchmarks, dataset loaders, scripts.
- Drafting and editing docs, decisions, diagrams.
- Following commit-message style and producing PR descriptions.

**ALWAYS**
- Disclose AI assistance via commit trailer when AI wrote any portion
  of the diff: `AI-Assisted: <short description>`. Absence of trailer
  means 100% user-written.
- Preserve conversation transcripts of non-trivial design discussions
  under `docs/conversations/`.
- Stop and ask before destructive operations or anything that crosses
  obvious scope. Approval in one task does not transfer to the next.

**NOT ALLOWED**
- Silently writing without the trailer. Disclosure is non-negotiable.
- Making design decisions the user accepts without independent
  rationale. AI proposes, user decides.

Professor signed off on the policy on 2026-05-04 ("LGTM"). The
spine-authorship gate that operated during Phase 1 is now satisfied
and lifted (decision 28). Full audit trail in `docs/ai-usage.md`.

## 12. What this doc is not

- Not the proposal. Proposal is for the professor; this is for agents.
- Not a replacement for `architecture/`. This is the briefing; those
  are the per-component deep dives.
- Not append-only. Update in place as decisions evolve. Use
  `decisions.md` for the audit trail of *changes*.

## 13. Pointers

- Full design conversation transcript:
  `docs/conversations/2026-05-02-initial-design.md`
- AI usage policy (full text + audit trail):
  `docs/ai-usage.md`
- Course context: MPCS 51045 (Advanced C++), Spring 2026
  https://mpcs-courses.cs.uchicago.edu/2025-26/spring/courses/mpcs-51045-1
- Reference libraries:
  - xgboost: `include/xgboost/` — DMatrix, Learner, GradientBooster
  - LightGBM: `include/LightGBM/` — Dataset, Boosting, TreeLearner
  - CatBoost: C API only, CLI-first for training
