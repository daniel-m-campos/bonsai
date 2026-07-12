# Design Review: bonsai (whole project)

## Project Overview
- **Repository**: github.com/daniel-m-campos/bonsai (reviewed at v1.0.0, `a8a0a07`)
- **Language**: C++23 (clang CUDA C++ for the GPU TU; nanobind Python layer; TOML config)
- **Description**: A from-scratch histogram GBDT library — CLI, Python bindings, CPU + CUDA backends — built to be reference-grade readable while matching xgboost/lightgbm/catboost on speed and quality.

## Scoping
- **Extensibility pressure**: **medium-high** — the component seams (objective, grower, sampler, engine, finder) are the product's stated identity, and the guide teaches extension; but there is one maintainer and the variant space is largely enumerated already.
- **Performance pressure**: **high** — hot paths are measured in GB/s and priced per second (docs 16, guide ch.11); the project trades blows with production libraries.
- **Data clarity**: **exceptional** — raw float matrix → quantile cuts → binned u8/u16 columns → per-node (grad, hess) histograms → splits → trees → score updates. The pipeline is documented as a compute DAG with measured node costs; few codebases can state their data flow this precisely.

Calibration cell: **high extensibility × high performance → SOLID at the boundaries, DOD inside** — which is, verifiably, the design the codebase actually implements. The review therefore grades against that standard, not against maximal abstraction.

## Per-Entity Analysis

### Booster / MulticlassBooster / IBooster (`include/bonsai/booster.hpp`, `multiclass_booster.hpp`)
| Principle | Rating | Analysis |
|-----------|--------|----------|
| S | ⚠️ | `Booster::update_one_iter` is 133 lines and changes when the objective seam changes, independently when DART normalization changes, and independently when warm-start replay changes. Extracting `apply_dart_round(...)` and `replay_warm_scores(...)` would leave a readable core loop. Same shape in `MulticlassBooster::update_one_iter`. |
| I | ⚠️ | `IBooster` is one wide boundary (~14 virtuals): the CLI's `dump` handler uses `{dump}` but is coupled to the ES quartet; the Python module never calls `seed_valid_scores`. Deliberate — one type-erased seam instead of many — but a `// grouped: prediction / introspection / training-loop` comment structure (or splitting an `ITrainLoop` view) would make the width legible. |
| D | ✅ | Objective/grower/sampler arrive as template parameters from the composition root (`make_booster`); nothing inside instantiates a concrete collaborator. Unit tests construct boosters with any combination directly. |
| O | ✅ | New objectives/growers/samplers require zero booster edits (`if constexpr (requires ...)` absorbs optional capabilities like `renew_leaf`); the softmax shape exception routes at dispatch (`BoosterFor`), not via flags inside. |
| L | ✅ | Two `IBooster` implementations; contracts hold with one documented semantic shift: `predict` emits class ids for multiclass vs raw scores for regression — task-dependent by design, surfaced in docs. The former throwing edges (multiclass ES/SHAP) were removed in the consistency pass. |
| D1 | ✅ | Transforms (scores, labels) → (grad, hess) → tree → score delta. The round loop reads as the guide describes it. |
| D2 | ✅ | Flat score/grad/hess arrays; multiclass scores row-major `n×K` matching the per-row softmax access. |
| D3 | ✅ | Virtual dispatch only at the per-fit boundary (~1 call/round each); the training interior is monomorphized. |
| D4 | ✅ | The separate `MulticlassBooster` looks like duplication but is shape-forced (1-D `Objective` concept can't express K outputs) — the alternative (widening every objective) was rightly rejected. |

### Growers + grower_detail (`src/grower_impl.hpp`)
| Principle | Rating | Analysis |
|-----------|--------|----------|
| S | ✅ | Control plane only: `plan_level`/`demote_empty_splits`/`commit_children` are free functions with single jobs; the data plane lives in `LevelStep`. |
| I | ✅ | Growers see engines only through the `HistogramEngine`/`GPULevelEngine` concepts; the leafwise grower doesn't see level batching it can't use. |
| D | ✅ | Engine and finder are policy parameters; `grower_detail` functions take everything as arguments (which is what makes `test_demote.cpp`'s direct-call testing possible). |
| O | ⚠️ | A new *growth strategy* is a new ~140-line `grow()` — the three existing ones (66/143/141 lines) share setup/finalize/OOB scaffolding by copy. Acceptable at n=3 with unification previously attempted and rejected at a gate, but the scaffolding (not the strategies) could be shared; see DRY. |
| L | N/A | No hierarchy; growers are unrelated types unified by the `TreeGrower` concept. |
| D1 | ✅ | Each grow() is frontier state → level transactions → tree; the vocabulary (open/apply/end) is the doc-14 narrative. |
| D2 | ✅ | Output buffers recycled across trees (PR #38: 12.8GB of memset deleted); row indices contiguous; per-level vectors reused. |
| D3 | ✅ | Fully static; the one measured overhead ever found here (buffer zero-init) was instrumented, named, and deleted. |
| D4 | ⚠️ | The oblivious grow() inlines its own OOB complement walk instead of sharing `route_unsampled`'s (`:339` vs `:569`) — a simpler shared iterator exists; see DRY. |

### LevelStep (`src/level_step.hpp`)
| Principle | Rating | Analysis |
|-----------|--------|----------|
| S | ✅ | The tree's data plane, one transaction vocabulary, two specializations (host primary, `GPULevelEngine` partial specialization). Changes when the narrative changes — one reason. |
| I | ✅ | The GPU specialization's fallback arms reuse the host statics via `using HostStep = LevelStep<CpuHistogramEngine, ...>` — unusual but explicit, and narrower than inheriting. |
| D | ✅ | Engine injected by reference; dataset/config/gradients by view. |
| O | ✅ | Decision 53's point: a third backend implements the same verbs; nothing in growers changes. |
| L | ✅ | The two specializations honor the same transaction contract; the one historical divergence (device-plane demote, PR #29) is now guarded *and* unit-pinned. |
| D1/D2/D3 | ✅ | This file is where "SOLID at the boundary, DOD inside" is most visible: span-based payloads, slot-indexed device buffers, decisions crossing the bus in bytes. |
| D4 | ✅ | `make_root` staying a distinct step (rather than folding into `open_level`) was evaluated and kept for control-plane observability — documented, not accidental. |

### Engines (`src/grower.cpp` CPU, `src/cuda/histogram_engine.cu` + `kernels.cuh`)
| Principle | Rating | Analysis |
|-----------|--------|----------|
| S | ✅ | Post-decision-53: `Impl` = `DeviceData` (per dataset) / `GradientPlane` (per tree) / `LevelPipeline` (per level) — lifetime is the organizing principle, stated in types. The CPU engine's two fills (row-wise u8, feature-parallel u16) are separate functions chosen by data width. |
| I | ✅ | The CPU engine satisfies only `HistogramEngine`; the CUDA engine adds `GPULevelEngine`. Host callers never see device methods. |
| D | ✅ | The engine boundary is the repo's masterstroke: `Dataset` is host-pure; the CUDA TU is the only place naming device types; stub builds link a throwing twin. |
| O | ⚠️ | `HistogramEngine` as a concept is anemic (two loosely-typed methods) — a structurally-wrong engine could satisfy it and break semantics. Mitigated by static_asserts and the parity suite, but the concept under-documents the contract it enforces elsewhere in prose. |
| L | N/A | No hierarchy. |
| D1 | ✅ | The 16M ledger names every line of this file's cost; conservation is a working invariant, not an aspiration. |
| D2 | ✅ | Feature-major device bins (u8 where possible), `float2` gradient pairs, ping-pong level buffers, warp-per-(node,feature) find — every layout chosen against a measured access pattern. |
| D3 | ✅ | f32-chunk/f64-merge precision scheme; per-level sync floor understood and priced; two refuted optimizations (52, 35) documented alongside the wins. |
| D4 | ✅ | `DeviceBuffer` deliberately immovable (caught a real bug at CI); `Staged<T>` has exactly the surface its ~15 users need. |

### Dataset / BinMappers / IngestPlane (`include/bonsai/dataset.hpp`, `src/dataset.cpp`, `src/bin_mapper.cpp`)
| Principle | Rating | Analysis |
|-----------|--------|----------|
| S | ⚠️ | `Dataset` now owns four concerns: binned storage, the lazy row-major mirror, the lazy plane-materialization (`HostBins` + `call_once`), and labels/weights. Cohesive around "the binned training data," but the two lazy mechanisms have different thread-safety stories living side by side — worth one unifying comment or a shared lazy<T> helper if a third appears. |
| I | ✅ | `visit_bins` monomorphizes per width; `bin_at` serves routing; consumers see only what they use. |
| D | ✅ | `IngestPlane` is the textbook TU-firewall: host-pure virtual receipt, backend-tag recognition instead of RTTI, one sanctioned exception documented in-header with the IBooster precedent. |
| O | ✅ | A second device backend mints its own plane + tag; `Dataset` doesn't change. |
| L | ✅ | Plane implementations (CUDA, test fake) honor `materialize`'s contract; the fake exists precisely to pin the laziness semantics host-pure. |
| D1 | ✅ | bin() is the pipeline's clearest transform: raw → cuts → bytes. |
| D2 | ✅ | Column-major for histogram fill, lazy row-major mirror for the row-wise fill, device plane for CUDA — three layouts, each justified by its consumer, none paid for by workflows that don't use it. |
| D3 | ✅ | Two virtual calls per fit at the plane boundary; measured irrelevant. |
| D4 | ✅ | The plane's `call_once` lazy exists because a real race corrupted a real heap — need, not speculation. |

### Split finders (`src/split.cpp`, `include/bonsai/split.hpp`)
| Principle | Rating | Analysis |
|-----------|--------|----------|
| S | ✅ | Scoring math (`score`, `l1_thresholded`, `bounded_leaf_weight`) is shared constexpr functions — finders, growers, and the CUDA kernel cannot disagree about the penalty. |
| I / D / O | ✅ | Static `find()` behind two small concepts; per-feature parallel with deterministic feature-order reduction. |
| L | N/A | — |
| D1/D2/D3 | ✅ | One prefix scan per feature; thread-local prefix buffers in the level finder; hot loop is branch-light. |
| D4 | ⚠️ | `update_best_for_feature_for_node` (65 lines) and `..._for_level` (74 lines) share ~70% of their body: same `split_sums_at`, same feasibility/monotone/gain/best-update sequence, different iteration shape. A shared per-candidate scoring core (already half-extracted as `split_sums_at`) would leave two thin drivers. Byte-identity gates make this refactor safe to attempt and verify. |

### Objectives / Samplers / Metrics (`objective.*`, `sampler.*`, `metric.*`)
| Principle | Rating | Analysis |
|-----------|--------|----------|
| S | ✅ | Each objective is grad/hess/eval/init (+ optional renew); samplers are pure row-selection with in-place GOSS reweighting whose lifetime contract is documented where it matters. |
| I | ✅ | `renew_leaf` is opt-in via `requires` — MSE never sees or pays for it. |
| D | ✅ | Config injected at construction; params live with the objective instance (which the ES `valid_loss` change exploited correctly). |
| O | ✅ | New objective = concept + registry (see Registry entity for the honest edit count). |
| L | ⚠️ | `SoftmaxObjective`'s three throwing methods satisfy the `Objective` concept syntactically while violating it semantically — it is a dispatch tag wearing an objective's interface. Works, is commented, but a `MulticlassTag` type routed explicitly in `BoosterFor` would say what it is. |
| D1–D3 | ✅ | Row-parallel loops through `parallel::for_each_index`; nothing exotic. |
| D4 | ⚠️ | Metrics reimplement three objective evals — see DRY; the logloss pair differs *numerically* (log1p-stable vs epsilon-clamped), which is a correctness hazard, not just duplication. |

### Registry & dispatch (`include/bonsai/registry/`, `src/registry/`)
| Principle | Rating | Analysis |
|-----------|--------|----------|
| S | ✅ | Typelists declare, `names.hpp` labels, traits describe, `make_booster` composes — each file one job. |
| I | ✅ | Traits (`link_inverse_of`, `default_metrics_of`, `task_of`) are per-capability, checked by static_assert sweeps. |
| D | ✅ | This *is* the composition root; string → monomorphized type happens exactly once. |
| O | ⚠️ | Open in principle, but extension touches **6–8 files** (typelist, names, 2–3 traits, impl, config field, TOML section) — the README's "two edits" claim is stale and should be corrected; a documented checklist (or an X-macro consolidating name+traits) would close the gap between claim and reality. |
| L | N/A | — |
| D1 | ✅ | Config strings → a 90-entry constexpr table → one constructor call. Linear search over 90 at startup: correctly not optimized. |
| D2/D3 | N/A | Cold path. |
| D4 | ⚠️ | 90 monomorphized boosters (6×5×3) is the registered-everywhere policy's compile-time/binary bill; many combos (e.g., quantile × cuda_oblivious × goss) will plausibly never run. A conscious trade for uniform testing — worth stating a budget (when combos hit ~200, prune or gate instantiation). The five near-identical `for_each_type` table builders across two TUs are mechanical repetition a small helper would fold. |

### IO / CLI / Python (`src/io/`, `src/cli/`, `src/python/module.cpp`, `python/bonsai/`)
| Principle | Rating | Analysis |
|-----------|--------|----------|
| S | ✅ | main.cpp is pure CLI11 wiring; handlers are thin; `pipeline.cpp` holds the testable logic (and the tests use it CLI-free). `train_with_progress` at 120 lines carries ticks + ES + warm-start — the ES block is a candidate extraction if it grows again. |
| I | ✅ | Handlers depend on pipeline functions, not on each other. |
| D | ⚠️ | Model IO discovers the concrete booster by `dynamic_cast` sweep over all 90 combos (`try_save_as`/`try_load_into`) — honest at a serialization boundary, but a registry mismatch degrades to a runtime "no combo matched" rather than a compile error; the version-7 hard-fail plus `.value("covers", {})` forward-compat is the right pattern and should be the documented rule for future fields. |
| O | ✅ | New subcommand = one `register_*`; new tree field = versioned json key. |
| L | N/A | — |
| D1 | ✅ | CSV parse is a clean read→index→parallel-parse pipeline with the atomic first-error pattern; 163 lines but linear to read. |
| D2/D3 | ✅ | Parse parallelized; predict arrays returned to numpy via capsule ownership — consistent GIL-release pattern across all bindings. |
| D4 | ⚠️ | Config precedence (TOML + overrides + thread-set) exists twice (`resolve_config` in CLI, `config_from_params` in module) — same algorithm, two owners. Extract to `config::resolve(...)`. |

## Supplementary Principles (cross-cutting)
| Principle | Rating | Analysis |
|-----------|--------|----------|
| DRY | ⚠️ | Five real items, one with teeth: **(1)** metric vs objective eval — `compute_logloss` (epsilon-clamped) and `LogLossObjective::eval` (log1p-stable) are two numerically *different* answers to "what is this model's logloss," so early stopping and `bonsai eval` can disagree on the same data; MSE/MAE pairs are benignly identical. Single source: metrics delegate to the objective evals (or both to one kernel). **(2)** The node/level split-scan cores (~70% overlap, split.cpp:46–185). **(3)** The OOB complement walk (grower_impl.hpp:339 vs :569). **(4)** Config precedence CLI/module. **(5)** Reference-library parameter mappings duplicated across three bench scripts with a "keep in sync" comment — this exact drift produced a wrong-by-0.001-r² conclusion during the quality-gap investigation (decision 55's follow-up), so it has already cost an experiment; hoist the mapping into one imported module. Not violations: the three grow() strategies (similar shape, genuinely different control flow; unification was attempted and rejected on the record). |
| CoI | ✅ | Near-total: composition and policy templates everywhere; the only inheritance is two 1-level interface boundaries (`IBooster`, `IngestPlane`), both existing because a TU/ABI firewall makes static dispatch impossible, both documented as such. No hierarchy anywhere models "the real world." |
| LoD | ✅ | Injected components are talked to directly; the engine's `im.data.bins8`-style access is its own nested state, not reaching through strangers; the fluent-looking chains are span/view plumbing over PODs. The tests' `src/` include-path backdoor (test_demote reaching `grower_detail`) is the one sanctioned reach-through, justified in the CMake comment. |

## Summary

| Principle | Grade | Highlights |
|-----------|-------|------------|
| S | A− | Free-function control planes and lifetime-named engine structs; two long round/grow bodies are the residue. |
| I | B+ | Concepts keep component views narrow; `IBooster`'s width is the price of a single erased boundary. |
| D | A | Composition root + concept injection + the TU firewall; testability everywhere is the proof. |
| O | B+ | Genuinely open seams; the extension *cost* (6–8 edits) contradicts the marketing ("two edits") — fix the claim or the cost. |
| L | A− | Contracts hold; `SoftmaxObjective`'s throwing tag is the one semantic wink. |
| D1–D4 (DOD) | A | The distinguishing strength: data flow documented as a priced DAG, layouts chosen per measured access pattern, abstractions audited by refutation, YAGNI enforced by "priced before betting." |
| DRY | B− | Five concrete items; the logloss numeric divergence and the bench-mapping drift have already produced (or can produce) wrong numbers. |
| CoI | A | Two thin, justified interfaces; otherwise pure composition. |
| LoD | A− | Clean; one documented test backdoor. |

## Top 3 Recommendations
1. **Unify the objective-eval/metric implementations** — start with logloss, where the two copies are numerically different: make `compute_logloss` delegate to `LogLossObjective::eval` (or extract one stable kernel both call), then fold the identical MSE/MAE pairs. This is the only finding that can silently produce *disagreeing numbers* inside one binary (ES decisions vs `bonsai eval`).
2. **Merge the split-scan cores** (`update_best_for_feature_for_node`/`_for_level`): one candidate-scoring inline function (feasibility → monotone → gain → best-update) with two thin iteration drivers. The byte-identity gate (`scripts/model_hash.py`) makes this refactor cheaply verifiable, and it removes the risk of the two scans drifting the way the logloss pair already has.
3. **Close the extension-cost gap**: correct the README's "two edits" claim, add a "adding an objective" checklist to the guide (the 8 real edits, in order), and hoist the bench scripts' reference-library parameter mappings into one shared module — the decision-55 follow-up showed a one-knob mapping drift is worth a full false conclusion.

## Key Takeaways

bonsai is a rare specimen: a codebase whose *stated* design philosophy ("compile-time dispatch where it counts, SOLID at boundaries, data-oriented inside") is verifiably the philosophy it practices. The scoping matrix calls for exactly this hybrid, and the evidence is structural, not aspirational — two virtual boundaries in the entire system, both at firewalls where static dispatch is impossible by construction, both documented with their justification in-header; everything else is concepts, policies, and free functions over flat data. The DOD dimension is the strongest reviewed here in any project this size: the data flow is characterized to the point of having a priced graph (doc 16), layouts are chosen per measured access pattern with the rejected alternatives on record, and the decision log's *refutations* (52, 55, PR #35) are as carefully written as its wins. That habit — abstractions audited by measurement, simplicity enforced by pricing — is what D3/D4 reviews hope to find and almost never do.

The weaknesses are the ordinary sediment of fast iteration, and they cluster in one place: duplication at the edges. The metric/objective eval split is the only finding with correctness teeth (two logloss numerics in one binary); the split-scan twins and the bench-script parameter mappings are drift accidents waiting to repeat (one already did, costing an experiment's conclusion). None of these threaten the core design; all are afternoon-sized with the existing gates making verification cheap.

Two honest tensions are worth accepting rather than fixing. The 90-combo monomorphization buys uniform testing and the "registered everywhere" guarantee at a compile-time/binary cost that will scale multiplicatively with new components — it deserves a stated budget, not removal. And the three ~140-line `grow()` bodies resist unification for a reason the decision log already records: they are three genuinely different control flows sharing a vocabulary, and forcing them into one template was tried and made things worse. The scaffolding around them (OOB walks, setup laps) can be shared; the strategies themselves should stay separate and legible — they are, after all, the chapters of the guide.
