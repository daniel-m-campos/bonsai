# Architecture

These notes are the historical engineering record, kept for reference and for agents working in the codebase. The curated design pages live in the [Design](../design/determinism.md) section.

Per-component design docs. Numbered roughly in build order. Source of truth for choices is [`../decisions.md`](../decisions.md); when a doc disagrees, decisions wins.

> **Naming note.** These bodies are kept as written, so they call the symmetric-tree grower `oblivious` and its CUDA twin `cuda_oblivious`. Those config names are now `levelwise` and `cuda_levelwise`; the C++ types (`ObliviousTree`, `ObliviousGrower`) keep the old spelling because the *tree* is oblivious while the *growth policy* is level-wise. See the CHANGELOG entry for the break.

> **Deleted-path note (2026-08-12).** The CUDA engine's CPU fallback is gone, so read docs 10, 11, 12, and 20 as history where they describe it: `begin_root` and `leaf_begin_root` no longer decline a tree to the host plane, they raise a `ConfigError` naming the shared-memory or pool limit and the `device="cpu"` remedy. With it went `CudaHistogramEngine`'s CPU member, the `cpu_fallback` line of the CUDA profile, and the `on_device` fork in both `LevelStep` and `LeafStep` (doc 12's "one irreducible runtime fork" is now no fork). The fallback trained a wrong model once (issue #12, fixed in cd4e726) and hid a large slowdown the rest of the time; every reference GPU trainer errors here instead.

> **Deleted-symbol note (2026-08-09).** An audit removed code these bodies name in the present tense. The bodies stay as written, so read the following as history: `finalize_rows` (docs 10, 12, 13, 14, 20) is gone from all six sites including the `GPULevelEngine` concept clause, because `finalize_tree` already downloads everything the epilogue needs; and `Dataset::is_categorical` with its backing flag (docs 1 and 17) is gone, having stayed always-false since doc 17's design was declined by measurement.

## Contents

| # | Doc | Status |
|---|---|---|
| 1 | [`1-dataset.md`](1-dataset.md): Dataset, BinMapper, readers | done |
| 2 | [`2-histogram.md`](2-histogram.md): gradient/hessian sums, subtraction, parallel reduce | done |
| 3 | [`3-tree.md`](3-tree.md): Tree concept, `DenseTree` + `ObliviousTree`, depth-wise + levelwise growers, histogram splitter | done |
| 4 | [`4-objective.md`](4-objective.md): Objective concept, MSE, logloss | done |
| 5 | [`5-booster.md`](5-booster.md): Booster, training loop, `update_one_iter` | done |
| 6 | [`6-dispatch.md`](6-dispatch.md): registry, runtime → static boundary | done |
| 7 | `7-parallel.md`: parallelism seam, determinism contract | done (decision 32) |
| 8 | [`8-config.md`](8-config.md): Config, TOML, CLI overrides | done |
| 9 | [`9-cli.md`](9-cli.md): subcommands, overrides, fit-time output | done |
| 10 | [`10-cuda.md`](10-cuda.md): GPU histogram backend: builder policy, level batching, runtime capability | done |
| 11 | [`11-gpu-resident.md`](11-gpu-resident.md): the GPU device plane: resident buffers, kernels, precision, cross-library results | done (decisions 40–42) |
| 12 | [`12-grower-backend.md`](12-grower-backend.md): grower data-plane: the `LevelStep` compile-time strategy | landed (decision 41) |
| 13 | [`13-device-residency.md`](13-device-residency.md): full device residency: gradients first, binning second | refuted (decision 52) |
| 14 | [`14-engine-narrative.md`](14-engine-narrative.md): one engine narrative: the level transaction | executed (decision 53) |
| 15 | [`15-device-binning.md`](15-device-binning.md): device binning: ingest joins the transaction narrative | implemented (decision 54) |
| 16 | [`16-compute-dag.md`](16-compute-dag.md): the compute DAG: placement as a first-class design object | framing (decision 54) |
| 17 | [`17-categorical-splits.md`](17-categorical-splits.md): native categoricals: Fisher set splits + ordered-TS sketch | declined by measurement (decision 58) |
| 18 | [`18-manual-bin-edges.md`](18-manual-bin-edges.md): explicit bin edges in the model artifact | shipped (decision 73) |
| 19 | [`19-multi-gpu.md`](19-multi-gpu.md): single-node multi-GPU: a data-parallel backend beside the single-GPU one | built and parked at measured parity (decision 76); reopener landed (decision 98) |
| 20 | [`20-cuda-leafwise.md`](20-cuda-leafwise.md): device leafwise: a slot-pool plane for best-first growth | admitted, stage 3 closed (decisions 97-98) |
| 21 | [`21-component-standings.md`](21-component-standings.md): component-timing standings: the DAG constants become the ledger | design (issue #323) |
| 22 | [`22-device-predict-shap.md`](22-device-predict-shap.md): device predict and device TreeSHAP: serving the resident Dataset | landed for width-1 dense models; pod ledger pending |

## Cross-cutting concerns

**Dispatch.** Static poly inside `Booster`, runtime at config boundary. Flat table over `cartesian_product_t<...>`; one vcall at boundary, zero inside `update_one_iter`. See [`6-dispatch.md`](6-dispatch.md) + decision 26.

**Threading.** Shipped as a single seam, not a backend concept (decision 32): `parallel::for_each_index` in `bonsai/parallel.hpp`, OpenMP body with serial fallback, `[parallel] n_threads` config. Every parallel site assigns each index to exactly one thread with no cross-thread reductions. Details in [`7-parallel.md`](7-parallel.md).

**Errors.** Component constructors validate their config slice, throw `ConfigError` with key path. No central validator. CLI top-level catches.

**Determinism contract** (decisions 32/59/60): model bits are a pure function of the input, the config, and the **configured thread count**: the row-parallel fill's block plan scales with `parallel.n_threads`, so different N legitimately produce different (equally valid) bits, but a fixed N reproduces byte-for-byte on **any machine and any architecture**: arm64 and x86-64 train identical models (`-ffp-contract=off`, decision 59), gated per commit by the cross-arch CI workflow. The original any-thread-count claim (decision 32) predates the row-parallel fill; a silent serial fallback that once let *builds* differ is now a hard configure error (decision 60).

**Precision.** Float storage, double accumulators. Matches xgb/lgbm.

