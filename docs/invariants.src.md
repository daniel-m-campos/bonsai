## Device planes

### multiclass-device-decline

A multiclass model declines the CUDA predict and TreeSHAP planes and rides the host bin walk; width-1 dense and levelwise models ride the device.

- code: `include/bonsai/booster.hpp` : `device_plan_input`
- code: `src/cuda/shap.cu` : `pack_shap_paths`
- code: `src/python/module.cpp` : `predict_on_device`
- elaboration: decisions 108 and 111; issue #407 tracks the multiclass half.

### device-paths-decline-never-error

Every device predict and SHAP path declines to the host walk, never errors, on capability shortfalls: no CUDA build or device, multiclass, a feature over 255 bins, merged path length over 32, foreign ingest plane, allocation failure. A missing-covers model stays a host-side error, a model defect rather than a capability gap.

- code: `src/cuda/predict.cu` : `cuda_predict_plan`
- code: `src/cuda/shap.cu` : `cuda_shap_plan`
- elaboration: decision 108.

### device-predict-bit-equality

Device predict is bit-equal to the host binned walk, row for row; the epilogue spells its rounding out because the kernel TU compiles with contraction on.

- code: `src/cuda/predict.cu` : `__fadd_rn`
- code: `include/bonsai/detail/bin_walk.hpp` : `split_bins`
- elaboration: decision 108.

### stub-trains-nothing-predicts-anywhere

The cuda growers are registered in every build; without a device, construction and load succeed and training throws with a message naming the fix. Models trained on a device predict on any build.

- code: `src/cuda/histogram_engine_stub.cpp` : `throw_unavailable`
- code: `include/bonsai/registry/typelists.hpp` : `CudaObliviousGrower`
- elaboration: decision 70.

### resident-objective-eligibility

The device-resident objective arms only for MSE, LogLoss, and Poisson, weighted or not, with no DART and a sampler that never reads gradient values; everything else takes the host path. `BONSAI_HOST_OBJECTIVE=1` forces the host path.

- code: `include/bonsai/booster.hpp` : `resident_begin`
- code: `src/cuda/detail/device_context.cu` : `resident_begin`
- elaboration: decision 78.

## Training semantics

### split-tie-break

Exact gain ties break to the earliest candidate: lowest feature id, then lowest bin, then default-left first. Load-bearing for the bit-identity contract; both scan orders preserve it.

- code: `src/split.cpp` : `update_best`
- elaboration: decision 16.

### subtraction-trick

Level batching populates only each smaller sibling's histograms; the larger sibling derives by cell-wise subtraction, on both planes. Node totals are computed at split time, never accumulated during the fill.

- code: `src/step/primitives.hpp` : `smaller_child`
- code: `src/step/primitives.hpp` : `finish_split`
- code: `include/bonsai/histogram.hpp` : `Histogram`
- elaboration: decision 19.

### gain-and-cover-stamped-at-grow

Split gains and per-node covers are stamped at grow time; neither is reconstructible from a stored tree, so a model saved without them loads with the dependent features (importance, TreeSHAP) declining and saying why.

- code: `include/bonsai/tree.hpp` : `covers`
- code: `src/shap_paths.cpp` : `pack_shap_paths`
- elaboration: the gain-storage entry in decisions.md.

### auto-thread-cap

`n_threads=0` resolves to hardware concurrency capped at `auto_thread_cap` (16) and further capped by any cgroup CPU quota; the quota read is what makes container fits honest.

- code: `include/bonsai/parallel.hpp` : `auto_thread_cap`
- elaboration: the threading entries in decisions.md.

### softmax-true-hessian

Multiclass softmax uses the true diagonal Hessian p(1-p), not the factor-2 convention, floored at 1e-6 before weighting.

- code: `include/bonsai/multiclass_booster.hpp` : `1e-6F`
- elaboration: the multiclass entry in decisions.md; [guide/12-multiclass.md](guide/12-multiclass.md) teaches it.

## Data plane

### host-determinism

Model bytes are identical across runs, across thread counts (outside one documented relaxation), and across CPU architectures; CI trains on arm64 and x86-64 and compares file hashes per commit. Host builds compile with contraction off, which is what makes the claim hold across compilers. A host fit and a device fit of one dataset agree to 1e-4 in predictions, never byte for byte: the host accumulates histogram cells in float and the device in int64 fixed point, and the device fit's own byte identity is the enforced `cuda-training-bit-reproducible`.

- code: `include/bonsai/parallel.hpp` : `for_each_index`
- code: `scripts/model_hash.py` : `sha256`
- elaboration: [learn/determinism-as-a-contract.md](learn/determinism-as-a-contract.md); decisions 59 and 60.

### device-binning-byte-identity

Device ingest produces bins byte-identical to the host mapper's transform: the kernel runs the same binary search over the same cuts, so a device-binned Dataset and a host-binned one are the same object to everything downstream.

- code: `src/cuda/detail/ingest_kernels.cuh` : `transform_bin`
- code: `src/bin_mapper.cpp` : `BinMapper::transform`
- elaboration: decision 99.

### routing-rule-one-source

A row routes left iff its bin is at or below the split bin, with the last bin following `default_left`. The device kernels mirror the host rule rather than reimplementing it, and the `[cuda]` parity suite fails when they diverge.

- code: `include/bonsai/dataset.hpp` : `routes_left`
- code: `src/cuda/detail/kernels.cuh` : `goes_left_dev`
- elaboration: decision 108's bit-equality claim rests on this.

### threshold-to-bin-one-inversion

A stored split threshold is turned back into a bin by exactly one function. The grower records `threshold = cuts()[bin]` and cuts are strictly increasing, so `lower_bound` recovers the bin exactly; DART, warm start, the device epilogue, and the SHAP packers all go through it rather than re-deriving the mapping.

- code: `include/bonsai/bin_mapper.hpp` : `bin_of_threshold`
- code: `src/cuda/predict.cu` : `bin_of_threshold`
- elaboration: an edit here is a wire-format change even though no byte layout moves.

### max-bin-default

`bin_mapper.max_bin` defaults to 255, which is what keeps the default bin plane u8; a wider setting switches storage to u16 and costs the device SHAP plane (8-bit intervals).

- code: `include/bonsai/config/bin_mapper_config.hpp` : `max_bin`
- code: `include/bonsai/bin_store.hpp` : `BinColumns`
- elaboration: [use/parameters.md](use/parameters.md) is the generated reference; prose says "255 by default" and links there.

### shap-additivity-exact

Host TreeSHAP satisfies the efficiency property exactly: sum(phi) plus the expected value reproduces the raw prediction, by construction of the closed form. The device fp32 path is tolerance-bound instead, and the raw-matrix fp64 host walk is the escape.

- code: `include/bonsai/shap.hpp` : `tree_shap`
- code: `src/shap.cpp` : `tree_expected_value`
- elaboration: decision 108; its status banner records the fp32 device bound reaching 1e-5 at 200 trees depth 10.

### dataset-not-picklable

Dataset objects, host-built or device-resident, do not pickle; rebuild from X and y in the target process. Model objects pickle.

- code: `src/python/module.cpp` : `__reduce__`
- elaboration: the pickling entry in decisions.md.

## Build and packaging

### wheel-arch-matrix

The linux x86_64 wheel fatbins the one kernel TU for sm_70 through sm_120 with a compute_90 PTX floor for forward-JIT, links cudart statically, and imports on GPU-less hosts.

- code: `CMakeLists.txt` : `BONSAI_CUDA_PTX_ARCH`
- elaboration: decision 70; [use/install.md](use/install.md) carries the support matrix.

### static-libomp

`BONSAI_OPENMP_STATIC=ON` links libomp into the module statically so bonsai and another OpenMP library (XGBoost, LightGBM) in one process cannot deadlock on two runtimes.

- code: `CMakeLists.txt` : `BONSAI_OPENMP_STATIC`
- elaboration: the libomp entry in decisions.md; [use/building.md](use/building.md) documents the flag.

### python-floor

Supported CPythons are 3.9 through 3.13 on every wheel platform; the 3.9 floor is why nanobind is pinned below 3.

- code: `pyproject.toml` : `requires-python`
- elaboration: [use/install.md](use/install.md) carries the matrix.

### eight-cli-subcommands

The CLI wires eight subcommands: fit, predict, eval, bench, dump, importance, info, params. `info` reads the same registry dispatch uses, so it cannot drift from what a build can construct.

- code: `src/cli/main.cpp` : `add_subcommand`
- elaboration: [guide/0-a-tree-by-hand.md](guide/0-a-tree-by-hand.md) walks them.

### multi-gpu-withdrawn

Multi-GPU's supported story is fit-parallelism: N independent single-GPU fits. The data-parallel engine reached measured parity and is withdrawn to a branch; config carries a singular `parallel.device_id`.

- code: `include/bonsai/config/parallel_config.hpp` : `device_id`
- elaboration: decisions 76 and 98.

## Deleted symbols

Symbols that docs or comments may still name, gone from the tree: `Dataset::is_categorical`, `Dataset::row_major_bins` (now `Dataset::mirror()`), `finalize_rows` (now `finalize_tree`), `populate_many`, `NoSampler` (now `AllRowsSampler`), `SplitCandidate` (now `SplitOutput`), the 4-axis dispatch (`Splitters` was never a registry axis). A doc resurrecting one of these is stale, not informative.

- code: `include/bonsai/dataset.hpp` : `mirror`
- code: `include/bonsai/sampler.hpp` : `AllRowsSampler`
- code: `include/bonsai/split.hpp` : `SplitOutput`
