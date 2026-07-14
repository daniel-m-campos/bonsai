# Changelog

All notable changes to bonsai. Format loosely follows [Keep a Changelog](https://keepachangelog.com/); versions are git tags. Design rationale for anything below lives in [`docs/decisions.md`](docs/decisions.md).

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
