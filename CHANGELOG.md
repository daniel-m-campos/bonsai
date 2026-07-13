# Changelog

All notable changes to bonsai. Format loosely follows [Keep a Changelog](https://keepachangelog.com/); versions are git tags. Design rationale for anything below lives in [`docs/decisions.md`](docs/decisions.md).

## [1.1.0] — 2026-07-13

The crown-week release: measured parity-or-better against xgboost, lightgbm, and catboost, plus a scikit-learn-shaped Python surface.

### Added
- **`BonsaiClassifier`** — sklearn-style classifier over the engine's `logloss` (binary) and `softmax` (multiclass) objectives. Binary `predict_proba`; multiclass `predict` returns labels (multiclass `predict_proba` is a tracked follow-up). Arbitrary label types are encoded/decoded via `classes_`.
- **scikit-learn estimator compatibility** for `BonsaiRegressor`/`BonsaiClassifier` — `get_params`/`set_params`/`score`, and drop-in use in `clone`, `Pipeline`, `GridSearchCV`, `cross_val_score`, and `pickle` — implemented **without a scikit-learn runtime dependency** (`import bonsai` never imports sklearn).
- **`sample_weight`** on `BonsaiRegressor.fit` / `bonsai.train` (sklearn convention) — per-row weighting of gradients and hessians.
- **`OrderedTargetEncoder`** — leak-free ordered target statistics for categorical features, including `cross=2` pair encodings (decision 58; guide chapter 13).
- **Poisson** regression objective (closes #44).

### Changed / Performance
- **Binning: one shared row sample for the whole matrix** instead of a per-feature reservoir pass — 24× faster mapper-fit at 16M, quality-neutral (decision 64).
- **CPU fill loop software prefetch** — the 16M-row fit now ties xgboost-hist (decision 61).
- **Fresh same-pod re-baseline** (decision 62–64): bonsai's GPU `oblivious` grower now **edges catboost and beats xgboost-GPU at 16M** at matched accuracy and ~3× less host memory, and holds the fastest slot at every row scale.

### Fixed
- **GPU `oblivious` grower** carried a split-selection defect (a missing port of the issue-#60 fix) that silently cost ~0.011 test r² at depth ≥ 5; now matches its CPU twin exactly (decision 63).
- **Cross-architecture bit determinism**: models are byte-identical across arm64/x86-64 at a fixed thread count — no floating-point contraction on the host plane (decisions 59–60), enforced by a cross-arch CI gate.
- **OpenMP build variance can no longer be silent** — a missing OpenMP is now a hard configure error, not a quiet serial fallback that changed model bits (decision 60).
- Quality-campaign correctness fixes: count-weighted cuts for heavy-value columns (#63/decision 57), one cut per distinct value on duplicate-heavy columns (#61), infeasible frontier nodes contribute zero gain rather than vetoing the split (#60), and the true diagonal softmax hessian p(1−p) (#62).

### Docs
- New guide material (categoricals, structure-vs-scheme) and a **claims-and-proofs** table in the README linking every performance/quality claim to a reproducible run and the decision that records it.
- Note: `pyproject.toml` had drifted to `0.6.0` after the `v1.0.0` tag; corrected to `1.1.0` here.

## [1.0.0] — 2026-07-11

First 1.0 release: histogram gradient-boosted trees with a C++23 core, three growers (depthwise / leafwise / oblivious), CPU and CUDA backends, a concept-checked component API, a CLI, and Python bindings.
