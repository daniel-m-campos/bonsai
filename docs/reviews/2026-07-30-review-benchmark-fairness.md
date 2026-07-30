# Adversarial review: benchmark fairness to the reference libraries

2026-07-30, on Daniel's request. Lens: where does the harness quietly favor bonsai. Scope: `bonsai.bench` (params, scaling, grinsztajn, airline), `scripts/compare.py`, `scripts/bench_gpu.py`, `scripts/gpu_pareto.py`. Fixes shipped in the same PR; the airline re-check ran on a rented pod (result recorded below).

## What held up under attack

Timed spans are symmetric: each library's own ingest sits inside `fit_s` (bonsai binning, XGBoost QuantileDMatrix, LightGBM Dataset, CatBoost Pool) and each predicts through its fast path, including XGBoost's `inplace_predict`. The CUDA warmup micro-fit applies to every GPU variant. CatBoost runs with `allow_writing_files=False`; LightGBM gets `deterministic=True` where quality is compared; every variant runs in its own subprocess so peak RSS is per-library; rows record library versions (runlog v1). The one knob known to translate ambiguously (`min_child_weight`) is bracketed by a second committed run and the README states the rank swap under the other reading. Ordinal categorical codes for every library are the documented convention (decision 68) with the labeled `ts_` exception, and decisions 80-85 quantified what the convention costs CatBoost; that is disclosure working as intended.

## Findings and resolutions

**1. Airline and gpu-pareto short-changed XGBoost (and LightGBM) by one bin.** `min(bins, 254)` — CatBoost's GPU border cap — was applied to the XGBoost and LightGBM `max_bin` in `airline.py` and to `bins_effective` for all references in `gpu_pareto.py`, so those libraries ran 254 bins where bonsai ran 255, in suites with cited quality cells. Fixed: the clamps are gone; references get the same bin count as bonsai. The airline reference arms were re-run at matched bins on a rented L40S to check whether any cited AUC cell moves (see the re-check note below); the gpu-pareto frontier is annotated for re-measurement on its next refresh rather than re-rented now (one bin at 16M rows sits far below the frontier's quoted resolution, and the frontier's headline is wall-clock shape, not the fourth AUC decimal).

**2. The CatBoost borders-vs-bins fencepost was translated at call sites, three different ways.** `border_count` counts splits; `max_bin` counts bins. grinsztajn passed bins-1 (correct), scaling CPU and bench_gpu passed bins (CatBoost got one MORE bin than everyone), airline passed 254 (correct by accident of the double clamp). Fixed: `catboost_core` now takes bin semantics and derives borders internally, with the GPU cap; every call site passes the plain bin count. `compare.py` never set `border_count` at all and rode CatBoost's default 254 borders = 255 bins, which happens to be matched; it now goes through nothing new but is recorded here as accidental-but-correct.

**3. Classification leaf floors are not equivalent, and the bracket never hits the equivalent point.** Under logloss, XGBoost's `min_child_weight=20` is hessian mass (worth 80+ rows at p=0.5), bonsai and LightGBM have a 20-row floor with a loose hessian floor, and CatBoost's symmetric-tree policy supports no per-leaf floor at all. The two published XGBoost runs (20 and 1) bracket the equivalent point (~20 x mean hessian) without hitting it. Resolution: stated in the protocol page rather than re-tuned; matched-knobs is the design and the bracket bounds the effect, but the caveat now names both the XGBoost bracketing and CatBoost's missing floor.

**4. Latent thread asymmetry in grinsztajn's recorded (unpublished) timings.** References were pinned to 8 threads; bonsai's estimator defaulted to auto (up to 16). Published standings are quality-only, so no published number was affected, but every row carries a `fit_s` measured under asymmetric threading. Fixed: bonsai is pinned to 8 in `fit_predict` like the others. Same family, noted without action: `compare.py` uses plain `DMatrix` where scaling uses `QuantileDMatrix`; harmless while campaign timings stay unpublished (the protocol already forbids citing them).

## The airline re-check

The fix changes only the XGBoost and LightGBM airline arms (254 to 255 bins); the bonsai and CatBoost rows are bit-unaffected. The reference arms were re-run at matched bins (same suite code, one rented L40S) and compared against the committed `airline-2026-07.jsonl` AUC values; the verdict lives in the PR that shipped this review and is summarized in the ledger only if a cited cell moved.

## Standing rule

The fencepost lesson generalizes the params.py doctrine: a translation that must be remembered at every call site will eventually be remembered differently. Any knob whose semantics differ between libraries gets translated inside `bonsai.bench.params`, once, or it will drift.
