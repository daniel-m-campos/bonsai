<p align="center">
  <img src="https://raw.githubusercontent.com/daniel-m-campos/bonsai/main/docs/assets/bonsai-logo.png" alt="bonsai" width="640">
</p>

<p align="center">
  <b>A histogram gradient-boosted tree library and CLI in modern C++23.</b>
</p>

<p align="center">
  <a href="https://github.com/daniel-m-campos/bonsai/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/daniel-m-campos/bonsai/actions/workflows/ci.yml/badge.svg?branch=main"></a>
  <img alt="C++23" src="https://img.shields.io/badge/C%2B%2B-23-00599C?logo=cplusplus&logoColor=white">
  <img alt="CMake" src="https://img.shields.io/badge/CMake-%E2%89%A5%203.25-064F8C?logo=cmake&logoColor=white">
  <a href="https://github.com/daniel-m-campos/bonsai/releases/latest"><img alt="Release" src="https://img.shields.io/github/v/release/daniel-m-campos/bonsai"></a>
  <a href="https://github.com/daniel-m-campos/bonsai/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
</p>

<p align="center">
  <a href="https://daniel-m-campos.github.io/bonsai/"><b>Documentation</b></a> &nbsp;·&nbsp;
  <a href="https://daniel-m-campos.github.io/bonsai/use/install/">Install</a> &nbsp;·&nbsp;
  <a href="https://daniel-m-campos.github.io/bonsai/guide/">Guide</a> &nbsp;·&nbsp;
  <a href="https://daniel-m-campos.github.io/bonsai/decisions/">Decisions</a> &nbsp;·&nbsp;
  <a href="https://github.com/daniel-m-campos/bonsai/releases/latest">Releases</a>
</p>

---

## What is bonsai?

bonsai is a from-scratch, histogram-based gradient boosted trees (GBT) library and command-line tool written in C++23. It pairs a small, concept-checked component API (objectives, growers, split finders, samplers) with compile-time dispatch in the training hot path, and ships the benchmark harness that pits it against XGBoost, LightGBM, and CatBoost on real data. The aim is a readable, thoroughly documented GBT: a reference-grade implementation that competes with the production libraries instead of merely tolerating comparison with them.

- **Compile-time dispatch, concept-checked components.** The runtime TOML config resolves once to a monomorphized `Booster<Objective, Grower, Splitter, Sampler>`; no virtual calls in the hot path, and contract violations fail at compile time. Adding a component is a [short recipe](https://daniel-m-campos.github.io/bonsai/use/building/#extending-bonsai).
- **Six growers, one engine.** `depthwise` (XGBoost-style), `leafwise` (LightGBM-style), `oblivious` (CatBoost-style), and their CUDA twins `cuda_depthwise` / `cuda_leafwise` / `cuda_oblivious`; with 7 objectives and 3 samplers the dispatch space is 126 statically-typed combinations, selectable per run from config.
- **Deterministic parallelism.** Models are bit-identical across runs, thread counts, and even CPU architectures (arm64 == x86-64), a property no reference library offers, enforced per-commit in CI ([the contract](https://daniel-m-campos.github.io/bonsai/design/determinism/)).
- **A guide, not just docs.** [The guide](https://daniel-m-campos.github.io/bonsai/guide/) explains gradient boosting chapter by chapter: concept, math, then the ~50 real lines that implement it here, then an experiment against the reference libraries.

## Install

```
pip install bonsai-gbt
```

Wheels cover Linux x86_64/aarch64 and macOS arm64, Python 3.9 to 3.13, no toolchain needed. The linux x86_64 wheel is CUDA-enabled at 2.3MB total: GPU training works out of the box on any NVIDIA driver R525+, it behaves exactly like a CPU wheel on machines without a GPU, and every release's CUDA wheel passes a live GPU validation before it ships ([decision 70](https://daniel-m-campos.github.io/bonsai/decisions/)). The full story, docker image included, is [Install](https://daniel-m-campos.github.io/bonsai/use/install/); everything past a wheel (the CLI binary, development setups, CUDA source builds) is [Building from source](https://daniel-m-campos.github.io/bonsai/use/building/).

## Quick start

```python
import bonsai

model = bonsai.BonsaiRegressor(
    n_iters=200, learning_rate=0.05, grower="leafwise",
    early_stopping_rounds=20,
    params={"tree.lambda_l1": 0.5},   # any dotted config key the CLI accepts
)
model.fit(X_train, y_train, eval_set=(X_valid, y_valid))
pred = model.predict(X_test)
model.save("model.msgpack")           # loadable by `bonsai predict` and vice versa
```

The CLI (a [source-build artifact](https://daniel-m-campos.github.io/bonsai/use/building/)) drives the same engine with the same keys and the same models:

```
bonsai fit      -c CONFIG --model OUT.msgpack
bonsai predict  -c CONFIG --model IN.msgpack [--data CSV] --out PREDS.csv
bonsai eval     -c CONFIG --model IN.msgpack [--data CSV]
bonsai info                        # list (objective, grower, sampler) combos
bonsai params                      # dump the default config as TOML
```

Any key overrides inline (`bonsai fit -c config.toml --set tree.max_depth=8 --set dispatch.grower_name=oblivious ...`), and `make fit-benchmark` trains and times bonsai against xgboost/lightgbm/catboost on California Housing in one command. The rest of the API is one read: [the API tour](https://daniel-m-campos.github.io/bonsai/use/api-tour/).

## Results

Two divisions, per the [benchmark charter](https://daniel-m-campos.github.io/bonsai/method/benchmark-protocol/): perf (latency and memory, accuracy as a sanity guard) and quality (accuracy, timing never citable). The evidence is [the results ledger](https://daniel-m-campos.github.io/bonsai/method/results/), one generated page per study.

<!-- standings:begin (generated by scripts/render_results.py) -->

### Perf

bonsai's CUDA growers hold the fastest slot at every measured row scale (10.5s at 16M rows against XGBoost-GPU's 35.7s) at 6.9GB peak host memory against XGBoost's 22.1GB and CatBoost's 19.2GB. On the narrow airline shape bonsai holds both best AUC and fastest fit from 1M rows up. The 2026-07-30 studies hold every width and aspect ratio, with measured device memory that sizes to the problem: 3.4GB at 16M x 128 at constant 2^31-cell volume against XGBoost's 18.9GB and CatBoost's 90.2GB. Every number is same-pod; identical-model GPUs across the rental fleet measure up to ~25% apart.

Same-pod re-baseline ladder, best of repeats, test r² in parentheses, fastest per row in bold. Measured at `77633a6` (2026-08-02, pod-NVIDIA-L40S).

| rows | bonsai cuda dw | bonsai cuda obl | bonsai cuda lw | xgb cuda | catboost gpu | lgbm cuda | lgbm cpu | bonsai cpu obl |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 250k | **0.4s** (.872) | 0.9s (.877) | 1.9s (.872) | 1.1s (.872) | 1.9s (.875) | 6.4s (.879) | 4.4s (.872) | 8.1s (.877) |
| 1M | **0.8s** (.877) | 1.2s (.877) | 2.5s (.877) | 2.6s (.876) | 2.7s (.876) | 7.6s (.884) | 8.9s (.877) | 12.7s (.877) |
| 4M | **3.1s** (.878) | 3.1s (.876) | 4.9s (.878) | 9.6s (.878) | 6.4s (.877) | 12.0s (.885) | 35.7s (.879) | 34.5s (.876) |
| 16M | 12.1s (.879) | **10.5s** (.877) | 13.8s (.879) | 35.7s (.880) | 24.3s (.876) | 31.8s (.886) | 211.1s (.879) | 109.2s (.877) |

The width, shape, and accuracy-time frontier tables live in [the ledger](https://daniel-m-campos.github.io/bonsai/method/results/).

### Quality

On the [Grinsztajn et al. tabular benchmark](https://arxiv.org/abs/2207.08815) (55 OpenML tasks selected by third parties, three seeds, matched knobs, best variant per library), bonsai takes the best mean rank with 36 outright wins:

| library | mean rank | outright wins |
|---|--:|--:|
| **bonsai** | **1.44** | **36** |
| lightgbm | 2.51 | 5 |
| xgboost | 2.84 | 6 |
| catboost | 3.22 | 8 |

<!-- standings:end -->
The one knob that translates ambiguously between libraries is bracketed in both directions on [the standings page](https://daniel-m-campos.github.io/bonsai/method/results/quality-grinsztajn/); reproduce with `pip install bonsai-gbt[bench]`, then `python -m bonsai.bench.grinsztajn out.jsonl` to run the suite and `python -m bonsai.bench.grinsztajn out.jsonl --report` to render the standings.

Every headline claim links a reproducible run and the decision that records it: [claims and proofs](https://daniel-m-campos.github.io/bonsai/method/).

## Documentation

The home is **[daniel-m-campos.github.io/bonsai](https://daniel-m-campos.github.io/bonsai/)**, four doors:

- **[Learn](https://daniel-m-campos.github.io/bonsai/guide/)**: gradient boosting from intuition to the shipping code, one concept per chapter, each with an experiment against the reference libraries.
- **[Use](https://daniel-m-campos.github.io/bonsai/use/install/)**: [install](https://daniel-m-campos.github.io/bonsai/use/install/), [the API in one read](https://daniel-m-campos.github.io/bonsai/use/api-tour/), [parameters](https://daniel-m-campos.github.io/bonsai/use/parameters/), [building from source](https://daniel-m-campos.github.io/bonsai/use/building/).
- **[Results](https://daniel-m-campos.github.io/bonsai/method/)**: the measurement discipline; its rules in the [benchmark charter](https://daniel-m-campos.github.io/bonsai/method/benchmark-protocol/), its evidence in [the results ledger](https://daniel-m-campos.github.io/bonsai/method/results/).
- **[Design](https://daniel-m-campos.github.io/bonsai/design/system-map/)**: the system map, the concept-to-type tour, determinism as a contract, and the archive ([decisions log](https://daniel-m-campos.github.io/bonsai/decisions/), [architecture notes](https://daniel-m-campos.github.io/bonsai/architecture/)).

Repo-only records, unpublished by design: the [project retrospective](https://github.com/daniel-m-campos/bonsai/blob/main/docs/report.md), the [original proposal](https://github.com/daniel-m-campos/bonsai/blob/main/docs/proposal.md), and the [context/roadmap](https://github.com/daniel-m-campos/bonsai/blob/main/docs/context.md).

## Project layout

```
include/bonsai/   public headers (Booster, Tree, Grower, Sampler, …)
src/              implementation + CLI (src/cli/)
python/           the bonsai package (bindings, encoding, bonsai.bench)
tests/unit/       Catch2 unit + parity tests (ctest)
benchmarks/       evidence docs + committed results data
scripts/          uv-managed Python: compare.py, probes, render_results.py
configs/          example TOML configs
docs/             the documentation site source + design records
```

## License

MIT © 2026 Daniel M Campos. See [LICENSE](https://github.com/daniel-m-campos/bonsai/blob/main/LICENSE).
