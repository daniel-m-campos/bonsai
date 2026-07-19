<p align="center">
  <img src="docs/assets/bonsai-logo.png" alt="bonsai" width="640">
</p>

<p align="center">
  <b>A histogram gradient-boosted tree library and CLI in modern C++23.</b>
</p>

<p align="center">
  <a href="https://github.com/daniel-m-campos/bonsai/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/daniel-m-campos/bonsai/actions/workflows/ci.yml/badge.svg?branch=main"></a>
  <img alt="C++23" src="https://img.shields.io/badge/C%2B%2B-23-00599C?logo=cplusplus&logoColor=white">
  <img alt="CMake" src="https://img.shields.io/badge/CMake-%E2%89%A5%203.25-064F8C?logo=cmake&logoColor=white">
  <a href="https://github.com/daniel-m-campos/bonsai/releases/latest"><img alt="Release" src="https://img.shields.io/github/v/release/daniel-m-campos/bonsai"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
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
- **Five growers, one engine.** `depthwise` (XGBoost-style), `leafwise` (LightGBM-style), `oblivious` (CatBoost-style), and their CUDA twins `cuda_depthwise` / `cuda_oblivious`; with 7 objectives and 3 samplers the dispatch space is 105 statically-typed combinations, selectable per run from config.
- **Deterministic parallelism.** Models are bit-identical across runs, thread counts, and even CPU architectures (arm64 == x86-64), a property no reference library offers, enforced per-commit in CI ([decisions 49/59/60](https://daniel-m-campos.github.io/bonsai/decisions/)).
- **A guide, not just docs.** [The guide](https://daniel-m-campos.github.io/bonsai/guide/) explains gradient boosting chapter by chapter: concept, math, then the ~50 real lines that implement it here, then an experiment against the reference libraries.

## Install

```
pip install bonsai-gbt --find-links https://github.com/daniel-m-campos/bonsai/releases/expanded_assets/v1.4.0
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

## Performance

Two divisions, per the [benchmark charter](https://daniel-m-campos.github.io/bonsai/method/benchmark-protocol/): quality (accuracy, timing never citable) and perf (latency and memory, quality as a sanity guard). The complete evidence, generated from every committed results file, is [the results ledger](https://daniel-m-campos.github.io/bonsai/method/results/).

### Quality: the external standings

On the [Grinsztajn et al. tabular benchmark](https://arxiv.org/abs/2207.08815), 55 OpenML tasks selected by third parties, three seeds, matched campaign knobs, best variant per library ([decision 68](https://daniel-m-campos.github.io/bonsai/decisions/), [evidence](benchmarks/grinsztajn-2026-07.md)):

| library | mean rank | outright wins |
|---|--:|--:|
| **bonsai** | **1.44** | **36** |
| lightgbm | 2.51 | 5 |
| xgboost | 2.84 | 6 |
| catboost | 3.22 | 8 |

The one knob that translates ambiguously between libraries is bracketed: under XGBoost's own `min_child_weight=1` convention the top two swap on mean rank (2.11 vs 2.04) while bonsai keeps the most second-or-better finishes; both runs are in the ledger. Reproduce either: `pip install bonsai-gbt[bench]`, then `python -m bonsai.bench.grinsztajn out.jsonl --report`.

### Perf: fit seconds at scale

From the same-pod 2026-07 re-baseline (dual-EPYC-9554 host with an L40S; `fit()` timed end to end including each library's own ingest; test r² in parentheses; [raw runs](benchmarks/results/rebaseline-2026-07.jsonl)):

| rows | bonsai cuda dw | bonsai cuda obl | xgb cuda | catboost gpu | lgbm cpu | bonsai cpu obl |
|---|--:|--:|--:|--:|--:|--:|
| 250k | **0.5s** (.871) | 1.0s (.875) | 0.8s (.872) | 1.6s (.875) | 2.5s (.872) | 5.2s (.875) |
| 1M | **1.1s** (.876) | 1.4s (.876) | 1.7s (.876) | 2.3s (.876) | 5.0s (.877) | 7.3s (.876) |
| 4M | 4.5s (.878) | **4.4s** (.875) | 5.3s (.878) | 5.0s (.877) | 19.9s (.879) | 20.2s (.875) |
| 16M | 20.5s (.879) | **18.4s** (.876) | 19.9s (.880) | 18.5s (.876) | 111.3s (.879) | 73.3s (.876) |

Honest caveats, because benchmarks without them are advertising: identical-model GPUs across the rental fleet measure up to ~25% apart, so only same-pod columns compare. bonsai owns the fastest slot at every row scale, edging CatBoost at 16M (18.4 vs 18.5s, both .876) and beating XGBoost-GPU (19.9s); on wide data CatBoost keeps the lead, with bonsai second (the cols-scaling table is in [the ledger](https://daniel-m-campos.github.io/bonsai/method/results/)). Off the fixed-iteration axis, the 16M accuracy-vs-time frontier now belongs to bonsai: fastest to every measured accuracy up to ~.895 r², a statistical tie with CatBoost through the .897-.898 plateau, and the measured ceiling (.8981); CatBoost's remaining edge is a cheaper marginal round that pays off only past ~450 rounds at this scale (decision 72, chart in the ledger). Peak host RSS at 16M is 7.0GB vs XGBoost's 22.2GB and CatBoost's 19.4GB, and predict is ~3x faster. Two earlier apparent gaps against CatBoost were bonsai bugs, since fixed (decisions 63/64); the path from 3x behind to this table is [guide chapter 11](https://daniel-m-campos.github.io/bonsai/guide/11-performance-engineering/).

## Claims and proofs

Every claim links a reproducible run and the decision that records it; the point of a small, measured library is that you can check it.

| Claim | Evidence |
|---|---|
| **Bit-identical models across CPU architectures** (arm64 == x86-64) at a fixed thread count; no reference library offers this | decisions [59/60](https://daniel-m-campos.github.io/bonsai/decisions/); asserted per-commit by [`cross-arch.yml`](.github/workflows/cross-arch.yml) via [`scripts/model_hash.py`](scripts/model_hash.py) |
| **Best mean rank on the 55-task Grinsztajn benchmark under either min_child_weight convention** (36 outright wins; second-or-better on 50/55, never last) | [grinsztajn-2026-07](benchmarks/grinsztajn-2026-07.md), [decision 68](https://daniel-m-campos.github.io/bonsai/decisions/) |
| **Fastest GPU slot at every row scale**; at 16M `oblivious` edges catboost and beats xgboost-GPU at matched accuracy | [rebaseline jsonl](benchmarks/results/rebaseline-2026-07.jsonl), [scale-edge](benchmarks/catboost-scale-edge-2026-07.md), decisions [62-64](https://daniel-m-campos.github.io/bonsai/decisions/) |
| **The only GBT whose GPU path ships in a 2.3MB pip install, validated on live GPU hardware per release** | [decision 70](https://daniel-m-campos.github.io/bonsai/decisions/); [`wheels.yml`](.github/workflows/wheels.yml) |
| **Within ~8% of xgboost-hist at 16M rows on CPU, host-dependent**: a dead tie on one pod, xgboost ahead on another | [decision 61](https://daniel-m-campos.github.io/bonsai/decisions/); [prefetch-round jsonl](benchmarks/results/cpu-prefetch-round-2026-07.jsonl) |
| **Best library on 9 of 10 datasets of the internal quality campaign** | [quality-campaign](benchmarks/quality-campaign-2026-07.md), decisions [56-57](https://daniel-m-campos.github.io/bonsai/decisions/) |
| **Categorical parity with catboost within chance-band**, via preprocessing not an engine feature | [decision 58](https://daniel-m-campos.github.io/bonsai/decisions/); [categorical-tradeoff](benchmarks/categorical-tradeoff-2026-07.md); [`encoding.py`](python/bonsai/encoding.py) |
| **~3x less host memory than xgboost at 16M** (7.0 vs 22.2GB) and ~3x faster predict | [rebaseline jsonl](benchmarks/results/rebaseline-2026-07.jsonl) |
| **Ranking is a measured, scoped gap**: ~+0.015 NDCG@10 to a listwise loss, not pairwise LambdaRank | [ranking-tradeoff](benchmarks/ranking-tradeoff-2026-07.md); [`probe_ranking.py`](scripts/probe_ranking.py) |
| **Every feature earns its place by measurement**; refutations are recorded too | the [feature-admission gate](https://daniel-m-campos.github.io/bonsai/method/how-we-decide/); declines in decisions 58/62/67 |

## Documentation

The home is **[daniel-m-campos.github.io/bonsai](https://daniel-m-campos.github.io/bonsai/)**, four doors and a notebook:

- **[Learn](https://daniel-m-campos.github.io/bonsai/guide/)**: gradient boosting from intuition to the shipping code, one concept per chapter, each with an experiment against the reference libraries.
- **[Use](https://daniel-m-campos.github.io/bonsai/use/install/)**: [install](https://daniel-m-campos.github.io/bonsai/use/install/), [the API in one read](https://daniel-m-campos.github.io/bonsai/use/api-tour/), [building from source](https://daniel-m-campos.github.io/bonsai/use/building/).
- **[Lineage](https://daniel-m-campos.github.io/bonsai/lineage/xgboost/)**: what XGBoost, LightGBM, and CatBoost each contributed, and whether measurement adopted, rebuilt, or declined it.
- **[Method](https://daniel-m-campos.github.io/bonsai/method/)**: the measurement discipline, portable beyond GBTs; its rules in the [benchmark charter](https://daniel-m-campos.github.io/bonsai/method/benchmark-protocol/) and its evidence in [the results ledger](https://daniel-m-campos.github.io/bonsai/method/results/).
- The engineering notebook: the [decisions log](https://daniel-m-campos.github.io/bonsai/decisions/) and the [architecture notes](https://daniel-m-campos.github.io/bonsai/architecture/).

Repo-only records, unpublished by design: the [project retrospective](docs/report.md), the [original proposal](docs/proposal.md), and the [context/roadmap](docs/context.md).

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

MIT © 2026 Daniel M Campos. See [LICENSE](LICENSE).
