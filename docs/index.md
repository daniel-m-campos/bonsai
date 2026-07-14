# bonsai

**Histogram gradient-boosted trees with a C++23 core.**

```
pip install <wheel from the latest release>   # Linux, macOS arm64, Python 3.9–3.13, no toolchain
```

bonsai is a from-scratch gradient-boosted trees library built around two problems the established libraries don't solve: their implementations are effectively unreadable, and their performance claims are hard to reproduce.

It incorporates the core ideas of xgboost, lightgbm, and catboost — histogram training, leaf-wise and symmetric tree growth, leak-free target statistics — in a codebase small enough to read in a sitting, and it competes with them directly: on GPU it holds the fastest slot at every row scale we measured, edging catboost and beating xgboost at 16M rows at matched accuracy, on ~3× less host memory.

Every performance and quality claim links to a reproducible run, in both directions: where bonsai wins, the run is linked; where it still loses — catboost on wide data, xgboost's last +0.001 r² of cut quality — that is linked too.

It also offers something none of the reference libraries do: models that are bit-identical across CPU architectures and across thread counts, enforced per-commit in CI.

## Four doors

**[Learn](guide/README.md)** — gradient boosting from math to code: each chapter takes one concept from intuition, through the mathematics, to the ~50 real lines that implement it, to an experiment against the reference libraries. Start with [a tree traced by hand on eight rows](guide/0-a-tree-by-hand.md).

**Use** *(being written)* — the whole API in one mental model: sklearn-shaped estimators and an explicit `train(params, ...)` layer over the same engine, dotted config keys that are exactly the CLI's `--set` keys, one `.msgpack` model that round-trips everywhere.

**[Lineage](lineage/catboost.md)** — where each major idea came from and what happened when we measured it here: adopted, rebuilt smaller, or declined with the evidence recorded. [CatBoost](lineage/catboost.md) is written; xgboost and lightgbm are next.

**Method** *(being written)* — the measurement discipline behind the results, portable to other performance-critical systems: instrument-first optimization, a feature-admission gate with pre-registered kill criteria, same-pod benchmarking, and bit-exact determinism as a testable contract.

## The engineering notebook

The [decisions log](decisions.md) records the project's sixty-six numbered decisions, including the hypotheses that measurement refuted — the raw material behind every claim on this site.

The [architecture notes](architecture/README.md) are its structured companion: one document per subsystem, written once, referenced instead of repeated.
