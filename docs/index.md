# bonsai

**Histogram gradient-boosted trees with a C++23 core.**

```
pip install <wheel from the latest release>   # Linux, macOS arm64, Python 3.9–3.13, no toolchain
```

bonsai began as a learning project: rebuild gradient-boosted trees from first principles to understand how the production libraries actually work, in a small codebase that takes modern C++23 and software design as seriously as the algorithms.

The scope escalated. [Developed with Claude](about.md), milestones that were meant to be the finish line kept falling: CPU parity, then GPU parity, then GPU leads. The ambition grew with them: assimilate the defining ideas of xgboost, lightgbm, and catboost into one small library, match or beat their performance, and keep the code clean enough that reading it is still the point.

Where that landed, measured on shared hardware at matched settings: on GPU, bonsai holds the fastest slot at every row scale tested — edging catboost and beating xgboost at 16M rows at matched accuracy, on ~3× less host memory. Where it still loses — catboost on wide data, xgboost's last +0.001 r² of cut quality — the runs are linked with the same prominence as the wins.

One property none of the reference libraries offer: models are bit-identical across CPU architectures and thread counts, enforced per-commit in CI.

## Four doors

**[Learn](guide/README.md)** — the learning project, kept as a product: each chapter takes one concept from intuition, through the mathematics, to the ~50 real lines that implement it, to an experiment against the reference libraries. Start with [a tree traced by hand on eight rows](guide/0-a-tree-by-hand.md).

**Use** *(being written)* — the whole API in one mental model: sklearn-shaped estimators and an explicit `train(params, ...)` layer over the same engine, dotted config keys that are exactly the CLI's `--set` keys, one `.msgpack` model that round-trips everywhere.

**[Lineage](lineage/catboost.md)** — the assimilation, idea by idea: what each reference library contributed, and whether measurement here adopted it, rebuilt it smaller, or declined it with the evidence recorded. [CatBoost](lineage/catboost.md) is written; xgboost and lightgbm are next.

**Method** *(being written)* — the measurement discipline that governed the scope escalation, portable to other performance-critical systems: instrument-first optimization, a feature-admission gate with pre-registered kill criteria, same-pod benchmarking, and bit-exact determinism as a testable contract.

## The engineering notebook

The [decisions log](decisions.md) records the project's sixty-six numbered decisions, including the hypotheses that measurement refuted — the raw material behind every claim on this site.

The [architecture notes](architecture/README.md) are its structured companion: one document per subsystem, written once, referenced instead of repeated.
