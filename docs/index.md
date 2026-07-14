# bonsai

**Histogram gradient-boosted trees with a C++23 core — written to be read.**

```
pip install <wheel from the latest release>   # Linux, macOS arm64, Python 3.9–3.13, no toolchain
```

bonsai is a from-scratch GBT library small enough to read in a sitting and fast enough to take seriously: on GPU it holds the fastest slot at every row scale we measured — edging catboost and beating xgboost at 16M rows at matched accuracy, on ~3× less memory — and every one of those words links to a reproducible run.

It exists because the great boosting libraries are wonderful to use and nearly impossible to read, and understanding them shouldn't require archaeology.

## This site is a love story

xgboost proved boosting could be an industrial tool; lightgbm made it fast enough to be a default; catboost made it careful about its own biases.

bonsai is built out of their ideas — adopted where measurement agreed, rebuilt smaller where it didn't, declined with evidence where the benefit failed to reproduce — and this site narrates those debts alongside the API, because a library that stands on three giants should say so in its documentation, not just its bibliography.

The claims stay honest in both directions: where bonsai wins, the run is linked; where it still loses — catboost on wide data, xgboost's last +0.001 r² of cut quality — that's linked too.

## Four doors

**[Learn](guide/README.md)** — gradient boosting from math to code: each chapter takes one concept from intuition, through the mathematics, to the ~50 real lines that implement it, to an experiment against the reference libraries. Start with [a tree traced by hand on eight rows](guide/0-a-tree-by-hand.md).

**Use** *(being written)* — the whole API in one mental model: sklearn-shaped estimators and an explicit `train(params, ...)` layer over the same engine, dotted config keys that are exactly the CLI's `--set` keys, one `.msgpack` model that round-trips everywhere.

**[Lineage](lineage/catboost.md)** — what bonsai owes each of its ancestors, idea by idea: adopted, rebuilt, or measured and respectfully declined. [CatBoost](lineage/catboost.md) is written; xgboost and lightgbm are next.

**Method** *(being written)* — the data-driven HPC discipline behind the results, portable to other systems: instrument-first optimization, a feature-admission gate with pre-registered kill criteria, same-pod benchmarking, and bit-exact determinism as a testable contract.

## The engineering notebook

The [decisions log](decisions.md) is the project's raw narrative — sixty-six numbered decisions including the refuted hypotheses, kept because a result you can't audit is advertising.

The [architecture notes](architecture/README.md) are its structured companion: one document per subsystem, written once, referenced instead of repeated.
