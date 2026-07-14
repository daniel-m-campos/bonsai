# bonsai

**Histogram gradient-boosted trees with a C++23 core — written to be read.**

bonsai is a from-scratch GBT library that trains as fast as the production libraries — on GPU, faster at every row scale we measured — while staying small enough to actually read. Every performance and quality claim links to a reproducible run and the decision that recorded it.

This site is being assembled from the project's living documents. Four doors are planned; two are open today.

## Learn

Gradient boosting from math to code: each chapter takes one concept from intuition, through the mathematics, to the ~50 real lines that implement it here, to an experiment you can run against xgboost, lightgbm, and catboost.

Start with [chapter 0 — a tree by hand](guide/0-a-tree-by-hand.md): one boosting round traced on eight rows.

## Design

How the implementation landed where it did, and what it owes to the libraries that came before it — the [architecture notes](architecture/README.md) and the full [decisions log](decisions.md), including the hypotheses that were refuted along the way. Refutations are deliverables here.

## Use *(coming)*

The API tour: two layers over one engine — scikit-learn-shaped estimators for pipelines, an explicit `train(params, ...)` layer whose dotted keys are exactly the CLI's `--set` keys, and one `.msgpack` model format that round-trips everywhere. Prebuilt wheels: `pip install`, no toolchain.

## Method *(coming)*

The data-driven HPC/ML discipline behind the results, written to transfer to other systems: instrument-first optimization (decompose → price → implement → validate → record), the feature-admission gate (prototype at zero core cost, pre-registered kill criteria), same-pod benchmarking, and bit-exact determinism as a testable contract.
