# The bonsai guide: gradient boosting from math to code

Production GBT libraries document their *parameters*; the papers document
the *math*; the code that connects them is hundreds of thousands of lines
tuned past readability. This guide is the missing middle, and it is the
reason bonsai exists: every chapter takes one concept from intuition,
through the (light) math, to the **actual lines in this repository that
implement it** (usually a few dozen), and ends with an experiment you can
run against xgboost, lightgbm, and catboost with the same knob turned.

The code the guide references is the shipping code, not a simplification.
When a chapter says "this is all there is to GOSS", the linked function is
the whole implementation.

## Chapters

| # | Chapter | One-line pitch |
|---|---------|----------------|
| 0 | [A tree by hand](0-a-tree-by-hand.md) | One boosting round on eight rows, every number traced |
| 1 | [Gradient boosting](1-gradient-boosting.md) | Why trees fit *gradients*, why second order, where leaf values come from |
| 2 | [Binning & histograms](2-binning-and-histograms.md) | Why 255 buckets beat exact splits, and the subtraction trick |
| 3 | [Finding splits](3-finding-splits.md) | The gain formula, one prefix scan, and where missing values go |
| 4 | [Growing trees](4-growing-trees.md) | Depth-wise vs best-first vs oblivious: three answers to "which leaf next?" |
| 5 | [Sampling](5-sampling.md) | Training on fewer rows: Bernoulli, GOSS, and a bug worth learning from |
| 6 | [Regularization & constraints](6-regularization-and-constraints.md) | L1/L2, column sampling, monotone and interaction constraints |
| 7 | [Early stopping & DART](7-early-stopping-and-dart.md) | Knowing when to stop, and dropout for trees |
| 8 | [Feature importance](8-feature-importance.md) | Split vs gain, why they disagree, and what to distrust |
| 9 | [Parallelism & determinism](9-parallelism-and-determinism.md) | Deterministic models at a fixed thread count, and what that costs |
| 10 | [GPU training](10-gpu-training.md) | Where the host/device boundary goes, and the precision scheme that makes it honest |
| 11 | [Performance engineering](11-performance-engineering.md) | The compute-DAG method: price moves before playing them |
| 12 | [Multiclass](12-multiclass.md) | Softmax boosting: K trees per round and one diagonal approximation |
| 13 | [Categorical features](13-categorical-features.md) | Ordered target statistics: the encoding that doesn't leak, and why the core stays numeric |

## The template

Every chapter has the same skeleton:

- **The idea**: what problem this solves, in plain language.
- **The math**: just enough notation to make the code inevitable.
- **In bonsai**: the real implementation, with file links.
- **Try it**: CLI and Python commands, and what to look for.
- **Gotchas & war stories**: where the [decision log](../decisions.md)
  supplies genuine ones (a divergence bug, a deadlock, a factor-of-20
  normalization mistake), not hypotheticals.

## Reading order

Chapter 0 is the on-ramp; 1–4 are the core algorithm and build on each
other. 5–9 are independent; read them as the corresponding knob becomes
relevant. 10–13 assume the core and go where the engineering is.
For design *rationale* (why this data layout, why this dispatch mechanism)
see [architecture/](../architecture/); for the audit trail of every
non-trivial choice, [decisions.md](../decisions.md).
