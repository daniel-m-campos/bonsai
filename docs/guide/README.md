# The bonsai guide: gradient boosting from math to code

This guide teaches gradient boosting (building a model from many small decision trees, each correcting the last) by linking each concept to the lines in bonsai that implement it. Read it if you know the parameters and want the mechanism, or know the math and want to see the code. The guide is the reason bonsai exists. Each chapter takes one concept from intuition to the math, then to the **actual lines in this repository that implement it**, and has a section you can run in bonsai.

The code the guide references is the shipping code, not a simplification.

## Chapters

| # | Chapter | One-line pitch |
|---|---------|----------------|
| 0 | [A tree by hand](0-a-tree-by-hand.md) | One boosting round on eight rows, every number traced |
| 1 | [Gradient boosting](1-gradient-boosting.md) | Why trees fit *gradients*, why second order, where leaf values come from |
| 2 | [Binning & histograms](2-binning-and-histograms.md) | How 255 bins replace sorting at every split, and how subtracting the smaller child's histogram from the parent's gives the larger child's |
| 3 | [Finding splits](3-finding-splits.md) | The gain formula, one prefix scan, and where missing values go |
| 4 | [Growing trees](4-growing-trees.md) | Depth-wise vs best-first vs level-wise: three answers to "which leaf next?" |
| 5 | [Sampling](5-sampling.md) | Training on fewer rows: Bernoulli, GOSS (Gradient-based One-Side Sampling: keep the largest-gradient rows, sample the rest), and a bug worth learning from |
| 6 | [Regularization & constraints](6-regularization-and-constraints.md) | L1/L2, feature sampling, monotone and interaction constraints |
| 7 | [Early stopping & DART](7-early-stopping-and-dart.md) | Knowing when to stop, and dropout for trees |
| 8 | [Feature importance](8-feature-importance.md) | Split vs gain, why they disagree, and what to distrust |
| 9 | [Parallelism & determinism](9-parallelism-and-determinism.md) | Deterministic models at a fixed thread count, and what that costs |
| 10 | [GPU training](10-gpu-training.md) | Which work runs on the CPU (the host) and which on the GPU (the device), and the integer arithmetic that makes GPU fits repeatable and close to CPU fits |
| 11 | [Performance engineering](11-performance-engineering.md) | The compute-DAG method: model training as a graph of steps and estimate the cost of each change before building it |
| 12 | [Multiclass](12-multiclass.md) | Softmax boosting: K trees per round and one diagonal approximation |
| 13 | [Categorical features](13-categorical-features.md) | Ordered target statistics: the encoding that doesn't leak, and why the core stays numeric |
| 14 | [Feature selection](14-feature-selection.md) | Ten selection methods raced on real data: who wins at which budget, and what it costs |
| 15 | [Explaining predictions](15-explaining-predictions.md) | Explain one prediction with Shapley values (each feature's share of the prediction, averaged over every order of adding features), then TreeSHAP and a global importance, all checked by hand on a two-feature, 100-day example |
| 16 | [Multi-target regression](16-multi-target.md) | One tree per target per round, bit-identical to K separate fits, and the K-wide leaf measured and declined |

## The shape of a chapter

Most chapters share a skeleton:

- **The idea**: what problem this solves, in plain language.
- **The math**: just enough notation to make the code inevitable.
- **In bonsai**: the real implementation, with file links.
- **Try it**: CLI and Python commands, and what to look for.
- **Gotchas & war stories**: where the [decision log](../decisions.md) supplies genuine ones (a divergence bug, a deadlock, a factor-of-20 normalization mistake), not hypotheticals.

Chapters 0, 7, 14, 15 and 16 run their own section order: 0 is a worked example in steps, 7 covers two methods in one chapter, 14 is built around a measured survey, 15 around one worked example, and 16 replaces In bonsai and Try it with In the code and Quick use. Every chapter but 14 links the implementation, and every chapter has code you can run.

## Reading order

Chapter 0 is the on-ramp. Chapters 1 to 4 are the core algorithm and build on each other. Chapters 5 to 9 are independent; read each as the matching parameter becomes relevant. Chapters 10 and 11 go where the engineering is: GPU training, then performance. Chapters 12 to 16 put the core to work: multiclass, categorical features, feature selection, explaining predictions, then multi-target regression. For why bonsai uses a given data layout or dispatch mechanism, read [decisions.md](../decisions.md), which holds the rationale, measurements, and rejected alternatives, cited by number. For the contracts that must hold across files, read [the invariants](../invariants.md).
