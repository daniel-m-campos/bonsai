# bonsai

**Histogram gradient-boosted trees with a C++23 core.**

```
pip install bonsai-gbt --find-links https://github.com/daniel-m-campos/bonsai/releases/expanded_assets/v1.3.0
```

Linux and macOS arm64, Python 3.9 to 3.13, no toolchain; on linux x86_64 the wheel trains on GPU out of the box. Details, docker, and extras: [Install](use/install.md).

bonsai began as a learning project: rebuild gradient-boosted trees from first principles to understand how the production libraries actually work, in a small codebase that takes modern C++23 and software design as seriously as the algorithms.

## The story

I started bonsai to learn gradient-boosted trees properly. Reading the papers was not enough, so I implemented a basic serial version from scratch. The same build was my practice ground for modern C++ and the C++23 features I wanted fluency in.

Once the first spine was designed and working, I brought in agentic workflows with Claude to add the features the production libraries had. Beyond pedagogy there was a slim hope this could become something useful. The first idea was to combine the three libraries' growth strategies in one engine: depthwise, leaf-wise, and oblivious. No reference library ships all three.

After that, things escalated. The concept-based design made each extension straightforward, though rarely easy. Milestones meant to be the finish line kept falling: CPU parity, then GPU parity, then GPU leads.

GPU support was the biggest swing, taken when a promotional window with Anthropic's newest model made the attempt affordable. I learned to rent GPU nodes from RunPod and wired the rental API into the agent's tools. From then on the agent could run tight measure-fix-measure loops on real hardware, priced by the discipline on [how we decide](method/how-we-decide.md).

The ambition grew with the milestones: assimilate the defining ideas of XGBoost, LightGBM, and CatBoost into one library, match or beat their performance, and keep the code readable enough that reading it is still the point.

Where that landed, measured on shared hardware at matched settings: on GPU, bonsai holds the fastest slot at every row scale tested, edging CatBoost and beating XGBoost at 16M rows at matched accuracy, on ~3x less host memory. Where it still loses (CatBoost on wide data, XGBoost's last 0.001 r² of cut quality), the runs are linked with the same prominence as the wins.

One property none of the reference libraries offer: models are bit-identical across CPU architectures and thread counts, enforced per-commit in CI.

## Four doors

**[Learn](guide/README.md)**: the learning project, kept as a product: each chapter takes one concept from intuition, through the mathematics, to the ~50 real lines that implement it, to an experiment against the reference libraries. Start with [a tree traced by hand on eight rows](guide/0-a-tree-by-hand.md).

**[Use](use/install.md)**: [install in one command](use/install.md), then [the whole API in one read](use/api-tour.md): sklearn-shaped estimators and an explicit `train(params, ...)` layer over the same engine, dotted config keys that are exactly the CLI's `--set` keys, one `.msgpack` model that round-trips everywhere. [Building from source](use/building.md) when a wheel is not enough.

**[Results](method/README.md)**: every speed and accuracy claim as a committed run on named hardware. The three headline numbers first, then [the benchmark protocol](method/benchmark-protocol.md), [how we decide](method/how-we-decide.md), and the full [results ledger](method/results.md).

**[Design](design/system-map.md)**: how the engine is built and the contracts it keeps: [the system map](design/system-map.md) of layers and data flow, [concepts to types](design/api-tour-concepts.md) for the surface you extend, [the HPC tension](design/the-hpc-tension.md) where the seams meet the GPU, and [determinism](design/determinism.md), bit-identical models as a testable property. The historical record sits in the archive: the [architecture notes](architecture/README.md), the [decisions log](decisions.md), and the [lineage](lineage/xgboost.md) of XGBoost, LightGBM, and CatBoost.

## Built with Claude

bonsai is built by a human maintainer working with Claude. What makes that trustworthy is verification, not trust. Models are bit-identical per commit in CI, performance claims come from committed same-pod runs, and every feature passes an admission gate with pre-registered kill criteria. Refuted hypotheses are recorded next to the adopted ones in the [decisions log](decisions.md). Every commit carries a session trailer linking the session that produced it.
