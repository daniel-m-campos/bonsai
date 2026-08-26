# bonsai

**Histogram gradient-boosted trees with a C++23 core.**

```
pip install bonsai-gbt
```

Linux and macOS arm64, Python 3.9 to 3.13, no toolchain; on linux x86_64 the wheel trains on GPU out of the box. Details, docker, and extras: [Install](use/install.md).

bonsai began as a learning project: rebuild gradient-boosted trees from first principles to understand how the production libraries actually work, in a small codebase that takes modern C++23 and software design as seriously as the algorithms.

## The story

I started bonsai to learn gradient-boosted trees properly. Reading the papers was not enough, so I implemented a basic serial version from scratch. The same build was my practice ground for modern C++ and the C++23 features I wanted fluency in.

Once the first spine was designed and working, I brought in agentic workflows with Claude to add the features the production libraries had. Beyond pedagogy there was a slim hope this could become something useful. The first idea was to combine the three libraries' growth strategies in one engine: depthwise, leaf-wise, and level-wise. No reference library ships all three.

After that, things escalated. The concept-based design made each extension straightforward, though rarely easy. Milestones meant to be the finish line kept falling: CPU parity, then GPU parity, then GPU leads.

GPU support was the biggest swing, taken when a promotional window with Anthropic's newest model made the attempt affordable. I learned to rent GPU nodes from RunPod and wired the rental API into the agent's tools. From then on the agent could run tight measure-fix-measure loops on real hardware, priced by the discipline on [how we decide](method/how-we-decide.md).

The ambition grew with the milestones: assimilate the defining ideas of XGBoost, LightGBM, and CatBoost into one library, match or beat their performance, and keep the code readable enough that reading it is still the point.

Here is where that landed, measured on shared hardware at matched settings. On GPU at the tall scenario (16M x 128), fit totals run 5.6s against XGBoost's 21.7s, 9.1s against LightGBM's 28.0s, and 5.9s against CatBoost's 16.9s. bonsai holds 9.1GB of peak host memory there, against LightGBM's 15.2GB and XGBoost's 29.5GB. At the extreme scenario (16M x 1024), XGBoost and CatBoost run out of memory and bonsai finishes. LightGBM wins test r2 in every GPU scenario: 0.885, 0.868 and 0.886 against 0.879, 0.860 and 0.879. On CPU the leafwise grower loses to LightGBM, 39.8s against 19.9s at the tall scenario and 518.3s against 69.1s at the wide one. Both sides sit on the same page, [the scenario panels](method/results/perf.md).

One property none of the reference libraries offer: models are bit-identical across CPU architectures and thread counts, enforced per-commit in CI.

## Four doors

<div class="grid cards" markdown>

-   **[Learn](guide/README.md)**

    ---

    One concept per chapter: intuition, the mathematics, the ~50 real lines that implement it, then an experiment against the reference libraries. Start with [a tree traced by hand on eight rows](guide/0-a-tree-by-hand.md).

-   **[Use](use/install.md)**

    ---

    [Install in one command](use/install.md), then [the API in one read](use/api-tour.md): sklearn-shaped estimators and an explicit `train(params, ...)` layer, dotted config keys shared with the CLI, one `.msgpack` model that round-trips everywhere.

-   **[Results](method/README.md)**

    ---

    Every speed and accuracy claim as a committed run on named hardware. [The results ledger](method/results.md) opens with the division summaries and links one generated page per study; the rules are [the benchmark protocol](method/benchmark-protocol.md).

-   **[Design](design/system-map.md)**

    ---

    [The system map](design/system-map.md), [concepts to types](design/api-tour-concepts.md), [the HPC tension](design/the-hpc-tension.md), and [determinism as a contract](design/determinism.md). The archive holds the [decisions log](decisions.md); [how we got here](learn/timeline.md) traces the lineage of the ideas.

</div>

## Built with Claude

bonsai is built by a human maintainer working with Claude. What makes that trustworthy is verification, not trust. Models are bit-identical per commit in CI, performance claims come from committed same-pod runs, and every feature passes an admission gate with pre-registered kill criteria. Refuted hypotheses are recorded next to the adopted ones in the [decisions log](decisions.md). Commits before 2026-07-29 carry a session trailer linking the session that produced them.
