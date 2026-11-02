---
template: home.html
hide:
  - navigation
  - toc
---

# bonsai

**Histogram gradient-boosted trees with a C++23 core.**

=== "pip"

    ```bash
    pip install bonsai-gbt
    ```

=== "uv"

    ```bash
    uv add bonsai-gbt
    ```

=== "source"

    ```bash
    git clone https://github.com/daniel-m-campos/bonsai && cd bonsai && pip install .
    ```

Wheels cover Linux (x86_64 and aarch64) and macOS (arm64), for Python 3.9 to 3.13, and install without compiling (no C++ toolchain needed). The Linux x86_64 wheel trains on GPU with only an NVIDIA driver installed. There is no Windows wheel, and the other two wheels train on CPU only. See [Install](api/install.md) for Docker and extras.

## Three doors

<div class="grid cards" markdown>

-   <span class="bz-count" data-bz-count="learn"></span>
    **[Learn](guide/README.md)**

    ---

    One concept per chapter: intuition, the mathematics, the ~50 real lines that implement it, then an experiment against the reference libraries. Start with [a tree traced by hand on eight rows](guide/0-a-tree-by-hand.md); the engine track and [how we got here](learn/timeline.md) sit behind the same door.

-   <span class="bz-count" data-bz-count="api"></span>
    **[API](api/install.md)**

    ---

    [Install in one command](api/install.md), then [the quickstart](api/quickstart.md). Estimators follow the scikit-learn fit and predict interface. The lower-level `train(params, ...)` function takes a dict of settings such as `{"tree.max_depth": 8}`, and the CLI takes the same keys through `--set tree.max_depth=8`. One `.msgpack` file written by Python loads in the CLI, and the reverse also works.

-   <span class="bz-count" data-bz-count="results"></span>
    **[Results](results/README.md)**

    ---

    Every speed and accuracy claim as a committed run on named hardware. [The results ledger](results/results.md) opens with the division summaries and links one generated page per study; the rules are [the benchmark protocol](results/benchmark-protocol.md). [The invariants](invariants.md) list the contracts the engine guarantees, most generated from the tests that enforce them. The [decisions log](decisions.md) records the measurements behind the numbers, including hypotheses the data rejected.

</div>

On GPU every bonsai grower (tree-growing strategy) fits the tall scenario (16M rows x 128 features, ingest plus train) ahead of the reference library it is paired with, in less peak host memory than any of them. At the extreme scenario (16M rows x 1024 features) XGBoost and CatBoost run out of memory, and bonsai finishes. LightGBM scores the higher test r2 on the GPU scenarios. On CPU every bonsai grower has the faster fit total at both the tall (2M rows x 128 features) and the wide (16k rows x 16384 features) scenario. All of it is measured on shared hardware at matched settings. The digits live only on [the scenario panels](results/results/perf.md), generated from the committed runs, so a refresh moves them there and nowhere else.

One property none of the reference libraries offer: models are bit-identical across CPU architectures at a fixed thread count, enforced per-commit in CI.

## The story

I started bonsai to learn gradient-boosted trees properly, from first principles, and to understand how the production libraries work. Reading the papers was not enough, so I implemented a basic serial version from scratch. The same build was my practice ground for modern C++ and the C++23 features I wanted fluency in.

Once a basic version of the trainer worked, I used Claude as a coding agent to add the features the production libraries had. Beyond pedagogy there was a slim hope this could become something useful. The first idea was to put the default growth strategies of XGBoost (depthwise), LightGBM (leaf-wise), and CatBoost (level-wise, one shared split per level) in one engine ([growing trees](guide/4-growing-trees.md)).

After that, the goals kept moving. Each target was met and replaced by a harder one: CPU parity with the reference libraries, then GPU parity, then a GPU lead. The engine is built on C++ concepts (compile-time interfaces that a type must satisfy), so a new extension, such as a new objective or growth strategy, plugs into the existing interface instead of changing the callers. Each extension was still hard work.

GPU support started when a promotional window with Anthropic's newest model made the work affordable. I learned to rent GPU nodes from RunPod and gave the agent the rental API as a tool. From then on the agent measured, changed code, and measured again on real hardware, following the [perf-round](https://github.com/daniel-m-campos/bonsai/blob/main/.claude/skills/perf-round/SKILL.md) procedure for optimization work and the [feature-admission](https://github.com/daniel-m-campos/bonsai/blob/main/.claude/skills/feature-admission/SKILL.md) procedure for deciding whether a new feature enters the core.

The ambition grew with the milestones: assimilate the defining ideas of XGBoost, LightGBM, and CatBoost into one library, match or beat their performance, and keep the code readable enough that reading it is still the point ([code metrics](results/results/code-metrics.md) counts its lines and complexity).

## Built with Claude

bonsai is built by a human maintainer working with Claude. What makes that trustworthy is verification, not trust. Models are bit-identical per commit in CI, performance claims come from committed same-pod runs, and every feature passes an admission gate with pre-registered kill criteria. Refuted hypotheses are recorded next to the adopted ones in the [decisions log](decisions.md). Commits before 2026-07-29 carry a session trailer linking the session that produced them.
