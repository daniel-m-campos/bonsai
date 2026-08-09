# Running the benchmarks

Every published table comes from a harness that ships inside the package: `bonsai.bench`. There are two ways to stand it up, and they differ only in where the package comes from.

**From the installed wheel** (the normal path):

```bash
pip install "bonsai-gbt[bench]"
python -m bonsai.bench.grinsztajn out.jsonl
```

The `[bench]` extra pulls the reference libraries the suites compare against (XGBoost, LightGBM, CatBoost, pandas, openml). Plain `import bonsai.bench` needs none of them; heavy libraries load lazily per suite.

**From a source tree** (when you are hacking on bonsai itself):

```bash
make python
PYTHONPATH=build/python python -m bonsai.bench.grinsztajn out.jsonl
```

`make python` builds the extension into `build/python/bonsai`; `PYTHONPATH=build/python` makes that tree the package. Everything below works identically under either setup.

## The suites

| suite | division | command | notes |
| --- | --- | --- | --- |
| `grinsztajn` | quality | `python -m bonsai.bench.grinsztajn out.jsonl` then `--report` | The external standings suite (55 third-party tasks). Datasets fetch from OpenML on first run; rows already in `out.jsonl` are skipped, so an interrupted run resumes by re-running the same command. `--report` renders the standings from the jsonl. |
| `scaling` | perf | `python -m bonsai.bench.scaling --smoke` | Fit seconds vs rows/cols/bins/threads against the reference libraries. `--smoke` is the laptop mode (small cells, minutes); the full grid (`--axis all`) takes hours and wants a CUDA build (`make python-cuda`) for the GPU variants. `--dry-run` prints the grid without running it. |
| `datasets` | (fetcher) | `python -m bonsai.bench.datasets --list` | Lists the pinned datasets and their cache state; `python -m bonsai.bench.datasets <name>` fetches one ahead of time. |

## Custom ladders: the spec-driven CLI

Cells that are not on a suite's built-in grid run through the unified CLI: a JSON spec names the cells (or a generator), the variants, threads, and repeats, and the driver handles child processes, resume, and row emission.

```bash
python -m bonsai.bench specs
python -m bonsai.bench plan --spec gpu-tall
python -m bonsai.bench run --spec gpu-tall --out gpu-tall.jsonl
python -m bonsai.bench variants
```

`plan` prints the expansion (every cell, variant, timeout, repeat count) without fitting anything. `run` resumes by default when the output file already exists: finished rows are skipped, failures re-attempt, so an interrupted sweep continues by re-running the same command. Campaign specs ship inside the package (`bench/specs/`), so a bare name works from any install and `--spec` also takes a path to your own JSON; the `iso_volume` generator holds rows x cols constant while sweeping the aspect ratio, and GPU rows record measured peak device memory (`dev_mem`) sampled while the child runs. Pod campaigns wrap this in `scripts/pod_bench_driver.sh` (see `benchmarks/README.md`).

## Reading what comes out

Suites append one JSON row per measurement to the output file, self-describing enough to reproduce: the command, the knob set (hashed for grouping), the git sha, and the host down to library versions (`bonsai.bench.runlog`). The published tables in [the results ledger](../method/results.md) are rendered from committed rows of exactly this shape.

One honest caveat before comparing numbers across machines: identical-model GPUs across rental fleets measure up to ~25% apart, so only same-host comparisons mean anything; [the benchmark protocol](../method/benchmark-protocol.md) is the full set of rules the published numbers follow.

The building blocks are importable directly when you want a custom harness:

```{.python .run}
from bonsai.bench import metrics, synth

X, y, X_test, y_test = synth.gen_data(rows=1000, cols=8, seed=0, n_test=200, informative=4)
print("train shape:", X.shape, "| r2 of predicting zero:", round(metrics.r2(y_test, y_test * 0), 3))
```
