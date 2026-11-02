# Reproducing the results

Every published table comes from a harness that ships inside the package: `bonsai.bench`. You can run the harness in two ways, and they differ only in where the package comes from.

**From the installed wheel** (the normal path):

```bash
pip install "bonsai-gbt[bench]"
python -m bonsai.bench.grinsztajn out.jsonl
```

The `[bench]` extra pulls the reference libraries the suites compare against (XGBoost, LightGBM, CatBoost, pandas, openml). Plain `import bonsai.bench` needs none of them; heavy libraries load lazily per suite.

**From a source tree** (when you are developing bonsai itself):

```bash
make python
PYTHONPATH=build/python python -m bonsai.bench.grinsztajn out.jsonl
```

`make python` builds the extension into `build/python/bonsai`; `PYTHONPATH=build/python` makes that tree the package. Everything below works identically under either setup.

## The suites

| suite | division | command | notes |
| --- | --- | --- | --- |
| `grinsztajn` | quality | `python -m bonsai.bench.grinsztajn out.jsonl` then `--report` | 55 third-party tabular tasks from the Grinsztajn et al. benchmark (OpenML, a public dataset host; data downloads on first run). `--report` prints the standings table (mean rank per library). |
| `scaling` | perf | `python -m bonsai.bench run --spec scaling-rows` | Fit seconds vs rows, features, bins, or threads against the reference libraries, one bundled spec per sweep: `scaling-rows`, `scaling-cols`, `scaling-bins`, `scaling-threads`. `make bench-scaling ARGS='run --spec scaling-rows'` is the same run through the Makefile. `plan --spec scaling-rows` prints the grid without fitting. The GPU variants want a CUDA build (`make python-cuda`). |
| `datasets` | (fetcher) | `python -m bonsai.bench.datasets --list` | Lists the pinned datasets and their cache state; `python -m bonsai.bench.datasets <name>` fetches one ahead of time. |

The `grinsztajn` and `scaling` suites append one row per measurement to a JSONL file (one JSON object per line); re-running the same command skips rows already present, so an interrupted run resumes.

## Custom sweeps: the spec-driven CLI

Measurements outside a suite's built-in grid run through the unified CLI. A cell is one dataset shape and its settings (rows, features, bins). A JSON spec names the cells, the variants (libraries and growers to compare), threads, and repeats, and the driver handles child processes, resume, and row emission.

```bash
python -m bonsai.bench specs
python -m bonsai.bench plan --spec gpu-tall
python -m bonsai.bench run --spec gpu-tall --out gpu-tall.jsonl
python -m bonsai.bench variants
```

`plan` prints the expansion (every cell, variant, timeout, repeat count) without fitting anything. `run` resumes by default when the output file already exists: finished rows are skipped, failures re-attempt, so an interrupted sweep continues by re-running the same command. Specs ship inside the package (`bench/specs/`), so a bare name works from any install. `--spec` also accepts a path to your own JSON file. GPU rows record measured peak device memory in `dev_mem`, sampled while the child process runs.

Every measurement runs in its own child process, so an out-of-memory, a segfault, or a timeout is written as a jsonl row and does not stop the sweep. The driver records cells it judges too large for host RAM or VRAM as `status="skipped"` with the memory estimate that ruled them out, so you can see which sizes do not fit without reading an error; a spec can switch the gates off (`gates.mem_gate: off`) when the measured failure is the point.

## GPU and pod runs

The GPU variants need the extension built for the same Python minor version that will run it, because the nanobind module (the C++ to Python binding) fails to import under a different minor version: `make python-cuda PYTHON=$(uv python find 3.12)`. The `bench-scaling` target picks up `build-cuda/python` on its own when that build exists.

On a rented pod (a RunPod machine), copy `scripts/pod_bench_driver.sh` to the pod and run it with four variables set: `HOST_TAG` (the host name stamped on each row), `SPEC` (a bundled spec name or a path to a spec file), `OUT` (the output jsonl path on the pod), and `RUN_LABEL` (the run name recorded on each row). The script clones the repository, builds, and runs the spec, and running it again resumes. Append the pod's rows to the ledger with `ssh pod cat <OUT path> >> benchmarks/results/<file>.jsonl` (jsonl files can be concatenated), then commit and delete the pod. The [RunPod runbook](https://github.com/daniel-m-campos/bonsai/blob/main/docs/ops/runpod-runbook.md) covers the image, pod acceptance, and the rest of the loop.

## Reading what comes out

Compare timings only between runs on the same machine in the same session: identical-model GPUs across rental fleets measure up to ~25% apart, so a number from one host says nothing about another. [The benchmark protocol](benchmark-protocol.md) is the full set of rules the published numbers follow.

Suites append one JSON row per measurement to the output file. Each row records the command, the settings it ran with (the `knobs` field, plus `knobs_hash`, a short hash you can group rows by), the git sha, and the host: GPU, driver, CPU model, RAM, operating system, and library versions (`bonsai.bench.runlog`). The published tables in [the results ledger](results.md) are rendered from committed rows of exactly this shape.

The building blocks are importable directly when you want a custom harness:

```python {.run}
from bonsai.bench import metrics, synth

X, y, X_test, y_test = synth.gen_data(rows=1000, cols=8, seed=0, n_test=200, informative=4)
print("train shape:", X.shape, "| r2 of predicting zero:", round(metrics.r2(y_test, y_test * 0), 3))
```
