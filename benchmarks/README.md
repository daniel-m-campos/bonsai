# Benchmarks

> The normative protocol (divisions, suites, primary metrics, timing modes, the result-row schema) is [docs/method/benchmark-protocol.md](../docs/method/benchmark-protocol.md); this file is the operational how-to-run companion.

The packaged suites run as modules, from an installed wheel (`pip install "bonsai-gbt[bench]"`) or a source tree (`PYTHONPATH=build/python` after `make python`); the reader-facing walkthrough is [Running the benchmarks](../docs/use/benchmarks.md):

```bash
python -m bonsai.bench.grinsztajn out.jsonl --report   # quality standings
python -m bonsai.bench.scaling --smoke                 # perf grid, laptop mode
python -m bonsai.bench.datasets --list                 # dataset cache state
```

## Reference-library comparison

bonsai vs xgboost, lightgbm, catboost on a CSV regression dataset. All four libraries see the same hyperparameters by reading them out of the bonsai TOML config.

The Python sidecar is two single-file scripts with [PEP 723][pep723] inline metadata. They run via [uv][uv] with no manual virtualenv setup. **Python 3.12 is required** (pinned in each script's header; uv enforces it).

[pep723]: https://peps.python.org/pep-0723/ [uv]: https://docs.astral.sh/uv/

### Setup

Install uv once:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Run

From the repo root:

```
# 1. Fetch the toy dataset (writes tests/data/california_housing_{train,test}.csv).
uv run scripts/fetch_toy.py

# 2. Build bonsai.
make build

# 3. Run all four libraries and write the report.
uv run scripts/compare.py --config configs/california_housing.toml
```

Results land in `benchmarks/results/<config-stem>.{json,md}`. The Python script also prints the markdown table to stdout when it finishes.

The first `uv run` of each script downloads its declared dependencies into uv's shared cache. Subsequent runs reuse the cache and are fast.

## Scaling suite (synthetic rows / cols / bins / threads)

`make bench-scaling ARGS="--axis all"` sweeps synthetic regression data — generalized Friedman-1 target over uniform float32 features, ~20 informative columns, noise sized for a best-achievable R² of ~0.9 — through bonsai (CPU + cuda growers via the Python module), xgboost, lightgbm, and catboost, appending one JSON line per (cell, variant, repeat) to the `--out` file. Methodology and grid rationale: decision 46; committed results follow the standings policy in [the benchmark protocol](../docs/method/benchmark-protocol.md).

Grid: base cell 1M×100×255 bins (depth 8, 100 iters, 16 threads, 3 repeats for a variance estimate), swept per axis — rows to 16M, cols to 65k (rows shrink past 4k cols to cap cells at 2^31), bins to 65535 (bonsai's uint16 cap; other libraries swept to their own caps), threads {1,4,16,64} at the base cell. The driver reads host RAM and `nvidia-smi` VRAM and records infeasible corners as `status="skipped"` with the estimate — the feasibility frontier is data, not an error — and a spec can disable the gates (`gates.mem_gate: off`) when a measured failure is the point. Every measurement runs in a child process, so OOM/segfault/timeout become jsonl lines and bonsai's `BONSAI_{INGEST,GROW,CUDA}_PROFILE` breakdowns are captured from the child's stderr.

Fairness: fit is timed from in-memory numpy arrays and includes each library's own ingestion (bonsai ColumnBatch + quantile binning, xgboost QuantileDMatrix, lgb.Dataset, catboost Pool); predict is timed from a raw test matrix; quality is R² train/test. `lgbm_cuda` runs LightGBM's CUDA tree learner via the source build baked into the bonsai-ci image (the PyPI wheel is CPU-only), so it needs a pod or a host with that build. CatBoost-GPU's 254-bin cap is applied and recorded as `bins_effective`.

Interpreter pinning: the nanobind module is ABI-tied to one Python — build it with `make python-cuda PYTHON=$(uv python find 3.12)` (or `make python` on CPU hosts) so the module matches the uv script environment; the `bench-scaling` target auto-selects `build-cuda/python` when present.

Custom ladders and campaigns run through the unified CLI instead of hand-written driver scripts: `python -m bonsai.bench run --spec <name-or-path>`, with campaign specs shipped in the package (`python -m bonsai.bench specs` lists them) and `python -m bonsai.bench plan --spec ...` printing the expanded job list without running anything. Spec mode resumes by default when the output file exists, so an interrupted sweep re-invokes with the same command.

Pod loop: for campaigns, `scp` the committed driver `scripts/pod_bench_driver.sh` to the pod and launch it with `HOST_TAG`/`SPEC`/`OUT`/`RUN_LABEL` set (see [docs/ops/runpod-runbook.md](../docs/ops/runpod-runbook.md), campaign driver section); it clones, builds `python-cuda`, and runs the spec, and re-invoking resumes. For the legacy scaling grid, toolchain + clone + build as in the GPU perf loop above, then `make python-cuda`, `make bench-scaling ARGS="--axis all --host-name <gpu-tag>"`. Either way, on the workstation `ssh pod cat <out>.jsonl >> benchmarks/results/<file>.jsonl` (jsonl appends compose), commit, delete the pod.

### Pod image and startup (the RunPod decoder ring)

Rent pods with `ghcr.io/daniel-m-campos/bonsai-ci:cuda12.8` (built from `docker/ci.Dockerfile` by the `ci-image` workflow): CUDA 12.8 toolkit, clang-21 + libc++, modern cmake/ninja, python 3.12 venv at `/opt/venv` with numpy/nanobind/lightgbm, and the FetchContent deps pre-cloned under `/opt/deps` — pod setup is clone + configure + build, nothing else. The `PUBLIC_KEY` env installs your key for direct SSH. cuda12.8 covers sm_120 (Blackwell) natively, no toolkit side-install, and L40S r550-driver pods still run its 12.8-built binaries via CUDA minor-version compatibility. Branches pinned to the retired `cuda12.4` tag keep pulling it unchanged.

Hard-won startup rules, image-independent: a pod's REST `desiredStatus: RUNNING` says nothing about the container — the GraphQL `pod { runtime { uptimeInSeconds } }` field is the API-side liveness signal (0 = never started; only the console shows *why*). Some CUDA-12.8-family images silently never start on 12.4-driver machines, though the bonsai-ci image itself has not hit this. A create call that returns an error can still create a billing pod — list and sweep after failures. Some machines are simply broken (missing `/dev/dri` nodes crash-loop `runc`); delete and re-roll rather than wait.

### Pod acceptance and the defective-host lesson

One rentable 5090 machine measured a ~300µs GPU sync round-trip (healthy: 4µs) with perfect PCIe and bandwidth — enough to add 11–14s to every cuda fit and to masquerade as a bonsai regression until diagnosed (decision 48). Before benchmarking on any rented pod, run the 30-second sync-latency probe (compile `cudaMemsetAsync` + `cudaDeviceSynchronize` × 2000; reject the pod above 50µs/op) and a base-cell CPU sanity fit; the round-1 fleet rows carrying this defect (host `pod-NVIDIA-GeForce-RTX-5090`, the retired scaling history) are archived in git history.
