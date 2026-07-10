# Benchmarks

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

### Microbenchmarks

`bench_split.cpp` here is a google-benchmark-style microbench of the histogram splitter, separate from the end-to-end Python harness.

## GPU node workflow (phase-3 perf loop)

The GPU perf work ([architecture/11-gpu-resident.md](../docs/architecture/11-gpu-resident.md)) iterates against the MSD ladder on a rented GPU node. A limited-context session needs only this section and that design doc.

**Node**: a Thunder Compute **4× A100 standard VM** ($5.96/hr; smallest standard-VM size). Do NOT use 1×/2× instances — those virtualize the GPU over the network (test suites wedge, kernel-launch latency is distorted). Billing stops **only when the instance is deleted**; there is no stop/pause, so delete the node at the end of every session.

**Setup** (one command on a fresh node; idempotent):

```
curl -fsSL https://raw.githubusercontent.com/daniel-m-campos/bonsai/cuda-phase2/scripts/setup_gpu_node.sh | bash -s -- <branch>
```

This disables Thunder's `/etc/ld.so.preload` telemetry shim (it busy-spins under concurrent CUDA-linked process startup), installs LLVM 21 + libc++, cmake ≥ 4 via uv, and CUDA 12.6 side-by-side (clang cannot target the preinstalled CUDA 13), then clones, builds, and fetches the datasets.

**The loop**:

```
# locally: edit -> push          # on the node:
git pull && ninja -C build-cuda
make bench-gpu                   # or: uv run scripts/bench_gpu.py --threads 16
ctest --test-dir build-cuda -j16 # before claiming a win
```

JSONL sync discipline: the node appends to the tracked results file, so always `scp` the node's copy back to the workstation and commit it there, then `git checkout -- benchmarks/results/gpu_msd.jsonl` on the node before `git pull` (the repo copy already contains the node's entries).

`bench-gpu` runs bonsai cuda_depthwise / bonsai CPU / xgboost CPU / xgboost GPU with the hyperparameters from `configs/year_prediction_msd.toml`, prints fit time, RMSE, the gap to xgboost-GPU, and the `BONSAI_CUDA_PROFILE` + `BONSAI_GROW_PROFILE` breakdowns, and appends one JSON line per variant to `benchmarks/results/gpu_msd.jsonl` (keyed by git sha + GPU name) — regression tracking is a diff of that file. Pin threads explicitly on many-core hosts (`n_threads=0` collapses: issue #2). A100 reference points (2026-07-04, commit 1e20154): bonsai GPU 12.2 s, xgboost GPU 1.8 s, RMSE 8.9911.

## Scaling suite (synthetic rows / cols / bins / threads)

`make bench-scaling ARGS="--axis all"` sweeps synthetic regression data — generalized Friedman-1 target over uniform float32 features, ~20 informative columns, noise sized for a best-achievable R² of ~0.9 — through bonsai (CPU + cuda growers via the Python module), xgboost, lightgbm, and catboost, appending one JSON line per (cell, variant, repeat) to `benchmarks/results/scaling.jsonl`. Methodology and grid rationale: decision 46.

Grid: base cell 1M×100×255 bins (depth 8, 100 iters, 16 threads, 3 repeats for a variance estimate), swept per axis — rows to 16M, cols to 65k (rows shrink past 4k cols to cap cells at 2^31), bins to 65535 (bonsai's uint16 cap; other libraries swept to their own caps), threads {1,4,16,64} at the base cell. `--tier auto` (default) reads host RAM and `nvidia-smi` VRAM and records infeasible corners as `status="skipped"` with the estimate — the feasibility frontier is data, not an error. Every measurement runs in a child process, so OOM/segfault/timeout become jsonl lines and bonsai's `BONSAI_{INGEST,GROW,CUDA}_PROFILE` breakdowns are captured from the child's stderr.

Fairness: fit is timed from in-memory numpy arrays and includes each library's own ingestion (bonsai ColumnBatch + quantile binning, xgboost QuantileDMatrix, lgb.Dataset, catboost Pool); predict is timed from a raw test matrix; quality is R² train/test. `lgbm_cuda` is recorded `unsupported` in v1 (the pip wheel has no CUDA backend). CatBoost-GPU's 254-bin cap is applied and recorded as `bins_effective`.

Interpreter pinning: the nanobind module is ABI-tied to one Python — build it with `make python-cuda PYTHON=$(uv python find 3.12)` (or `make python` on CPU hosts) so the module matches the uv script environment; the `bench-scaling` target auto-selects `build-cuda/python` when present.

Pod loop: toolchain + clone + build as in the GPU perf loop above, then `make python-cuda`, `make bench-scaling ARGS="--axis all --host-name <gpu-tag>"`, smoke-check the printed lines, and on the workstation `ssh pod cat .../benchmarks/results/scaling.jsonl >> benchmarks/results/scaling.jsonl` (jsonl appends compose), commit, delete the pod. Analyze with `uv run scripts/analyze_scaling.py` → `benchmarks/results/scaling.md` + log-log plots under `benchmarks/results/scaling/`; exponents are log-log slopes of fit_s per axis, with a joint 2D rows+cols fit de-confounding the shrinking-rows tail of the cols axis.

### Pod image and startup (the RunPod decoder ring)

Rent pods with `ghcr.io/daniel-m-campos/bonsai-ci:cuda12.4` (built from `docker/ci.Dockerfile` by the `ci-image` workflow): CUDA 12.4 toolkit, clang-21 + libc++, modern cmake/ninja, python 3.12 venv at `/opt/venv` with numpy/nanobind/lightgbm, and the FetchContent deps pre-cloned under `/opt/deps` — pod setup is clone + configure + build, nothing else. The `PUBLIC_KEY` env installs your key for direct SSH.

Hard-won startup rules, image-independent: a pod's REST `desiredStatus: RUNNING` says nothing about the container — the GraphQL `pod { runtime { uptimeInSeconds } }` field is the API-side liveness signal (0 = never started; only the console shows *why*). CUDA-12.8-family images silently never start on 12.4-driver machines. A create call that returns an error can still create a billing pod — list and sweep after failures. Some machines are simply broken (missing `/dev/dri` nodes crash-loop `runc`); delete and re-roll rather than wait.

### Pod acceptance and the defective-host lesson

One rentable 5090 machine measured a ~300µs GPU sync round-trip (healthy: 4µs) with perfect PCIe and bandwidth — enough to add 11–14s to every cuda fit and to masquerade as a bonsai regression until diagnosed (decision 48). Before benchmarking on any rented pod, run the 30-second sync-latency probe (compile `cudaMemsetAsync` + `cudaDeviceSynchronize` × 2000; reject the pod above 50µs/op) and a base-cell CPU sanity fit; the round-1 fleet rows for host `pod-NVIDIA-GeForce-RTX-5090` in `scaling.jsonl` carry this defect and must not be used for cuda timing comparisons — the `g1pre-`/`g1new-` rows are the healthy replacements.
