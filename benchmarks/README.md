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
