---
name: pod-validate
description: >
  Validate GPU/CUDA changes on a rented RunPod L40S: create pod, build,
  run the [cuda] suite and a profiled 16M benchmark, tear down. Use for
  any change to src/cuda/ or src/level_step.hpp's device plane, or when
  the user asks to benchmark against xgboost/lightgbm/catboost on GPU.
---

The human-readable version of this workflow with every failure mode explained is `docs/ops/runpod-runbook.md` — read it if anything below misbehaves. This file is the agent fast path.

## Invariants (violating any of these wastes money or corrupts results)

1. `PUBLIC_KEY` env at create time is mandatory; it cannot be added later.
2. Only **same-pod** before/after comparisons are valid (~25% fleet spread between identical GPUs). Never quote cross-pod absolutes.
3. Export the PATH below in every SSH session — sshd does not inherit Docker ENV.
4. After ANY create failure: list pods and delete strays (failed creates can still bill).
5. Delete the pod the moment results are copied off. Verify list-pods is empty before ending.

## The loop

Create (MCP `create-pod` or REST per the runbook): image `ghcr.io/daniel-m-campos/bonsai-ci:cuda12.4`, `NVIDIA L40S`, SECURE, 80GB disk, ports `["22/tcp"]`, env `{"PUBLIC_KEY": "<contents of the user's ~/.ssh/id_ed25519.pub>"}`.

Poll GraphQL until `runtime` is non-null; read the public ip/port for privatePort 22 from the same response.

On the pod (single-branch shallow clones need FETCH_HEAD; `pkill -f <script>` inside an ssh command kills the session — bracket the pattern):

```bash
export PATH=/opt/venv/bin:/root/.local/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd /root && git clone --depth 1 https://github.com/daniel-m-campos/bonsai.git && cd bonsai
git fetch --depth 1 origin <branch> && git checkout -f FETCH_HEAD
make python-cuda PYTHON=/opt/venv/bin/python
cmake --build build-cuda --target bonsai_tests -j"$(nproc)"
./build-cuda/tests/bonsai_tests "[cuda]"
```

The profiled ledger cell (before AND after, same pod, when perf is claimed):

```bash
spec='{"variant":"bonsai_cuda_depthwise","cell":{"axis":"rows","rows":16000000,"cols":100,"bins":255,"bins_effective":255,"depth":8,"iters":100,"lr":0.1,"informative":20,"n_test":500000,"seed":42},"threads":16}'
PYTHONPATH=$PWD/build-cuda/python BONSAI_GROW_PROFILE=1 BONSAI_CUDA_PROFILE=1 \
  BONSAI_INGEST_PROFILE=1 BONSAI_FIT_PROFILE=1 \
  /opt/venv/bin/python scripts/bench_scaling.py --worker <<<"$spec" >/tmp/r.out 2>/tmp/r.err
grep -o "RESULT .*" /tmp/r.out
grep -E "grow-profile|fit-profile|ingest-profile|cuda-upload-decomp" /tmp/r.err
```

Reference variants for same-pod ladders: `xgb_cuda`, `lgbm_cpu`, `catboost_gpu` (same spec, different `variant`).

## Reading the result

- r² must EQUAL the before-run exactly for behavior-preserving changes (host gradients unchanged ⇒ identical models); GPU-vs-GPU value comparisons in tests use 1e-4 (atomic order).
- Check conservation: fit-profile's lines should explain the wall clock; grow should equal the grow-profile bucket sum. An unexplained gap is the next target, not noise (doc 16).
- A silent `dbin=0.00` / unexpected `bins_upload>0` / `cpu_fallback>0` means a device path silently declined — investigate before trusting any number.
