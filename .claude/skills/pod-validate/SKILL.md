---
name: pod-validate
description: >
  Validate GPU/CUDA changes or run measurement sessions on a rented RunPod
  GPU: create pod, build, run, babysit, tear down. Use for any change to
  src/cuda/ or the device planes, for hand-driven standings refreshes, and
  when benchmarking against xgboost/lightgbm/catboost on GPU.
---

The human-readable version with every failure mode explained is `docs/ops/runpod-runbook.md`; this file is the agent fast path. Release wheels do NOT need this skill: wheels.yml's `validate-cuda` job self-validates every release (decision 70).

## Invariants (violating any of these wastes money or corrupts results)

1. `PUBLIC_KEY` env at create time is mandatory; it cannot be added later.
2. Only **same-pod** interleaved comparisons are valid (~25% fleet spread between identical GPUs). Never quote cross-pod absolutes; interleave reps ACROSS arms so pod drift cannot masquerade as an effect.
3. Export the PATH below in every ssh session; sshd does not inherit Docker ENV.
4. After ANY create failure: list pods and delete strays (failed creates can still bill).
5. Delete the pod the moment results are copied off; verify list-pods is empty before ending.
6. **A GPU pod cannot time the CPU plane.** Which mechanism caps the container decides what a fit needs, because it decides whether bonsai's spinning barriers cost anything. A GPU pod meters CPU *bandwidth* (13.6 cores against the 128 `nproc` advertises), so spin-wait drains the shared allowance and the host needs `threads * 1.5` cores. A rented **CPU** pod enforces the purchase as a *cpuset*: `cpu.max` reads `max`, each thread owns a whole core, spin costs only that core, so one vCPU per thread is enough. CPU-plane runs therefore go on a CPU pod at `threads` vCPU; assert before measuring that either no quota exists and the cpuset covers the threads, or a quota exists and clears `threads * 1.5`. Run with `OMP_WAIT_POLICY=passive` either way, and confirm `nr_throttled` barely moves around the fit (runbook section 11, issues #355, #360).
7. **A pod never waits on a human gate without a deadline.** If a run finishes and the next step needs a merge or a decision, tear down and re-rent later; every watcher carries a wall-clock cap sized about 2x the expected run.

## Provisioning

- Standings card: `NVIDIA RTX PRO 6000 Blackwell Server Edition` (96GB, SECURE ~$2.09/h; the Workstation Edition is the same silicon when Server has no capacity). Dev and validation workhorse: `NVIDIA L40S` (~$0.99/h).
- Image `ghcr.io/daniel-m-campos/bonsai-ci:cuda12.8`: covers sm_120/Blackwell natively; r550 L40S pods run its 12.8-built binaries via CUDA minor-version compatibility. The frozen `cuda12.4` tag serves historical branches only.
- Create errors decoded: "machine does not have the resources" means shrink `containerDiskInGb` (40 suffices, data lives in RAM) and retry; "no instances available" means retry on a ~10-minute timer, capacity churns.
- Prefer data centers with a warm image cache (cold pulls take minutes, warm seconds); pass `dataCenterIds` when a prior session proved one. EUR-IS-2 failed to boot this image twice.
- **Host RAM is the cgroup limit, not what `free` shows**: read `/sys/fs/cgroup/memory.max` (v1 fallback `/sys/fs/cgroup/memory/memory.limit_in_bytes`, whose near-2^63 sentinel means unlimited). `free` and `/proc/meminfo` report the machine even inside a 4GB container, so a memory guard reading them cannot fail (BLACKWELL_96 pods cap at 188GB against 1.5TB physical). Size host-side data generation to ~60% of the cap; `killed by signal 9` on the worker is the symptom of ignoring this.
- Reference-arm sanity before measuring: XGBoost is pinned `>=3.2,<3.4` (a CUDA-13 wheel on a CUDA-12 driver silently falls back to CPU, issue #333); `uv pip install` the pin if the image predates it, pre-flight a tiny `device="cuda"` fit, and check `save_config()` reports cuda. The #334 runner guard is the in-band backstop.

## The loop

Create (MCP `create-pod` or REST per the runbook), poll `get-pod` until `runtime.ports` lists the public ip/port for privatePort 22, then:

```bash
export PATH=/opt/venv/bin:/root/.local/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd /root && git clone https://github.com/daniel-m-campos/bonsai.git && cd bonsai
git fetch origin <sha-or-branch> && git checkout -f FETCH_HEAD
make python-cuda PYTHON=/opt/venv/bin/python
cmake --build build-cuda --target bonsai_tests -j"$(nproc)"
./build-cuda/tests/bonsai_tests "[cuda]"
```

Launch anything long DETACHED, never through a held ssh stream: `setsid nohup bash /root/<script>.sh > /root/run.log 2>&1 < /dev/null & sleep 2; pgrep -c -f <script>` and confirm the count before disconnecting. The script's last act prints `<NAME>_DONE`, and every guarded failure prints `<NAME>_FAIL <why>`; a watcher that greps only for DONE misses tracebacks (`set -e` deaths are silent). `pkill -f <pattern>` inside an ssh command kills the session itself; bracket the pattern.

## Babysitting (mandatory for anything over ~10 minutes)

A local background watcher polls every ~4 minutes and prints one heartbeat: `procs=<pgrep count> gpu=<util>% load=<loadavg> :: <log tail>`. It exits nonzero (alerting the operator) on: the script dead without its DONE marker; log frozen AND gpu idle AND cpu idle for three consecutive polls; or the wall-clock cap. Expected-quiet phases (host-side data generation runs load high with gpu at 0%) do not trip the double-idle rule. Pull result files incrementally on every poll so a late death loses nothing already measured.

## Measurement discipline

- Profile counters emit one block PER FIT, and the worker's untimed micro-fit emits a near-zeros block first: always parse the LAST block.
- Any experimental toggle must print a provable engagement line under `BONSAI_CUDA_PROFILE` (the `BONSAI_HIST_TILE=8 active: ...` pattern), and the session asserts its presence per run. An equality test between arms cannot detect an inert toggle: both sides silently run the same code and pass.
- Device-memory claims use per-process NVML numbers (`dev_mem.peak_gb_pid` with `source: "nvml"`); device-wide totals are contaminated by co-tenants and are never per-arm evidence.
- r² must EQUAL the before-run exactly for a behavior-preserving change on either plane: the device fit is bit-reproducible on one device and build (decision 124), so a GPU-vs-GPU rep that differs in any digit is a real difference, not atomic-order flutter. Host-vs-device comparisons are tolerance-equal at 1e-4. `scripts/model_hash.py --grower cuda_depthwise` three times is the device proof; it must print one hash.
- Check conservation: the profile buckets must explain the wall clock; grow should equal the grow-profile bucket sum. An unexplained gap is the next target, not noise (doc 16).
- A silent `dbin=0.00`, unexpected `bins_upload>0`, or `cpu_fallback>0` means a device path silently declined; a reference arm that errors is a loud row, never a quietly slow number. If an arm dies mid-session under `set -e`, patch and resume rather than trusting partial output.

## The profiled ledger cell (before AND after, same pod, when perf is claimed)

```bash
spec='{"variant":"bonsai_cuda_depthwise","cell":{"axis":"rows","rows":16777216,"cols":128,"bins":255,"bins_effective":255,"depth":8,"iters":100,"lr":0.1,"informative":20,"n_test":500000,"seed":42},"threads":16}'
PYTHONPATH=$PWD/build-cuda/python BONSAI_GROW_PROFILE=1 BONSAI_CUDA_PROFILE=1 \
  BONSAI_INGEST_PROFILE=1 BONSAI_FIT_PROFILE=1 \
  /opt/venv/bin/python scripts/bench_scaling.py --worker <<<"$spec" >/tmp/r.out 2>/tmp/r.err
grep -o "RESULT .*" /tmp/r.out
grep -E "grow-profile|fit-profile|ingest-profile|cuda-round-decomp" /tmp/r.err | tail -4
```

Reference variants for same-pod ladders: `xgb_cuda`, `lgbm_cuda`, `catboost_gpu` (same spec, different `variant`).
