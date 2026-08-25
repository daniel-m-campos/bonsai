# RunPod validation runbook: every workflow on the CLI

> **Status:** operational reference. Every command block here was used live during the 2026-07 optimization campaign; the failure-mode table at the end is the decoder ring for when they misbehave. GPU changes are validated on rented pods because the dev Mac has no CUDA device, so the loop is: create pod → wait for SSH → clone/build → run the `[cuda]` suite + a profiled benchmark → read the profile lines → delete the pod.
>
> **Release wheels validate themselves**: wheels.yml's `validate-cuda` job (decision 70) automates this loop per release with the `RUNPOD_API_KEY` repo secret, booting the candidate runtime image (`ghcr.io/daniel-m-campos/bonsai:candidate-<tag>`) and running the wheel smokes on it, with an unconditional teardown and leftover-pod sweep. This runbook remains the manual path for development sessions, benchmarks, and anything the gate doesn't cover.

## 0. Prerequisites

- A RunPod API key (`rpa_...`). Keep it out of the repo and out of files. GitHub push protection has already blocked one near-miss commit of a key file. Export it per shell session and rotate it if it ever touches disk:

```bash
export RUNPOD_KEY="rpa_..."   # from https://console.runpod.io/user/settings
```

- Your SSH public key. The bonsai-ci image only installs the key you pass at create time (`PUBLIC_KEY` env); the account-level key in RunPod settings is used by the `ssh.runpod.io` proxy, which this image does NOT support (no proxy agent, direct IP only).

```bash
PUB=$(cat ~/.ssh/id_ed25519.pub)
```

- The image: `ghcr.io/daniel-m-campos/bonsai-ci:cuda12.8` (public GHCR, built by the repo's `ci-image` workflow from `docker/ci.Dockerfile`). CUDA 12.8 toolkit, clang-21 + libc++, cmake/ninja + python 3.12 with numpy/nanobind/xgboost/lightgbm/catboost at `/opt/venv`, FetchContent deps pre-baked. Clone-to-benchmark in under 5 minutes. cuda12.8 covers sm_120 (Blackwell) natively, no toolkit side-install; L40S r550-driver pods still run its 12.8-built binaries via CUDA minor-version compatibility. Branches pinned to the old `cuda12.4` tag keep pulling it unchanged.

## 1. Create a pod

REST v1 is the reliable API surface (the GraphQL mutation works too; the v2 endpoint has had DNS outages). **`PUBLIC_KEY` in `env` is mandatory**: without it the pod boots but SSH refuses your key and the only fix is delete + recreate.

```bash
curl -s https://rest.runpod.io/v1/pods \
  -H "Authorization: Bearer $RUNPOD_KEY" -H 'content-type: application/json' \
  -d "{
    \"name\": \"bonsai-validate\",
    \"imageName\": \"ghcr.io/daniel-m-campos/bonsai-ci:cuda12.8\",
    \"gpuTypeIds\": [\"NVIDIA L40S\"],
    \"gpuCount\": 1,
    \"cloudType\": \"SECURE\",
    \"containerDiskInGb\": 80,
    \"ports\": [\"22/tcp\"],
    \"env\": {\"PUBLIC_KEY\": \"$PUB\"}
  }" | python3 -m json.tool
```

Note the returned `id`. GPU choice: L40S (SECURE, ~$1/hr) is the workhorse, consistently available with direct public IPs. A100-80GB SECURE (~$1.64/hr) is the fallback; community 4090/5090 capacity comes and goes hourly. Fleet caveat: identical code on two same-model pods has measured ~25% apart, so **only same-pod before/after comparisons are valid**; never quote cross-pod absolute numbers.

### Pin the datacenter

A create that names no datacenter is placed wherever RunPod picks, which fails with a 500 "no instances" while that same GPU sits in stock two regions over (measured on the 2026-08-14 GPU refresh). Read the per-datacenter stock first, then pin one in the create body.

The stock reading is a v2 catalog call, on `api.runpod.io` rather than the `rest.runpod.io` host the v1 pod endpoints live on:

```bash
GPU="NVIDIA RTX PRO 6000 Blackwell Server Edition"
curl -s "https://api.runpod.io/v2/catalog/gpus?include=AVAILABILITY&cloud=SECURE&product=POD" \
  -H "Authorization: Bearer $RUNPOD_KEY" \
  | python3 -c "import json,sys,os; g=next(x for x in json.load(sys.stdin)['gpus'] if x['id']==os.environ['GPU']); print(g['dataCenters'])"
```

Each entry is `{id, name, availability}` with availability `HIGH`, `MEDIUM`, `LOW`, or `NONE`. Pick one that is not `NONE` and add `"dataCenterIds": ["EU-RO-1"]` to the create body above.

Three rules, all measured:

- **One datacenter per create.** A multi-entry `dataCenterIds` list can 400 on the create schema, so iterate over candidates on failure instead of listing them all at once.
- **A 400 on a pinned create means try the next one, not stop.** The v1 create schema carries its own enum of datacenter ids and it is not the set the catalog reports: US-MO-2 and US-NC-2 are readable as stocked and rejected at create.
- **EUR-IS-2 is never rented.** That is policy, not availability; skip it whatever it reports.

`standings_refresh.py` encodes all three in `stocked_datacenters()`, tries the stocked datacenters best-first with one create each, and falls back to an unpinned create if the catalog call fails, so a lookup outage is never worse than not pinning.

## 2. Wait for liveness and get the SSH endpoint

`desiredStatus: RUNNING` from REST means nothing: the container may never have started. REST publishes `publicIp` and `portMappings` once the pod is placed, which is necessary but still not proof the container is up, so the only trustworthy readiness signal is sshd answering. `standings_refresh.py` polls the REST mapping and then probes ssh; the legacy GraphQL `runtime.uptimeInSeconds` query this runbook used to recommend now returns 403 for current API keys.

```bash
POD=<pod-id>
while true; do
  OUT=$(curl -s https://api.runpod.io/graphql \
    -H "Authorization: Bearer $RUNPOD_KEY" -H 'content-type: application/json' \
    -d "{\"query\":\"query{pod(input:{podId:\\\"$POD\\\"}){runtime{uptimeInSeconds ports{ip isIpPublic privatePort publicPort type}}}}\"}")
  echo "$OUT" | grep -q '"runtime":null' || { echo "$OUT" | python3 -m json.tool; break; }
  sleep 10
done
```

Read the entry with `"isIpPublic": true` and `"privatePort": 22`: that pair is your SSH target. Typical boot is 1–3 minutes (image pull is cached in each datacenter after the first use). If `runtime` stays null past ~5 minutes, read the container logs in the web console (the reason lives only there) and see the failure table.

## 3. Connect

```bash
IP=<public ip>; PORT=<public port for 22>
SSH="ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -p $PORT root@$IP"
$SSH 'nvidia-smi --query-gpu=name,driver_version --format=csv,noheader'
```

Two traps:

- **sshd sessions do not inherit the Docker image's ENV**, so `cmake`, `python`, and everything else in `/opt/venv` is off PATH. Prefix every session (or every remote command) with:

```bash
export PATH=/opt/venv/bin:/root/.local/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
```

- **`pkill -f <script>` inside an SSH remote command matches the SSH session's own command line** (the whole string is the remote bash's argv) and kills your session with exit 255. Split kill and relaunch into separate `ssh` invocations, or bracket the pattern: `pkill -f "pod_validat[e]"`.

Before any benchmark that quotes a reference library, check that the reference wheels can actually reach the device. The PyPI wheels bundle their own CUDA runtime, and a wheel built against a newer CUDA major than the host driver supports cannot see the GPU at all: xgboost logs "No visible GPU is found" and trains on CPU without raising, which is a plausible-looking number rather than a failure. The driver's ceiling is the `CUDA Version` field of `nvidia-smi`, and CUDA 13 wheels need an r580 driver, so they cannot run on the 12.4-driver pool at all.

```bash
$SSH 'export PATH=/opt/venv/bin:$PATH; nvidia-smi | head -3
  python -c "import xgboost; print(xgboost.__version__, xgboost.build_info()[\"CUDA_VERSION\"])"'
```

## 4. Clone and build

```bash
$SSH 'export PATH=/opt/venv/bin:$PATH && cd /root \
  && git clone --depth 1 https://github.com/daniel-m-campos/bonsai.git && cd bonsai \
  && git fetch --depth 1 origin <branch> && git checkout -f FETCH_HEAD \
  && make python-cuda PYTHON=/opt/venv/bin/python 2>&1 | tail -1 \
  && cmake --build build-cuda --target bonsai_tests -j"$(nproc)" 2>&1 | tail -1'
```

Shallow single-branch clones have no `origin/<branch>` refs, so **checkout `FETCH_HEAD`** after fetching the branch (this trap has been hit three times). `make python-cuda` builds the CUDA python module into `build-cuda/python`; the tests target builds the Catch2 binary.

## 5. Run the validation

The CUDA test suite (cases SKIP without a device, so a passing run on a GPU host is the real signal):

```bash
$SSH 'cd /root/bonsai && ./build-cuda/tests/bonsai_tests "[cuda]" 2>&1 | tail -2'
```

The profiled single-cell benchmark is a spec JSON piped into the bench harness's worker mode. This is the 16M×100 ledger cell; scale `rows`/`iters` down for smoke tests. The four profile envs produce the `grow-profile` / `cuda-profile` / `cuda-upload-decomp` / `ingest-profile` / `fit-profile` stderr lines that every optimization decision is priced against:

```bash
$SSH 'cd /root/bonsai && spec="{\"variant\":\"bonsai_cuda_depthwise\",\"cell\":{\"axis\":\"rows\",\"rows\":16000000,\"cols\":100,\"bins\":255,\"bins_effective\":255,\"depth\":8,\"iters\":100,\"lr\":0.1,\"informative\":20,\"n_test\":500000,\"seed\":42},\"threads\":16}"; \
  PYTHONPATH=$PWD/build-cuda/python BONSAI_GROW_PROFILE=1 BONSAI_CUDA_PROFILE=1 \
  BONSAI_INGEST_PROFILE=1 BONSAI_FIT_PROFILE=1 \
  /opt/venv/bin/python scripts/bench_scaling.py --worker <<<"$spec" > /tmp/run.out 2> /tmp/run.err; \
  grep -o "RESULT .*" /tmp/run.out; grep -E "grow-profile|fit-profile|ingest-profile|cuda-upload-decomp" /tmp/run.err | tail -4'
```

Other variants for same-pod ladders: `xgb_cuda`, `lgbm_cpu`, `catboost_gpu`, `bonsai_cuda_levelwise`, etc. (the `VARIANTS` table in `scripts/bench_scaling.py`). Full sweeps go through `make bench-scaling ARGS="--axis rows"` instead of raw worker calls; only the make target writes the results JSONL.

For anything longer than a few minutes, detach it so an SSH drop doesn't kill the run, and poll:

```bash
$SSH 'cd /root/bonsai && nohup env PATH=/opt/venv/bin:$PATH bash my_run.sh > /root/run.log 2>&1 & echo launched'
$SSH 'tail -5 /root/run.log'
```

When killing a detached run, kill explicit PIDs and verify `pgrep` returns nothing before relaunching. An orphaned worker (PPID 1) still fitting 16M rows makes every subsequent measurement ~2× slow (the contention signature: everything inflates uniformly, including phases you didn't touch).

## 6. Campaign driver

For multi-hour campaigns (spec ladders, Pareto frontiers), raw heredoc drivers are retired: `scripts/pod_bench_driver.sh` is the committed template. Copy it to the pod and launch it under `setsid` so an SSH drop cannot kill the run, then poll the log:

```bash
scp -i ~/.ssh/id_ed25519 -P $PORT scripts/pod_bench_driver.sh root@$IP:/root/
$SSH 'setsid env HOST_TAG=<gpu-tag> BRANCH=main SPEC=gpu-wide \
  OUT=/root/gpu-wide.jsonl RUN_LABEL=<label> \
  bash /root/pod_bench_driver.sh > /root/campaign.log 2>&1 < /dev/null & echo launched'
$SSH 'tail -5 /root/campaign.log'
```

The script clones or fetches the branch, builds `python-cuda` (side-installing CUDA 12.8 on Blackwell hosts), and runs the spec through `python -m bonsai.bench run`. Re-invoking it after a spot reap or crash resumes: rows already in `$OUT` are kept and only failed or missing jobs re-attempt. One-off probes can still use the worker-mode call in section 5.

## 7. Tear down and sweep

```bash
curl -s -X DELETE https://rest.runpod.io/v1/pods/$POD -H "Authorization: Bearer $RUNPOD_KEY"
curl -s https://rest.runpod.io/v1/pods -H "Authorization: Bearer $RUNPOD_KEY" | python3 -m json.tool | grep -c '"id"'
```

The sweep is not optional: **an error-returning create can still have created a billing pod.** After any failed create, list and delete strays. Zero pods listed = zero billing.

## 8. Failure decoder ring

| Symptom | Cause | Fix |
|---|---|---|
| `runtime` stays null forever | Image needs a newer driver than the host (e.g. any `cu1281` image on a 12.4-driver machine, most of the SECURE pool) | The bonsai-ci image is now cuda12.8; L40S r550-driver pods run its 12.8-built binaries via CUDA minor-version compatibility. If it still stalls, check driver via console logs; delete and re-roll |
| A reference arm trains at CPU speed on a GPU pod, or the bench raises "silently fell back to CPU" | The library's PyPI wheel is built against a newer CUDA major than the host driver supports; xgboost 3.4.0 ships a CUDA 13.3 build, which needs an r580 driver and sees zero devices on the 12.4-driver pool | Compare `xgboost.build_info()["CUDA_VERSION"]` against the `CUDA Version` in `nvidia-smi` (section 3), then pin the last CUDA 12 release (`uv pip install "xgboost<3.4"`) or re-roll onto a newer-driver host |
| runc mount error in console, crash-loop | Broken host (missing `/dev/dri/cardN`) | Delete immediately, re-roll; waiting never helps |
| `Permission denied (publickey)` | Pod created without `PUBLIC_KEY` env | Delete + recreate; env cannot be added to a running pod |
| `ssh.runpod.io`: "container not found" | Image lacks RunPod's proxy agent (bonsai-ci is plain sshd) | Use direct IP + port from the GraphQL port mapping |
| `cmake: not found` on the pod | sshd didn't inherit Docker ENV | `export PATH=/opt/venv/bin:...` (section 3) |
| SSH session dies with exit 255 during cleanup | Your `pkill -f` matched the session's own argv | Separate ssh calls or bracket the pattern |
| `origin/<branch>` doesn't exist | Shallow single-branch clone | `git fetch origin <branch> && git checkout FETCH_HEAD` |
| Everything ~2× slower than the last run, uniformly | Orphaned worker from a killed run still computing | `pgrep -af python`, kill PIDs, verify empty, re-run |
| One pod much slower than another, same GPU model | Fleet variance (~25% measured) or the defective-host class (GPU sync ~300µs vs ~4µs healthy) | Same-pod comparisons only; for latency-sensitive work run a 30s sync probe first and reject hosts >50µs |
| "No resources" / capacity errors, 500 "no instances" | The datacenter RunPod placed the create in is out of that GPU, even when others hold stock | Pin a stocked datacenter (section 1), then switch GPU type (L40S↔A100) or cloudType if every one is empty |
| A pinned create 400s on the request body | The create schema's datacenter enum is narrower than the catalog's stock list (US-MO-2, US-NC-2) | Try the next stocked datacenter; send one `dataCenterIds` entry per create |
| Create succeeded per billing but API returned an error | Known API quirk | Always list-and-sweep after failures (section 7) |
| CPU pod create 500s with "Container Disk must be less than or equal to 20" | CPU pods cap the container disk per flavor: 20GB on the CPU3 flavors, 30GB on the CPU5 ones, against a GPU pod's 80 | Ask for 30 or less (`CPU_DISK_GB` in `standings_refresh.py`) |
| A CPU pod reports the host's RAM, or its `cpu.max` reads `max` | A CPU pod caps by cpuset and `memory.max`, not by CFS bandwidth, and `free`/`/proc/meminfo` show the machine rather than the container | Read `nproc` for CPUs and `/sys/fs/cgroup/memory.max` for RAM (section 11) |

## 9. Cost notes

L40S SECURE ≈ $0.99/hr, billed per-minute while the pod exists (not just while computing). A typical validation session (boot, build, `[cuda]` suite, two 16M profiled fits, teardown) is 25–40 minutes ≈ **$0.40–0.70**. The entire 2026-07 optimization campaign's pod spend was dominated by a handful of full-fleet sweeps, not validation sessions; there is no reason to hesitate about spinning a pod to check a GPU change, and no excuse for leaving one running overnight.

## 10. Standings refresh (decisions 92, 96)

The refresh runs from a developer machine via `scripts/standings_refresh.py`, not CI: the standings-refresh workflow retired under decision 96 after failing its tail on all four dispatches, each a bug only discoverable by paying for the ~3h pod run that reaches it. The measurement half was always reliable; only the untestable-without-a-pod tail broke.

**Prerequisites.** `RUNPOD_API_KEY` exported in the shell (never printed, never committed; rotate if it ever touches disk) and the SSH key from section 0 (`PUBLIC_KEY` env at pod create time; the account-level key does not work against this image).

```bash
export RUNPOD_API_KEY="rpa_..."
```

**Phase 1: measure.** Rents one RTX PRO 6000 Blackwell (96GB, the host the redesigned matrix is measured on), runs `scripts/standings_refresh_pod.sh` detached on the pod, polls the run log, and pulls the axis jsonl files (plus `ab.jsonl` when `--prev-version` is set) incrementally, every poll rather than only at the end, so a pod that dies late still leaves finished axes on this machine.

```bash
python3 scripts/standings_refresh.py measure --only-stale \
  --prev-version <last-release-version>
```

The GPU create is pinned to a stocked datacenter per section 1, one per attempt; `measure --dry-run` prints the list it would try, live, without renting anything.

A `finally` block deletes the pod and sweeps any stray `bonsai-standings-*` pods regardless of how the run ends. Verify the fleet is empty afterward (same check as section 7): zero pods listed means zero billing. `--keep-pod` skips teardown for debugging; delete it yourself if you use it. Results land in a dated directory printed at the end (`--out-dir` to choose one).

The six axes are the scenario matrix of decision 103: `gpu-tall`, `gpu-wide`, `gpu-extreme`, `cpu-tall`, `cpu-wide`, `gpu-early-stop`, each one bundled spec of the same name writing `<axis>-YYYY-MM.jsonl`. `--axes` selects a subset; `--only-stale` intersects that subset with `python3 scripts/check_standings.py --stale`, which lists the axes whose plane digest has moved since their last refresh, so a CUDA-only change measures the gpu axes and leaves the cpu ones alone. Run without `--only-stale` for a release refresh, which re-measures everything on one host.

`gpu-extreme` is the VRAM-maxout scenario: 16,777,216 x 1,024 is 2^34 cells, so its float32 input needs about 68.7GB of RAM on the host that generates it. The pod script checks the container's ceiling first, `/sys/fs/cgroup/memory.max` where one is set and the machine's RAM otherwise, and prints a `SKIP gpu-extreme:` line if the rental is thinner, so a degraded session is loud rather than silent; pick a higher-RAM machine and rerun that axis alone. Reading `free` alone would be worse than no check at all, since it reports the machine and passes inside any container.

**Phase 2: supersede.** Works from any local results directory, independent of the pod, so a failed or interrupted supersede reruns without paying for measurement again: copies the axis files into `benchmarks/results/` (with `parity.jsonl` beside them as the gpu-tall axis's companion, which is where the perf page's one fused fit total comes from), updates the registry per axis, stages `git add -A benchmarks/` **before** rendering (the committed-files gate reads `git ls-files`, and a month-rollover refresh deletes the old dated files), renders, prints the A/B verdict from `ab.jsonl`, then branches, commits, and opens the PR.

```bash
python3 scripts/standings_refresh.py supersede --results-dir standings-<date> \
  --axes gpu-tall,gpu-wide,gpu-extreme,cpu-tall,cpu-wide,gpu-early-stop
```

`--no-pr` stops after the local commit so you can inspect the diff before pushing and opening the PR by hand.

Release ordering is unchanged from decision 92: merge the version-bump PR FIRST, then run the refresh with `--prev-version` = the last release and no `--only-stale` (a release re-measures every axis on one host), merge the refresh PR (a **moved** verdict needs a `Standings:`-tagged decision first; docs-check enforces), then tag. The order matters because `update_standings.py` stamps `refreshed_for` from pyproject at the refresh's checkout sha: a refresh run before the bump stamps the old version and the publish gate then fails at tag time.

The CPU axes are measured on the GPU pod's own CPU, so a full refresh is one rental; see section 11 before running `cpu-tall` or `cpu-wide`.

A pushed tag alone publishes nothing: wheels.yml's publish path triggers on the GitHub release event, so the release recipe ends with `gh release create v<version> --notes-file <changelog section>`, which is what fires the build, the pod self-validation, the standings gate, and the PyPI upload. This has been re-learned twice; it is written here so there is no third time.

## 11. The CPU plane host

**The host of record for the CPU plane is a GPU pod's CPU, at 12 threads.** That is server silicon, an EPYC or Xeon rather than the desktop-class parts a CPU rental buys, metered by a bandwidth quota exactly as a cloud VM or a Kubernetes job is. The choice is measured rather than preferred: of four hosts timed at the tall cell, the two server-class ones agreed on the ordering while a 16-core dual-channel EPYC and a shared Threadripper disagreed with them and with each other, moving LightGBM by 3.3x. A CPU ranking is a claim about a class of machine as much as about an engine, so the class is pinned, disclosed in every row, and reproducible by anyone who rents the same thing.

A GPU rental sells you a device and leaves the CPU share to whatever the host has spare, so that ceiling has to be read rather than assumed. The standings pods advertise 128 vCPUs and cap the container at 13.6 cores, other pods in the same fleet cap at 27.2, and `nproc` shows neither number. bonsai's barriers spin-wait, so they spend the capped bandwidth on waiting, and a 16-thread fit sat on the ceiling at 13.4 cpu-seconds per wall second, throttled in 97% of enforcement periods against XGBoost's 1.2% (issue #355 step 18).

**The 1.5-cores-per-thread rule is retired.** An earlier version of this section demanded that margin, and no rental reliably offers it: twelve consecutive draws across the L40S and A100 families all metered 13.6 cores, and the one 26.35-core host found in thirteen attempts was a machine, not a family trait. A margin that cannot be bought is not a protocol. What replaces it is one spare core plus a measurement. The CPU specs pin 12 threads under a 13.6-core quota, which leaves 1.6 cores for the allocator, the runtime's own threads, and the reference library's helpers, and the throttle counters bracketing each axis's first fit measure directly what the margin was guessing at.

**CPU pods remain available** through `--cpu-plane-host cpupod` (sized by `--cpu-vcpu`), and they cap by a different mechanism. They are not the host of record, because their silicon is not what anyone trains on, but the difference is worth having written down. Measured on one rented pod (`cpu3c`, 2 vCPU, AMD EPYC 7713, US-CA-2):

| file | reading | meaning |
|---|---|---|
| `/sys/fs/cgroup/cpu.max` | `max 100000` | no CFS bandwidth quota at all |
| `/sys/fs/cgroup/cpu.stat` | `nr_periods 0`, `nr_throttled 0` | bandwidth enforcement is not even running |
| `/sys/fs/cgroup/cpuset.cpus.effective` | two CPU ids | the purchase, enforced as an affinity mask |
| `nproc` | `2` | equals the rented vCPU count |
| `os.cpu_count()` | `256` | still the whole host, so read the mask, not this |
| `/sys/fs/cgroup/memory.max` | `4.0GB` | vCPU count times the flavor's GB per vCPU, exactly |
| `/proc/meminfo` MemTotal, `free -g` | 1TB | the host's RAM, **not** the container's; a RAM check must read `memory.max` |

So a CPU pod's cap is a cpuset, not a bandwidth quota, and it equals what was bought. Under a cpuset the container is handed whole CPUs, one per rented vCPU, and nothing meters how they are spent: a thread that spins at a barrier burns the core it already owns and takes nothing from its siblings, so spin-wait is free and one vCPU per thread is enough. Under a bandwidth quota the threads share one pool of core-seconds, a spinning thread draws on that pool while it waits, and the spare core plus the throttle counters are what keep the fit off the ceiling. A container that cannot be throttled cannot have its timing eaten by throttling, which is the one thing a CPU rental buys that a GPU pod does not. It also means a quota read alone reports "unlimited" on a CPU pod, which is why `cpu_quota()` and the pod script both take the tighter of the CFS quota and the cpuset.

The rental has been exercised against the API: `POST /v1/pods` with `computeType: "CPU"`, `cpuFlavorIds: ["cpu5g"]` and `vcpuCount: 16` returns 201 with `memoryInGb` 64 and `costPerHr` 0.736, matching the flavor's 4GB per vCPU and its published per-vCPU price. The MCP `create-pod` tool cannot express a CPU flavor or a vCPU count, so that REST call is the provisioning path. The assertion below still confirms each session's rental rather than trusting this table.

**Measuring.** One command; the CPU axes ride the same session that measures the device planes, and everything lands in one results directory:

```bash
python3 scripts/standings_refresh.py measure --prev-version <last-release>
python3 scripts/standings_refresh.py measure --dry-run     # plan and sizing only
python3 scripts/standings_refresh.py supersede --results-dir <dir>
```

`--dry-run` prints the plan and the thread count read out of the bundled specs, and rents nothing. `--cpu-plane-host` defaults to `gpu`, which is what makes the refresh one rental; pass `cpupod` to split the CPU axes onto their own pod. `--cpu-vcpu` sets how wide that `cpu5g` rental is (default 16); rent at least one vCPU per spec thread, since the cpuset is the whole cap there. The flavor is pinned at `cpu5g`, because the container disk is pinned at the CPU5 cap and any other flavor is refused at create. The gate itself is the pod script's, and only the pod can read the container it landed in. `parity.jsonl` and `ab.jsonl` are cuda measurements and belong to the GPU axes.

**The gate on the pod.** It reads `cpu.max` (with the v1 files as a fallback) for the bandwidth quota and `nproc` for the cpuset, then checks whichever cap the host actually carries: a quota host must exceed the spec's thread count by one spare core, a cpuset host must have at least one CPU per thread. It exports `OMP_WAIT_POLICY=passive` for the CPU arms only so the flag never crosses into a device axis, and brackets the axis's first fit with the `nr_throttled` counters. More than 5% throttled enforcement periods aborts the rest of the axis, and so does a probe that yields no percentage at all: a gate that cannot read its own counters has measured nothing, and nothing is not a pass. On a CPU pod that counter cannot move at all, since bandwidth enforcement is not running, which makes it a check on the rental rather than a knob to tune. `OMP_WAIT_POLICY=passive` stays on either way: it costs nothing where spin is free and it is the correct setting where spin is metered.

A failed gate is loud in three places: a `STANDINGS_REFRESH_FAIL` line in the run log, a `quota-fail.txt` that travels back with the results and makes `measure` exit nonzero, and the partial rows renamed to `QUOTAFAIL-<axis>-<YYYY-MM>.jsonl` so the supersession glob cannot pick them up. Every row also carries its host block, with `n_vcpu` counting the CPUs the container can see, `cpu_quota` the bandwidth ceiling that binds under them, and `omp_wait_policy` beside both, so a committed standings file says which machine and which ceiling produced it without anyone having to remember.

**Cost.** The CPU axes add pod minutes to a rental that is already running rather than a second invoice. Both cells, six variants, two repeats on the tall cell, sum to about 22 minutes of measured fits at the published numbers, plus data generation, on the same pod hour the GPU axes are paying for. A separate CPU rental is priced on its own: cpu5g is $0.046 per vCPU per hour, so a 16-vCPU CPU pod is about **$0.74/hr** and a full CPU refresh on one lands near **$1**.
