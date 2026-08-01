# RunPod validation runbook — every workflow on the CLI

> **Status:** operational reference. Every command block here was used live during the 2026-07 optimization campaign; the failure-mode table at the end is the decoder ring for when they misbehave. GPU changes are validated on rented pods because the dev Mac has no CUDA device — the loop is: create pod → wait for SSH → clone/build → run the `[cuda]` suite + a profiled benchmark → read the profile lines → delete the pod.
>
> **Release wheels validate themselves**: wheels.yml's `validate-cuda` job (decision 70) automates this loop per release with the `RUNPOD_API_KEY` repo secret, booting the candidate runtime image (`ghcr.io/daniel-m-campos/bonsai:candidate-<tag>`) and running the wheel smokes on it, with an unconditional teardown and leftover-pod sweep. This runbook remains the manual path for development sessions, benchmarks, and anything the gate doesn't cover.

## 0. Prerequisites

- A RunPod API key (`rpa_...`). Keep it out of the repo and out of files — GitHub push protection has already blocked one near-miss commit of a key file. Export it per shell session and rotate it if it ever touches disk:

```bash
export RUNPOD_KEY="rpa_..."   # from https://console.runpod.io/user/settings
```

- Your SSH public key. The bonsai-ci image only installs the key you pass at create time (`PUBLIC_KEY` env); the account-level key in RunPod settings is used by the `ssh.runpod.io` proxy, which this image does NOT support (no proxy agent — direct IP only).

```bash
PUB=$(cat ~/.ssh/id_ed25519.pub)
```

- The image: `ghcr.io/daniel-m-campos/bonsai-ci:cuda12.4` (public GHCR, built by the repo's `ci-image` workflow from `docker/ci.Dockerfile`). CUDA 12.4 toolkit, clang-21 + libc++, cmake/ninja + python 3.12 with numpy/nanobind/xgboost/lightgbm/catboost at `/opt/venv`, FetchContent deps pre-baked. Clone-to-benchmark in under 5 minutes.

## 1. Create a pod

REST v1 is the reliable API surface (the GraphQL mutation works too; the v2 endpoint has had DNS outages). **`PUBLIC_KEY` in `env` is mandatory** — without it the pod boots but SSH refuses your key and the only fix is delete + recreate.

```bash
curl -s https://rest.runpod.io/v1/pods \
  -H "Authorization: Bearer $RUNPOD_KEY" -H 'content-type: application/json' \
  -d "{
    \"name\": \"bonsai-validate\",
    \"imageName\": \"ghcr.io/daniel-m-campos/bonsai-ci:cuda12.4\",
    \"gpuTypeIds\": [\"NVIDIA L40S\"],
    \"gpuCount\": 1,
    \"cloudType\": \"SECURE\",
    \"containerDiskInGb\": 80,
    \"ports\": [\"22/tcp\"],
    \"env\": {\"PUBLIC_KEY\": \"$PUB\"}
  }" | python3 -m json.tool
```

Note the returned `id`. GPU choice: L40S (SECURE, ~$1/hr) is the workhorse — consistently available with direct public IPs. A100-80GB SECURE (~$1.64/hr) is the fallback; community 4090/5090 capacity comes and goes hourly. Fleet caveat: identical code on two same-model pods has measured ~25% apart — **only same-pod before/after comparisons are valid**; never quote cross-pod absolute numbers.

## 2. Wait for liveness and get the SSH endpoint

`desiredStatus: RUNNING` from REST means nothing — the container may never have started. The API-side truth is GraphQL `runtime.uptimeInSeconds` (null = not started yet; a number = alive). The same query returns the SSH port mapping.

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

Read the entry with `"isIpPublic": true` and `"privatePort": 22` — that pair is your SSH target. Typical boot is 1–3 minutes (image pull is cached in each datacenter after the first use). If `runtime` stays null past ~5 minutes, read the container logs in the web console (the reason lives only there) and see the failure table.

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

## 4. Clone and build

```bash
$SSH 'export PATH=/opt/venv/bin:$PATH && cd /root \
  && git clone --depth 1 https://github.com/daniel-m-campos/bonsai.git && cd bonsai \
  && git fetch --depth 1 origin <branch> && git checkout -f FETCH_HEAD \
  && make python-cuda PYTHON=/opt/venv/bin/python 2>&1 | tail -1 \
  && cmake --build build-cuda --target bonsai_tests -j"$(nproc)" 2>&1 | tail -1'
```

Shallow single-branch clones have no `origin/<branch>` refs — **checkout `FETCH_HEAD`** after fetching the branch (this trap has been hit three times). `make python-cuda` builds the CUDA python module into `build-cuda/python`; the tests target builds the Catch2 binary.

## 5. Run the validation

The CUDA test suite (cases SKIP without a device, so a passing run on a GPU host is the real signal):

```bash
$SSH 'cd /root/bonsai && ./build-cuda/tests/bonsai_tests "[cuda]" 2>&1 | tail -2'
```

The profiled single-cell benchmark — a spec JSON piped into the bench harness's worker mode. This is the 16M×100 ledger cell; scale `rows`/`iters` down for smoke tests. The four profile envs produce the `grow-profile` / `cuda-profile` / `cuda-upload-decomp` / `ingest-profile` / `fit-profile` stderr lines that every optimization decision is priced against:

```bash
$SSH 'cd /root/bonsai && spec="{\"variant\":\"bonsai_cuda_depthwise\",\"cell\":{\"axis\":\"rows\",\"rows\":16000000,\"cols\":100,\"bins\":255,\"bins_effective\":255,\"depth\":8,\"iters\":100,\"lr\":0.1,\"informative\":20,\"n_test\":500000,\"seed\":42},\"threads\":16}"; \
  PYTHONPATH=$PWD/build-cuda/python BONSAI_GROW_PROFILE=1 BONSAI_CUDA_PROFILE=1 \
  BONSAI_INGEST_PROFILE=1 BONSAI_FIT_PROFILE=1 \
  /opt/venv/bin/python scripts/bench_scaling.py --worker <<<"$spec" > /tmp/run.out 2> /tmp/run.err; \
  grep -o "RESULT .*" /tmp/run.out; grep -E "grow-profile|fit-profile|ingest-profile|cuda-upload-decomp" /tmp/run.err | tail -4'
```

Other variants for same-pod ladders: `xgb_cuda`, `lgbm_cpu`, `catboost_gpu`, `bonsai_cuda_oblivious`, etc. (the `VARIANTS` table in `scripts/bench_scaling.py`). Full sweeps go through `make bench-scaling ARGS="--axis rows"` instead of raw worker calls — only the make target writes the results JSONL.

For anything longer than a few minutes, detach it so an SSH drop doesn't kill the run, and poll:

```bash
$SSH 'cd /root/bonsai && nohup env PATH=/opt/venv/bin:$PATH bash my_run.sh > /root/run.log 2>&1 & echo launched'
$SSH 'tail -5 /root/run.log'
```

When killing a detached run, kill explicit PIDs and verify `pgrep` returns nothing before relaunching — an orphaned worker (PPID 1) still fitting 16M rows makes every subsequent measurement ~2× slow (the contention signature: everything inflates uniformly, including phases you didn't touch).

## 6. Campaign driver

For multi-hour campaigns (spec ladders, Pareto frontiers), raw heredoc drivers are retired: `scripts/pod_bench_driver.sh` is the committed template. Copy it to the pod and launch it under `setsid` so an SSH drop cannot kill the run, then poll the log:

```bash
scp -i ~/.ssh/id_ed25519 -P $PORT scripts/pod_bench_driver.sh root@$IP:/root/
$SSH 'setsid env HOST_TAG=<gpu-tag> BRANCH=main SPEC=gpu-pareto-16M \
  OUT=/root/gpu-pareto-16M.jsonl RUN_LABEL=<label> \
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
| `runtime` stays null forever | Image needs a newer driver than the host (e.g. any `cu1281` image on a 12.4-driver machine — most of the SECURE pool) | Use the bonsai-ci image (cu12.4) or check driver via console logs; delete and re-roll |
| runc mount error in console, crash-loop | Broken host (missing `/dev/dri/cardN`) | Delete immediately, re-roll — waiting never helps |
| `Permission denied (publickey)` | Pod created without `PUBLIC_KEY` env | Delete + recreate; env cannot be added to a running pod |
| `ssh.runpod.io`: "container not found" | Image lacks RunPod's proxy agent (bonsai-ci is plain sshd) | Use direct IP + port from the GraphQL port mapping |
| `cmake: not found` on the pod | sshd didn't inherit Docker ENV | `export PATH=/opt/venv/bin:...` (section 3) |
| SSH session dies with exit 255 during cleanup | Your `pkill -f` matched the session's own argv | Separate ssh calls or bracket the pattern |
| `origin/<branch>` doesn't exist | Shallow single-branch clone | `git fetch origin <branch> && git checkout FETCH_HEAD` |
| Everything ~2× slower than the last run, uniformly | Orphaned worker from a killed run still computing | `pgrep -af python`, kill PIDs, verify empty, re-run |
| One pod much slower than another, same GPU model | Fleet variance (~25% measured) or the defective-host class (GPU sync ~300µs vs ~4µs healthy) | Same-pod comparisons only; for latency-sensitive work run a 30s sync probe first and reject hosts >50µs |
| "No resources" / capacity errors | DC out of that GPU | Retry, switch GPU type (L40S↔A100), or switch cloudType |
| Create succeeded per billing but API returned an error | Known API quirk | Always list-and-sweep after failures (section 7) |

## 9. Cost notes

L40S SECURE ≈ $0.99/hr, billed per-minute while the pod exists (not just while computing). A typical validation session — boot, build, `[cuda]` suite, two 16M profiled fits, teardown — is 25–40 minutes ≈ **$0.40–0.70**. The entire 2026-07 optimization campaign's pod spend was dominated by a handful of full-fleet sweeps, not validation sessions; there is no reason to hesitate about spinning a pod to check a GPU change, and no excuse for leaving one running overnight.

## 10. Standings refresh (decisions 92, 96)

The refresh runs from a developer machine via `scripts/standings_refresh.py`, not CI: the standings-refresh workflow retired under decision 96 after failing its tail on all four dispatches, each a bug only discoverable by paying for the ~3h pod run that reaches it. The measurement half was always reliable; only the untestable-without-a-pod tail broke.

**Prerequisites.** `RUNPOD_API_KEY` exported in the shell (never printed, never committed; rotate if it ever touches disk) and the SSH key from section 0 (`PUBLIC_KEY` env at pod create time; the account-level key does not work against this image).

```bash
export RUNPOD_API_KEY="rpa_..."
```

**Phase 1: measure.** Rents one L40S, runs `scripts/standings_refresh_pod.sh` detached on the pod, polls the run log, and pulls the axis jsonl files (plus `ab.jsonl` when `--prev-version` is set) incrementally, every poll rather than only at the end, so a pod that dies late still leaves finished axes on this machine.

```bash
python3 scripts/standings_refresh.py measure --axes rows,width,frontier,airline \
  --prev-version <last-release-version>
```

A `finally` block deletes the pod and sweeps any stray `bonsai-standings-*` pods regardless of how the run ends. Verify the fleet is empty afterward (same check as section 7): zero pods listed means zero billing. `--keep-pod` skips teardown for debugging; delete it yourself if you use it. Results land in a dated directory printed at the end (`--out-dir` to choose one).

**Phase 2: supersede.** Works from any local results directory, independent of the pod, so a failed or interrupted supersede reruns without paying for measurement again: copies the axis files into `benchmarks/results/`, updates the registry per axis, stages `git add -A benchmarks/` **before** rendering (the committed-files gate reads `git ls-files`, and a month-rollover refresh deletes the old dated files), renders, prints the A/B verdict from `ab.jsonl`, then branches, commits, and opens the PR.

```bash
python3 scripts/standings_refresh.py supersede --results-dir standings-<date> \
  --axes rows,width,frontier,airline
```

`--no-pr` stops after the local commit so you can inspect the diff before pushing and opening the PR by hand.

Release ordering is unchanged from decision 92: merge the version-bump PR FIRST, then run the refresh with `--prev-version` = the last release, merge the refresh PR (a **moved** verdict needs a `Standings:`-tagged decision first; docs-check enforces), then tag. The order matters because `update_standings.py` stamps `refreshed_for` from pyproject at the refresh's checkout sha: a refresh run before the bump stamps the old version and the publish gate then fails at tag time.
