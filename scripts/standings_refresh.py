"""The standings refresh, run locally (decision 96): measure on a rented
pod, then build the supersession PR. Replaces the retired standings-refresh
CI workflow, whose runner-side tail failed on every dispatch while this
exact ritual succeeded by hand.

    export RUNPOD_API_KEY=rpa_...   # never printed; runbook section 0
    python3 scripts/standings_refresh.py measure --prev-version 1.5.4
    python3 scripts/standings_refresh.py supersede --results-dir <dir>

Phases are independent: `measure` rents one pod, runs
scripts/standings_refresh_pod.sh detached, polls, pulls the jsonl files
into a local directory, and tears the pod down (sweep + verify-empty).
`supersede` works from any results directory: registry update, staging,
render, A/B verdict, branch, and `gh pr create`. A failed supersede is
rerunnable without touching a pod; that separation is the whole point.

The axes are the redesigned scenario matrix (decision 103): tall and wide
iso-volume pairs on each plane, a VRAM-maxout extreme, and early stopping.
`measure --only-stale` asks `check_standings.py --stale` which of the
requested axes a source change can actually have moved, and runs only
those, so a CUDA-only change never pays for a CPU-plane sweep.

The parity rows do double duty. They gate the supersession, and the ones
that pass are committed beside the standings as the gpu-tall axis's
companion file, because the fused fit total the perf page publishes once is
their median: the anchor is read from a measurement the session already
takes, never from a new run.

The two planes rent different machines. A GPU pod is sold by device and
throws in whatever CPU share the host has left, which is how a 16-thread CPU
comparison ended up describing a cgroup rather than the code (issue #355):
the container's CPU ceiling is invisible to nproc, and bonsai's spin-wait
barriers spend exactly what the ceiling withholds. A CPU pod is sold by vCPU,
so the ceiling is a line on the invoice. The CPU axes therefore run on a
rented CPU pod sized by the rule below, and the GPU axes run on the GPU pod
exactly as before. Both sessions pull into one results directory, and every
row carries the host block it was measured under.

The sizing rule: rent one vCPU per thread the CPU specs ask for, 16 at the
published thread count. What makes that enough is the enforcement mechanism. A
CPU pod hands the container a cpuset, a set of whole CPUs equal to the
purchase, and a thread that spins at a barrier burns only the core it already
owns, so spin-wait costs the fit nothing. A host that meters CPU bandwidth
instead, which is what a GPU pod does, shares one pool of core-seconds across
all the threads, and there spin-wait does eat the allowance: such a host needs
1.5 cores per thread before its numbers mean anything. The pod script asserts
whichever cap it landed under before it measures anything, and fails the axis
rather than publishing a number made under a ceiling nobody declared.

It gates on the pod's fused/two-step parity rows before touching
anything: the published ingest/train split is only honest while bonsai's
two-step form still bins where its fused call does, so a parity failure
stops the supersession instead of shipping a breakdown that describes a
pipeline no cuda grower runs. A missing parity.jsonl fails the same gate:
absence means the check never ran (a lost scp, a pod that died before the
parity phase), not that it does not apply. `supersede --no-parity` accepts
a results dir with no parity evidence for the hosts where the check truly
cannot run.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request

REPO = pathlib.Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "results"
POD_SCRIPT = REPO / "scripts" / "standings_refresh_pod.sh"

REST = "https://rest.runpod.io/v1"
# cuda12.8: the GPU below is sm_120, past 12.4's --offload-arch=native reach.
IMAGE = "ghcr.io/daniel-m-campos/bonsai-ci:cuda12.8"
GPU = "NVIDIA RTX PRO 6000 Blackwell Server Edition"

# The CPU plane's rental. cpu5g is the general-purpose CPU5 flavor at 4GB of
# RAM per vCPU, which covers both cpu cells with room to spare. Not a knob:
# CPU_DISK_GB below is pinned at the CPU5 disk cap, so any other flavor fails
# at create.
CPU_FLAVOR = "cpu5g"
CPU_VCPU = 16
# CPU pods cap the container disk far below a GPU pod's 80GB: the API refuses
# anything over 20GB on the CPU3 flavors and 30GB on the CPU5 ones.
CPU_DISK_GB = 30
GPU_DISK_GB = 80
PLANE_GPU = "gpu"
PLANE_CPU = "cpu"

# The band the fused and two-step forms must agree inside. Measured at 2.5%
# on one L40S at 4M x 512 and inside repeat noise at 16M; 5% matches the A/B
# band, which is the same pod-noise question.
PARITY_BAND_PCT = 5

# Axis -> the stem of its dated results file; the bundled spec of the same
# name is what the pod script runs to produce it.
AXES = ("gpu-tall", "gpu-wide", "gpu-extreme", "cpu-tall", "cpu-wide",
        "gpu-early-stop")
AXIS_FILE = {axis: axis for axis in AXES}

# The axis the parity rows anchor: same cell, same pod, same session.
PARITY_AXIS = "gpu-tall"

# Axis name prefix -> the plane whose pod measures it.
CPU_PREFIX = "cpu-"
SPECS = REPO / "python" / "bonsai" / "bench" / "specs"
# The pod script's loud half: written when the CPU-cap assertion fails or
# the throttle bracket catches material throttling, pulled back with results.
QUOTA_FAIL = "quota-fail.txt"

SSH_OPTS = ["-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=15"]


def main() -> int:
    """Parse arguments and dispatch to the requested phase.

    Returns
    -------
    int
        Process exit code: 0 on success, 1 on a phase-reported failure.
    """
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="phase", required=True)
    m = sub.add_parser("measure", help="rent a pod, run the suites, pull results")
    m.add_argument("--axes", default=",".join(AXES))
    m.add_argument("--prev-version", default="",
                   help="wheel for the A/B old arm (empty skips the A/B)")
    m.add_argument("--out-dir", default=None,
                   help="where the jsonl files land (default: a dated dir)")
    m.add_argument("--keep-pod", action="store_true",
                   help="skip teardown (debugging; delete it yourself)")
    m.add_argument("--only-stale", action="store_true",
                   help="drop the requested axes whose plane digest has not "
                   "moved since their last refresh (check_standings --stale)")
    m.add_argument("--cpu-vcpu", type=int, default=CPU_VCPU,
                   help="vCPUs to buy for the cpu-plane pod; must be at "
                        "least one per thread the specs claim")
    m.add_argument("--dry-run", action="store_true",
                   help="print the rental plan and the sizing arithmetic, "
                        "then exit without renting anything")
    s = sub.add_parser("supersede", help="build the supersession PR from results")
    s.add_argument("--results-dir", required=True)
    s.add_argument("--axes", default=",".join(AXES))
    s.add_argument("--no-pr", action="store_true",
                   help="stop after commit (inspect before pushing)")
    s.add_argument("--no-parity", action="store_true",
                   help="accept a results dir with no parity evidence")
    args = ap.parse_args()
    if args.phase == "measure":
        return measure(args)
    return supersede(args)


# Measure ==========================================================================================

def measure(args: argparse.Namespace) -> int:
    """Run one pod session per plane: create, launch, poll, pull, tear down.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``measure`` arguments: ``axes``, ``only_stale``,
        ``prev_version``, ``out_dir``, ``keep_pod``, ``cpu_vcpu``,
        ``dry_run``.

    Returns
    -------
    int
        0 on success, 1 if the requested CPU rental is too small for the
        specs, ``RUNPOD_API_KEY`` is not set, or a pod's gate failed an axis.
    """
    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    if args.only_stale:
        stale = stale_axes()
        current = [a for a in axes if a not in stale]
        axes = [a for a in axes if a in stale]
        if current:
            print(f"--only-stale: {', '.join(current)} unchanged since their "
                  "last refresh, skipping")
        if not axes:
            print("every requested axis is current; nothing to measure")
            return 0
    planes = [(PLANE_GPU, [a for a in axes if not a.startswith(CPU_PREFIX)]),
              (PLANE_CPU, [a for a in axes if a.startswith(CPU_PREFIX)])]
    planes = [(plane, plane_axes) for plane, plane_axes in planes if plane_axes]
    cpu_axes = dict(planes).get(PLANE_CPU, [])
    if cpu_axes:
        # One vCPU per thread is the whole sizing rule: a rented CPU pod
        # enforces the purchase as a cpuset, so a thread that spins at a
        # barrier burns only the core it already owns.
        needed = max(spec_threads(axis) for axis in cpu_axes)
        print(f"cpu plane: {', '.join(cpu_axes)} at {needed}t need >= "
              f"{needed} vCPU (one per thread, because a cpu pod caps by "
              f"cpuset); renting {args.cpu_vcpu} x {CPU_FLAVOR}")
        if args.cpu_vcpu < needed:
            print(f"ERROR: --cpu-vcpu {args.cpu_vcpu} is below the {needed} "
                  f"the sizing rule requires; the axis would run more "
                  f"threads than the cpuset has cpus", file=sys.stderr)
            return 1
    if args.dry_run:
        print("--dry-run: nothing rented")
        return 0
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        print("ERROR: export RUNPOD_API_KEY first (runbook section 0)",
              file=sys.stderr)
        return 1
    out_dir = pathlib.Path(args.out_dir or
                           f"standings-{time.strftime('%Y%m%d-%H%M')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True, cwd=REPO).stdout.strip()
    pubkey = (pathlib.Path.home() / ".ssh" / "id_ed25519.pub").read_text().strip()
    for plane, plane_axes in planes:
        _run_session(key, args, plane=plane, axes=plane_axes, out_dir=out_dir,
                     sha=sha, pubkey=pubkey)
    quota_fail = out_dir / QUOTA_FAIL
    if quota_fail.exists():
        print("ERROR: a pod's gate failed an axis; its rows were renamed "
              "QUOTAFAIL-* and must not be superseded:\n"
              + quota_fail.read_text().rstrip(), file=sys.stderr)
        return 1
    print(f"results in {out_dir}/; next:\n"
          f"  python3 scripts/standings_refresh.py supersede "
          f"--results-dir {out_dir}")
    return 0


def spec_threads(axis: str) -> int:
    """The thread count an axis's bundled spec publishes its claim at."""
    spec = json.loads((SPECS / f"{axis}.json").read_text())
    return max(spec.get("threads", [16]))


# Supersede ========================================================================================

def supersede(args: argparse.Namespace) -> int:
    """Build the supersession PR from a local results directory.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``supersede`` arguments: ``results_dir``, ``axes``, ``no_pr``,
        ``no_parity``.

    Returns
    -------
    int
        0 on success, 1 if a required axis jsonl file is missing or the
        parity gate fails.
    """
    src = pathlib.Path(args.results_dir)
    axes = [a.strip() for a in args.axes.split(",")]
    parity_path = src / "parity.jsonl"
    parity, parity_ok = _parity(parity_path, allow_absent=args.no_parity)
    print(parity)
    if not parity_ok:
        if not parity_path.exists():
            print("ERROR: no parity.jsonl in this results dir; absence "
                  "means the check never ran (a lost scp, a pod that died "
                  "before the parity phase), which is exactly the failure "
                  "mode the gate exists to catch. Pass --no-parity to "
                  "proceed deliberately without parity evidence.",
                  file=sys.stderr)
        else:
            print("ERROR: fused/two-step parity failed; the ingest/train "
                  "split in these rows is not trustworthy. Fix the "
                  "runner's device hint and re-measure.", file=sys.stderr)
        return 1
    files = {}
    for axis in axes:
        got = sorted(src.glob(f"{AXIS_FILE[axis]}-*.jsonl"))
        if not got:
            print(f"ERROR: no {AXIS_FILE[axis]}-*.jsonl in {src}",
                  file=sys.stderr)
            return 1
        files[axis] = got[-1].name
        shutil.copy2(got[-1], RESULTS / got[-1].name)
    companion = _copy_parity(parity_path, files.get(PARITY_AXIS))
    for axis in axes:
        cmd = [sys.executable, "scripts/update_standings.py",
               "--axis", axis, "--file", files[axis]]
        if axis == PARITY_AXIS and companion:
            cmd += ["--companion", companion]
        subprocess.run(cmd, check=True, cwd=REPO)
    # Stage supersessions BEFORE rendering: the committed-files gate reads
    # git ls-files, and a month-rollover refresh deletes the old dated files.
    subprocess.run(["git", "add", "-A", "benchmarks/"], check=True, cwd=REPO)
    subprocess.run([sys.executable, "scripts/render_results.py"],
                   check=True, cwd=REPO)
    verdict = _verdict(src / "ab.jsonl")
    print(verdict or "A/B skipped (no ab.jsonl)")

    cpu_axes = [a for a in axes if a.startswith(CPU_PREFIX)]
    hosts_note = ("" if not cpu_axes else
                  f"\n\nCPU axes ({', '.join(cpu_axes)}) were measured on a "
                  "rented CPU pod, whose vCPU count is bought rather than "
                  "inherited from a GPU rental (issue #355). Every row "
                  "carries its own host block, so the registry records "
                  "which machine and which ceiling stands behind each axis.")
    branch = f"standings-refresh-{time.strftime('%Y%m%d')}"
    subprocess.run(["git", "checkout", "-b", branch], check=True, cwd=REPO)
    subprocess.run(["git", "add", "-A", "benchmarks/", "docs/method/",
                    "README.md"], check=True, cwd=REPO)
    axes_label = ",".join(axes)
    subprocess.run(["git", "commit", "-m",
                    f"bench(standings): refresh {axes_label}\n\n"
                    "Same-pod refresh via scripts/standings_refresh.py "
                    "(decision 96); superseded files deleted, registry "
                    "updated, ledger and README regenerated." + hosts_note],
                   check=True, cwd=REPO)
    if args.no_pr:
        print(f"committed on {branch}; push and open the PR when ready")
        return 0
    subprocess.run(["git", "push", "-u", "origin", branch], check=True, cwd=REPO)
    body = (f"Standings refresh via `scripts/standings_refresh.py` "
            f"(decision 96).{hosts_note}\n\nIngest/train parity (bonsai's "
            f"fused call vs the two-step Dataset form, same pod, interleaved, "
            f"+-{PARITY_BAND_PCT}% band):\n\n{parity}\n\n"
            f"A/B verdict (previous release wheel vs HEAD, "
            f"same pod, interleaved, +-5% band):\n\n"
            f"{verdict or 'A/B skipped.'}\n\nA **moved** verdict requires a "
            f"`Standings:`-tagged decision entry before merge; the docs-check "
            f"gate enforces it.")
    subprocess.run(["gh", "pr", "create", "--base", "main", "--title",
                    f"bench(standings): refresh {axes_label}", "--body", body],
                   check=True, cwd=REPO)
    return 0


def stale_axes() -> set[str]:
    """The axes `check_standings.py --stale` says a refresh must measure."""
    out = subprocess.run([sys.executable, "scripts/check_standings.py",
                          "--stale"], capture_output=True, text=True,
                         check=True, cwd=REPO)
    return {line.strip() for line in out.stdout.splitlines() if line.strip()}


# Private Helpers ==================================================================================

def _copy_parity(path: pathlib.Path, axis_file: str | None) -> str | None:
    """Commit the session's parity rows as the anchor axis's companion.

    The perf page publishes one fused fit total, and this is where it comes
    from, so the rows ship with the standings they anchor and supersede with
    them. The file is dated from the axis file it accompanies, which is the
    only honest date: the two were measured in the same session.

    Parameters
    ----------
    path : pathlib.Path
        The pod's ``parity.jsonl``.
    axis_file : str or None
        The dated standings file the parity rows anchor, or None when this
        session did not measure that axis.

    Returns
    -------
    str or None
        The committed file name, or None when there is nothing to commit.
    """
    if not axis_file or not path.exists():
        return None
    stamp = "-".join(pathlib.Path(axis_file).stem.split("-")[-2:])
    name = f"parity-{stamp}.jsonl"
    shutil.copy2(path, RESULTS / name)
    return name



def _run_session(key: str, args: argparse.Namespace, *, plane: str,
                 axes: list[str], out_dir: pathlib.Path, sha: str,
                 pubkey: str):
    """One pod, one plane: create, launch the on-pod script, poll, tear down.

    Both planes write into the same results directory. Their file names do
    not collide (one dated file per axis), and the GPU session owns the
    parity and A/B rows, which are cuda measurements the CPU pod skips.
    """
    pod_id = _create_pod(key, pubkey, plane=plane, vcpu=args.cpu_vcpu)
    print(f"{plane} pod {pod_id} created for {', '.join(axes)}; waiting for ssh")
    host_tag = (f"cpupod-{CPU_FLAVOR}-{args.cpu_vcpu}vcpu"
                if plane == PLANE_CPU else "")
    # The A/B compares cuda growers against a released wheel; only the GPU
    # session can run it, so the CPU pod is never asked to.
    prev_version = args.prev_version if plane == PLANE_GPU else ""
    try:
        ip, port = _wait_ssh(key, pod_id)
        ssh = ["ssh", "-i", str(pathlib.Path.home() / ".ssh" / "id_ed25519"),
               *SSH_OPTS, "-p", str(port), f"root@{ip}"]
        _wait_until(lambda: subprocess.run([*ssh, "true"],
                                           capture_output=True).returncode == 0,
                    timeout_s=180, what="sshd")
        subprocess.run(["scp", "-i", str(pathlib.Path.home() / ".ssh" / "id_ed25519"),
                        *SSH_OPTS, "-P", str(port), str(POD_SCRIPT),
                        f"root@{ip}:/root/"], check=True)
        # Detached: an ssh drop must not kill a multi-hour sweep.
        subprocess.run([*ssh, f"nohup env AXES='{','.join(axes)}' "
                        f"GIT_SHA='{sha}' "
                        f"PREV_VERSION='{prev_version}' "
                        f"PLANE='{plane}' HOST_TAG='{host_tag}' "
                        "bash /root/standings_refresh_pod.sh "
                        "> /root/refresh.log 2>&1 & echo launched"], check=True)
        _poll_pod_run(ssh, out_dir, ip, port)
    finally:
        if args.keep_pod:
            print(f"pod {pod_id} KEPT per --keep-pod; delete it yourself")
        else:
            _delete_pod(key, pod_id)
            _sweep(key)


def _api(url: str, key: str, payload: dict | None = None,
         method: str = "POST") -> dict:
    """One authenticated RunPod API call; the key never reaches stdout."""
    req = urllib.request.Request(
        url, method=method,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Authorization": f"Bearer {key}",
                 "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode()
    return json.loads(body) if body.strip() else {}


def _create_pod(key: str, pubkey: str, *, plane: str, vcpu: int) -> str:
    """Create a standings pod for one plane, with the mandated PUBLIC_KEY env.

    A GPU pod is bought by device. A CPU pod is bought by vCPU, which is the
    point: the CPU ceiling is a number the invoice states rather than
    whatever share of the host the device rental happened to leave over.
    """
    name = f"bonsai-standings-{plane}-{time.strftime('%Y%m%d-%H%M')}"
    body = {"name": name, "imageName": IMAGE, "cloudType": "SECURE",
            "ports": ["22/tcp"], "env": {"PUBLIC_KEY": pubkey}}
    if plane == PLANE_CPU:
        body |= {"computeType": "CPU", "cpuFlavorIds": [CPU_FLAVOR],
                 "vcpuCount": vcpu, "containerDiskInGb": CPU_DISK_GB}
    else:
        body |= {"gpuTypeIds": [GPU], "gpuCount": 1,
                 "containerDiskInGb": GPU_DISK_GB}
    for attempt in (1, 2):
        try:
            out = _api(f"{REST}/pods", key, body)
            if out.get("id"):
                return out["id"]
        except OSError as e:
            print(f"create attempt {attempt}: {e}", file=sys.stderr)
        time.sleep(15)
    raise SystemExit(f"no usable {plane} pod after 2 attempts")


def _wait_ssh(key: str, pod_id: str) -> tuple[str, int]:
    """The public (ip, port) for sshd, once sshd actually answers.

    Two waits, because neither signal alone is the truth. REST publishes
    publicIp and portMappings once the pod is placed, which is necessary
    but not sufficient: the container may still be starting. So the
    mapping is polled first, then sshd itself, which is the only signal
    that means the next step will work.
    """
    def mapping():
        out = _api(f"{REST}/pods/{pod_id}", key, method="GET")
        port = (out.get("portMappings") or {}).get("22")
        ip = out.get("publicIp")
        if ip and port:
            return ip, int(port)
        # A CPU pod publishes the same publicIp and portMappings pair a GPU
        # pod does, measured; runtime.ports is the older shape, kept because
        # a pod that reports it would otherwise look like it never placed.
        for entry in ((out.get("runtime") or {}).get("ports") or []):
            if entry.get("private") == 22 and entry.get("ip"):
                return entry["ip"], int(entry["public"])
        return None

    # A pod reports RUNNING long before it is reachable: a 2 vCPU cpu5g took
    # 173s to publish its mapping, and a rental large enough to wait for
    # placement takes longer, so this budget is generous on purpose.
    ip, port = _wait_until(mapping, timeout_s=900, what="pod port mapping")

    probe = ["ssh", "-i", str(pathlib.Path.home() / ".ssh" / "id_ed25519"),
             *SSH_OPTS, "-p", str(port), f"root@{ip}", "true"]
    def sshd():
        return subprocess.run(probe, capture_output=True).returncode == 0

    _wait_until(sshd, timeout_s=600, what="sshd")
    return ip, port


def _poll_pod_run(ssh: list[str], out_dir: pathlib.Path, ip: str, port: int):
    """Poll the detached run; pull the session directory incrementally.

    Pulling every poll (not just at the end) means a pod that dies late
    still leaves the finished axes on this machine. The whole directory
    comes over, not just *.jsonl, because the quota gate's marker file is
    the evidence that some of those axes must not be published.
    """
    scp_base = ["scp", "-i", str(pathlib.Path.home() / ".ssh" / "id_ed25519"),
                *SSH_OPTS, "-P", str(port)]
    while True:
        time.sleep(120)
        subprocess.run([*scp_base, f"root@{ip}:/root/standings/*",
                        str(out_dir) + "/"], capture_output=True)
        tail = subprocess.run([*ssh, "tail -2 /root/refresh.log"],
                              capture_output=True, text=True)
        last = tail.stdout.strip().splitlines()[-1:] or [""]
        print(f"  pod: {last[0][:110]}", flush=True)
        if "STANDINGS_REFRESH_DONE" in tail.stdout:
            subprocess.run([*scp_base, f"root@{ip}:/root/standings/*",
                            str(out_dir) + "/"], check=True)
            return
        if tail.returncode != 0:
            print("  ssh poll failed; retrying", flush=True)


def _delete_pod(key: str, pod_id: str):
    """Teardown; failures print but never mask the run's own status."""
    try:
        _api(f"{REST}/pods/{pod_id}", key, method="DELETE")
        print(f"pod {pod_id} deleted")
    except OSError as e:
        print(f"WARNING: delete failed for {pod_id}: {e}", file=sys.stderr)


def _sweep(key: str):
    """Delete stray bonsai-standings pods; loudly report survivors."""
    try:
        out = _api(f"{REST}/pods", key, method="GET")
    except OSError as e:
        print(f"WARNING: sweep list failed: {e}", file=sys.stderr)
        return
    items = out if isinstance(out, list) else out.get("items", [])
    strays = [p["id"] for p in items
              if str(p.get("name", "")).startswith("bonsai-standings")]
    for pid in strays:
        _delete_pod(key, pid)
    if strays:
        print(f"swept {len(strays)} stray pod(s)")


def _wait_until(fn, *, timeout_s: int, what: str):
    """Poll fn every 10s until truthy; SystemExit on timeout."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        got = fn()
        if got:
            return got
        time.sleep(10)
    raise SystemExit(f"timed out waiting for {what}")


def _parity(path: pathlib.Path, *, allow_absent: bool = False) -> tuple[str, bool]:
    """The fused/two-step parity table, and whether it passes.

    The two arms fit the same anchor cell through bonsai's one-call form
    and through the Dataset + train form the runner reports ingest_s and
    train_s from. Only their agreement makes that split honest: a
    two-step Dataset that lost its device hint bins on the host, which
    costs seconds and host memory, so peak RSS is banded alongside time.

    Parameters
    ----------
    path : pathlib.Path
        The pod's ``parity.jsonl``. A skipped-only file (no visible CUDA
        device on the measuring host) passes with a stated caveat: the
        check ran and declared itself inapplicable. A missing file is a
        different case, the check never ran at all, so it fails unless
        ``allow_absent`` says the operator is accepting the gap on
        purpose.
    allow_absent : bool, optional
        Treat a missing file as a pass with a caveat instead of a
        failure. False by default; ``supersede --no-parity`` sets it.

    Returns
    -------
    tuple[str, bool]
        The markdown table (or a one-line note) and the pass flag.
    """
    if not path.exists():
        if allow_absent:
            return ("Parity check absent (no parity.jsonl in this results "
                     "dir).", True)
        return ("Parity check FAILED: no parity.jsonl in this results "
                 "dir. Absence is not evidence the check does not apply, "
                 "it means the check never ran.", False)
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    live = [r for r in rows if not r.get("skipped")]
    if not live:
        return "Parity check skipped on this host (no visible CUDA device).", True
    cell = f"{live[0]['rows']}x{live[0]['cols']} {live[0]['grower']}"
    lines = [f"| metric ({cell}) | fused | two-step | delta |",
             "|---|--:|--:|--:|"]
    ok = True
    for metric, unit in (("fit_s", "s"), ("peak_rss_gb", "GB")):
        arms = {arm: [r[metric] for r in live
                      if r["arm"] == arm and r.get(metric) is not None]
                for arm in ("fused", "two_step")}
        if not arms["fused"] or not arms["two_step"]:
            lines.append(f"| {metric} | n/a | n/a | n/a |")
            continue
        f = statistics.median(arms["fused"])
        t = statistics.median(arms["two_step"])
        d = 100 * (t - f) / f
        moved = abs(d) > PARITY_BAND_PCT
        ok = ok and not moved
        lines.append(f"| {metric} | {f:.2f}{unit} | {t:.2f}{unit} | "
                     f"{d:+.1f}%{' **FAIL**' if moved else ''} |")
    split = [(r["ingest_s"], r["train_s"]) for r in live
             if r["arm"] == "two_step" and r.get("ingest_s") is not None]
    if split:
        lines.append("")
        lines.append(f"Two-step split: ingest "
                     f"{statistics.median(i for i, _ in split):.2f}s, train "
                     f"{statistics.median(t for _, t in split):.2f}s.")
    lines.append("")
    lines.append(f"Verdict: {'PASS' if ok else 'FAIL'} "
                 f"(band +-{PARITY_BAND_PCT}%).")
    return "\n".join(lines), ok


def _verdict(ab_path: pathlib.Path) -> str:
    """The A/B verdict table, or empty when the file is absent.

    Time and peak RSS both get a column and both can move the verdict:
    memory is a standings claim, and a path change can cost RSS before it
    costs seconds.
    """
    if not ab_path.exists():
        return ""
    rows = [json.loads(ln) for ln in ab_path.read_text().splitlines()
            if ln.strip()]
    med: dict[tuple, list[float]] = {}
    for r in rows:
        for metric in ("fit_s", "peak_rss_gb"):
            if r.get(metric) is None:
                continue
            med.setdefault((r["rows"], r["cols"], r["grower"], r["arm"],
                            metric), []).append(r[metric])
    lines = ["| cell | grower | old | new | delta | old RSS | new RSS | "
             "RSS delta |", "|---|---|--:|--:|--:|--:|--:|--:|"]
    for (rw, c, g) in sorted({(r["rows"], r["cols"], r["grower"])
                              for r in rows}):
        cells, moved = [], False
        for metric, unit in (("fit_s", "s"), ("peak_rss_gb", "GB")):
            old = med.get((rw, c, g, "old", metric))
            new = med.get((rw, c, g, "new", metric))
            if not old or not new:
                cells += ["n/a", "n/a", "n/a"]
                continue
            o, n = statistics.median(old), statistics.median(new)
            d = 100 * (n - o) / o
            moved = moved or abs(d) > 5
            cells += [f"{o:.2f}{unit}", f"{n:.2f}{unit}", f"{d:+.1f}%"]
        cells[-1] += " **moved**" if moved else ""
        lines.append(f"| {rw}x{c} | {g} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
