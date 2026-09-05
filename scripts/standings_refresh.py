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

The A/B is the perf-change detector, and both planes take it: the pod fits
the previous release wheel, the fixed anchor wheel and HEAD at anchor cells,
every arm once per rep with the reps interleaved, cuda growers on the GPU
session and the cpu growers on whichever host measures cpu-tall. The math
(min over reps, the two bands, what counts as moved) is
check_standings.py's, so the verdict printed here, the table the perf page
renders and the gate that demands a decision entry for a move read one
answer.

The host of record for the CPU plane is a GPU pod's CPU, at the 12 threads
the CPU specs pin (runbook section 11). That is server silicon rather than
the desktop-class parts a CPU rental buys, and a CPU ranking is a claim
about a class of machine as much as about an engine. So the default is one
rental: every axis rides the GPU session, and `--cpu-plane-host cpupod`
buys a separate CPU pod for the CPU axes when that is what is wanted.

The two hosts cap by different mechanisms, which is why the pod script
asserts whichever cap it landed under rather than one rule. A GPU pod
meters CPU bandwidth: the threads share one pool of core-seconds, bonsai's
spin-wait barriers spend on waiting whatever the ceiling withholds, and a
16-thread fit on a 13.6-core quota sat throttled in 97% of enforcement
periods (issue #355). What the specs' 12 threads buy under that quota is
one spare core, and the throttle counters bracketing each axis's first fit
measure directly what an older 1.5-cores-per-thread margin was guessing at.
That margin is retired: no rental reliably offers it. A CPU pod instead
hands the container a cpuset, whole CPUs equal to the purchase, so a
spinning thread burns only the core it already owns and one vCPU per thread
is enough; the sizing check below applies to that path only. Either way the
axis fails rather than publishing a number made under a ceiling nobody
declared, and every row carries the host block it was measured under.

It gates on the pod's fused/two-step parity rows before touching
anything: the published ingest/train split is only honest while bonsai's
two-step form still bins where its fused call does, so a parity failure
stops the supersession instead of shipping a breakdown that describes a
pipeline no cuda grower runs. A missing parity.jsonl fails the same gate:
absence means the check never ran (a lost scp, a pod that died before the
parity phase), not that it does not apply. The one exception is a session
whose axes do not include the one those rows anchor, since the pod does not
take them at all then. `supersede --no-parity` accepts a results dir with no
parity evidence for the hosts where the check truly cannot run.
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

sys.path.insert(0, str(REPO / "scripts"))
import check_standings  # noqa: E402

REST = "https://rest.runpod.io/v1"
# Per-datacenter stock is a v2 catalog reading; pods are still created on v1.
CATALOG = "https://api.runpod.io/v2/catalog/gpus"
# cuda12.8: the GPU below is sm_120, past 12.4's --offload-arch=native reach.
IMAGE = "ghcr.io/daniel-m-campos/bonsai-ci:cuda12.8"
# The create ladder, tried in order: the Workstation Edition is the same
# silicon (runbook), rented only once no Server-Edition instance exists
# anywhere. The rows tag themselves from nvidia-smi, never from this list.
GPUS = ("NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "NVIDIA RTX PRO 6000 Blackwell Workstation Edition")

# Never rented, whatever it reports in stock: excluded by policy, not supply.
BANNED_DATACENTERS = ("EUR-IS-2",)
# The stock levels worth trying, best first. A datacenter reporting NONE, or
# any level not on this list, is not a candidate.
STOCK_ORDER = ("HIGH", "MEDIUM", "LOW")

# The optional separate CPU rental (--cpu-plane-host cpupod), not the
# default host of record. cpu5g is the general-purpose CPU5 flavor at 4GB of
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
# Where the CPU axes are measured. "gpu" is the host of record: they ride
# the GPU session and no second pod is rented.
CPU_PLANE_HOSTS = (PLANE_GPU, "cpupod")

# The A/B bands and the anchor wheel are check_standings.py's: the gate,
# the renderer and this driver read one number.
AB_BAND_PCT = check_standings.AB_BAND_PCT
ANCHOR_BAND_PCT = check_standings.ANCHOR_BAND_PCT
ANCHOR_VERSION = check_standings.ANCHOR_VERSION

# The band the fused and two-step forms must agree inside. Measured at 2.5%
# on one L40S at 4M x 512 and inside repeat noise at 16M; the A/B band is
# the same pod-noise question, so the two share a number.
PARITY_BAND_PCT = AB_BAND_PCT

# One A/B file per plane in a results directory, so the two sessions
# cannot land on each other's rows.
AB_FILES = {PLANE_GPU: "ab-gpu.jsonl", PLANE_CPU: "ab-cpu.jsonl"}

# Every axis is the stem of its dated results file AND the name of the
# branch the pod script runs to produce it.
AXES = ("gpu-tall", "gpu-wide", "gpu-extreme", "cpu-tall", "cpu-wide",
        "gpu-early-stop", "gpu-shap", "quality-grinsztajn")

# Measured on this machine, not the pod: the code division reads the tree
# rather than running it, so a rental would only rent a checkout. Named here
# so the registry-agreement check below can tell "deliberately local" from
# "silently unreachable", which is the drift that let a release ship two axes
# the driver could not refresh (issue #433).
LOCAL_AXES = ("code",)

# The axis the parity rows anchor: same cell, same pod, same session.
PARITY_AXIS = "gpu-tall"

# Axis name prefix -> the plane whose pod measures it.
CPU_PREFIX = "cpu-"
SPECS = REPO / "python" / "bonsai" / "bench" / "specs"
# The pod script's loud half: written when the CPU-cap assertion fails or
# the throttle bracket catches material throttling, pulled back with results.
QUOTA_FAIL = "quota-fail.txt"

# How long `measure` waits for the pod's DONE marker before tearing it down,
# roughly twice the observed run: a nine-axis release refresh lands near three
# hours. Sized per axis because a one-axis re-measure should not inherit a
# nine-axis deadline. POLL_MAX_MISSES ends a wait on a pod that stopped
# answering at all, which a deadline alone would sit through (issue #423).
POLL_BASE_S = 45 * 60
POLL_PER_AXIS_S = 45 * 60
POLL_MAX_MISSES = 20

# The create ladder is walked until this deadline: multi-GPU stock churns on
# roughly ten-minute timers, and one refresh needed 14 refused creates across
# six regions before a draw succeeded about forty minutes in.
CREATE_DEADLINE_S = 45 * 60
CREATE_BACKOFF_S = 90

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
                   help="wheel for the A/B old arm (the anchor arm is "
                        f"always {ANCHOR_VERSION})")
    m.add_argument("--out-dir", default=None,
                   help="where the jsonl files land (default: a dated dir)")
    m.add_argument("--keep-pod", action="store_true",
                   help="skip teardown (debugging; delete it yourself)")
    m.add_argument("--only-stale", action="store_true",
                   help="drop the requested axes whose plane digest has not "
                   "moved since their last refresh (check_standings --stale)")
    m.add_argument("--cpu-plane-host", choices=CPU_PLANE_HOSTS,
                   default=PLANE_GPU,
                   help="where the cpu axes are measured: on the GPU pod's "
                        "own CPU (default, the host of record) or on a "
                        "separately rented CPU pod")
    m.add_argument("--cpu-vcpu", type=int, default=CPU_VCPU,
                   help="vCPUs to buy when --cpu-plane-host is cpupod; must "
                        "be at least one per thread the specs claim")
    m.add_argument("--gpu-type", default="",
                   help="rent exactly this GPU card instead of walking the "
                        "Server-then-Workstation Edition ladder")
    m.add_argument("--dry-run", action="store_true",
                   help="print the rental plan, the sizing arithmetic, and "
                        "the datacenters the GPU is in stock in, then exit "
                        "without renting anything")
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
    """Run one pod session per host: create, launch, poll, pull, tear down.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``measure`` arguments: ``axes``, ``only_stale``,
        ``prev_version``, ``out_dir``, ``keep_pod``, ``cpu_plane_host``,
        ``cpu_vcpu``, ``dry_run``.

    Returns
    -------
    int
        0 on success, 1 if the requested CPU rental is too small for the
        specs, ``RUNPOD_API_KEY`` is not set, a pod's gate failed an axis,
        or a pod finished without delivering rows for a requested axis.
    """
    if drift := registry_drift():
        print(drift, file=sys.stderr)
        return 1
    axes = _requested_axes(args.axes)
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
    cpu_axes = [a for a in axes if a.startswith(CPU_PREFIX)]
    if args.cpu_plane_host == PLANE_GPU:
        # The host of record: the cpu axes ride the GPU session, so one
        # rental measures the whole matrix and the pod script's cap
        # assertion is the bandwidth-quota branch.
        sessions = [(PLANE_GPU, axes)]
        if cpu_axes:
            print(f"cpu plane: {', '.join(cpu_axes)} on the GPU pod's own "
                  f"CPU at {max(spec_threads(a) for a in cpu_axes)}t "
                  "(runbook section 11); no second rental")
    else:
        sessions = [(PLANE_GPU, [a for a in axes if a not in cpu_axes]),
                    (PLANE_CPU, cpu_axes)]
        sessions = [(p, sax) for p, sax in sessions if sax]
        if cpu_axes:
            # One vCPU per thread is the whole sizing rule for this path: a
            # rented CPU pod enforces the purchase as a cpuset, so a thread
            # that spins at a barrier burns only the core it already owns.
            needed = max(spec_threads(axis) for axis in cpu_axes)
            print(f"cpu plane: {', '.join(cpu_axes)} at {needed}t need >= "
                  f"{needed} vCPU (one per thread, because a cpu pod caps by "
                  f"cpuset); renting {args.cpu_vcpu} x {CPU_FLAVOR}")
            if args.cpu_vcpu < needed:
                print(f"ERROR: --cpu-vcpu {args.cpu_vcpu} is below the "
                      f"{needed} the sizing rule requires; the axis would "
                      f"run more threads than the cpuset has cpus",
                      file=sys.stderr)
                return 1
    key = os.environ.get("RUNPOD_API_KEY")
    gpus = (args.gpu_type,) if args.gpu_type else GPUS
    if args.dry_run:
        _dry_run_datacenters(key, sessions, gpus)
        print("--dry-run: nothing rented")
        return 0
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
    for plane, session_axes in sessions:
        _run_session(key, args, plane=plane, axes=session_axes,
                     out_dir=out_dir, sha=sha, pubkey=pubkey, gpus=gpus)
    quota_fail = out_dir / QUOTA_FAIL
    if quota_fail.exists():
        print("ERROR: a pod's gate failed an axis; its rows were renamed "
              "QUOTAFAIL-* and must not be superseded:\n"
              + quota_fail.read_text().rstrip(), file=sys.stderr)
    # DONE only means the pod script ran to its last line; an axis its RAM
    # guard skipped ends there with no rows and no failure count, so the
    # delivery is verified file by file rather than trusted from the marker.
    missing = [a for a in axes
               if not any(out_dir.glob(f"{a}-*.jsonl"))
               and not any(out_dir.glob(f"QUOTAFAIL-{a}-*.jsonl"))]
    if missing:
        print("ERROR: the pod reported done but delivered no rows for "
              f"{', '.join(missing)}; a pod-side SKIP is not a measurement. "
              "Re-run these axes on a host that can take them.",
              file=sys.stderr)
    if quota_fail.exists() or missing:
        return 1
    print(f"results in {out_dir}/; next:\n"
          f"  python3 scripts/standings_refresh.py supersede "
          f"--results-dir {out_dir}")
    return 0


def spec_threads(axis: str) -> int:
    """The thread count an axis's bundled spec publishes its claim at."""
    spec = json.loads((SPECS / f"{axis}.json").read_text())
    return max(spec.get("threads", [16]))


def stocked_datacenters(gpu_type: str, key: str) -> list[str]:
    """The datacenters holding one GPU type, best stock first.

    A create that names no datacenter lands wherever RunPod places it, and
    that region can be empty while the same GPU sits in stock two regions
    over: the create then fails 500 "no instances" against a fleet that has
    the machine. So the GPU create body pins one datacenter read from here.
    EUR-IS-2 is off the list by policy rather than by stock.

    Parameters
    ----------
    gpu_type : str
        The ``gpuTypeIds`` value the create body asks for.
    key : str
        The RunPod API key. Never printed, here or on the failure path.

    Returns
    -------
    list[str]
        Datacenter ids, HIGH before MEDIUM before LOW. Empty when the
        lookup fails or the fleet reports no stock anywhere, which leaves
        the caller with the unpinned create it would have issued anyway.
    """
    try:
        # product=POD is mandatory with include=AVAILABILITY, and Cloudflare
        # rejects urllib's default User-Agent with a bare 403 (error 1010).
        out = _api(f"{CATALOG}?include=AVAILABILITY&cloud=SECURE&product=POD",
                   key, method="GET")
    except OSError as e:
        print(f"WARNING: availability lookup failed ({e}); the create goes "
              "unpinned", file=sys.stderr)
        return []
    gpu = next((g for g in out.get("gpus", []) if g.get("id") == gpu_type), {})
    ranked = sorted((STOCK_ORDER.index(dc["availability"]), dc["id"])
                    for dc in gpu.get("dataCenters", [])
                    if dc.get("availability") in STOCK_ORDER
                    and dc.get("id") not in BANNED_DATACENTERS)
    return [dc for _, dc in ranked]


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
    axes = _requested_axes(args.axes)
    parity = _cleared_parity(src / "parity.jsonl", axes,
                             no_parity=args.no_parity)
    if parity is None:
        return 1
    files = _stage_axis_files(src, axes)
    if files is None:
        return 1
    _restamp_and_render(axes, files, _copy_parity(src / "parity.jsonl",
                                                  files.get(PARITY_AXIS)))
    verdict = _ab_verdicts(src)
    print(verdict or "A/B skipped (no ab-*.jsonl)")

    hosts_note = _cpu_hosts_note(axes, files)
    branch = _commit_refresh(axes, hosts_note)
    if args.no_pr:
        print(f"committed on {branch}; push and open the PR when ready")
        return 0
    _open_refresh_pr(axes, branch, parity, verdict, hosts_note)
    return 0


def registry_drift() -> str:
    """Why this driver cannot service the registry, or "" when it can.

    The registry decides what a release gate demands; this module decides
    what a rental measures. When the two disagree the failure is silent in
    the worst direction: `measure` refreshes a subset and reports success,
    and the missing axes are noticed only if someone reads the registry.
    1.15.0 shipped that way, with `gpu-shap` and `quality-grinsztajn`
    reachable by no branch of the pod script (issue #433).

    Returns
    -------
    str
        An error naming both directions of the disagreement, or "".
    """
    known = set(AXES) | set(LOCAL_AXES)
    # Read rather than import: scripts/ is not a package, and check_standings
    # is already reached by subprocess everywhere else in this module.
    registry = json.loads((REPO / "benchmarks" / "standings.json").read_text())
    registry.pop("_", None)  # the schema comment key, as load_registry drops it
    registered = set(registry)
    if known == registered:
        return ""
    lines = ["ERROR: the driver and benchmarks/standings.json disagree "
             "about which axes exist"]
    if missing := sorted(registered - known):
        lines.append(f"  in the registry, measured by nothing: "
                     f"{', '.join(missing)}")
    if extra := sorted(known - registered):
        lines.append(f"  measured here, in no registry entry: "
                     f"{', '.join(extra)}")
    lines.append("  add the branch to standings_refresh_pod.sh and the name "
                 "to AXES, or to LOCAL_AXES if it is measured off-pod")
    return "\n".join(lines)


def stale_axes() -> set[str]:
    """The axes `check_standings.py --stale` says a refresh must measure."""
    out = subprocess.run([sys.executable, "scripts/check_standings.py",
                          "--stale"], capture_output=True, text=True,
                         check=True, cwd=REPO)
    return {line.strip() for line in out.stdout.splitlines() if line.strip()}


# Private Helpers ==================================================================================

def _refresh_title(axes: list[str]) -> str:
    """The commit and PR title; the axis list appears only while it fits.

    The commit hook hard-caps titles at 72 characters, which a full
    six-axis list exceeds, so past the 50-character target the title
    carries the count and the body carries the list.
    """
    title = f"bench(standings): refresh {','.join(axes)}"
    if len(title) <= 50:
        return title
    return f"bench(standings): refresh {len(axes)} axes"


def _dry_run_datacenters(key: str | None, sessions: list[tuple[str, list[str]]],
                         gpus: tuple[str, ...]):
    """Print the datacenters each GPU create would try, renting nothing.

    This is the only way to exercise the stock reading without buying a
    pod, so it hits the live catalog rather than reporting a plan.
    """
    if not any(plane == PLANE_GPU for plane, _ in sessions):
        return
    if not key:
        print("--dry-run: RUNPOD_API_KEY unset, skipping the stock reading")
        return
    for gpu in gpus:
        stocked = stocked_datacenters(gpu, key)
        print(f"{gpu}: in stock, best first: "
              f"{', '.join(stocked) or 'none; the create would go unpinned'}")


def _row_host(path: pathlib.Path) -> str:
    """The host name the rows in one results file were measured under."""
    with path.open() as fh:
        for line in fh:
            if line.strip():
                return json.loads(line).get("host", {}).get("name", "")
    return ""


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


def _requested_axes(spec: str) -> list[str]:
    """The axis names in a comma-separated --axes value, blanks dropped."""
    return [a.strip() for a in spec.split(",") if a.strip()]


def _cleared_parity(path: pathlib.Path, axes: list[str], *,
                    no_parity: bool) -> str | None:
    """The parity table these results carry, or None when the gate refuses."""
    # The pod takes parity rows only when the session measures the axis they
    # anchor, so for any other session absence is the expected state rather
    # than a lost file. A parity.jsonl that is there anyway is still read.
    anchored = PARITY_AXIS in axes
    if not anchored:
        print("Parity not expected: these axes do not include "
              f"{PARITY_AXIS}, which the parity rows anchor.")
    table, ok = _parity(path, allow_absent=no_parity or not anchored)
    print(table)
    if ok:
        return table
    if not path.exists():
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
    return None


def _stage_axis_files(src: pathlib.Path, axes: list[str]) -> dict | None:
    """Copy each axis's newest results file in, or None if one never arrived."""
    files = {}
    for axis in axes:
        got = sorted(src.glob(f"{axis}-*.jsonl"))
        if not got:
            print(f"ERROR: no {axis}-*.jsonl in {src}", file=sys.stderr)
            return None
        files[axis] = got[-1].name
        shutil.copy2(got[-1], RESULTS / got[-1].name)
    return files


def _restamp_and_render(axes: list[str], files: dict,
                        companion: str | None) -> None:
    """Move every axis's stamp onto its new file, then regenerate the pages."""
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


def _cpu_hosts_note(axes: list[str], files: dict) -> str:
    """The sentence naming which machine measured the cpu axes, or "".

    Every row carries its own host block; this is where that reaches a
    reader of the PR, since a CPU number means nothing without the ceiling
    it was made under.
    """
    cpu_axes = [a for a in axes if a.startswith(CPU_PREFIX)]
    if not cpu_axes:
        return ""
    cpu_hosts = sorted({_row_host(RESULTS / files[a]) for a in cpu_axes})
    return (f"\n\nCPU axes ({', '.join(cpu_axes)}) were measured on "
            f"{', '.join(h for h in cpu_hosts if h)} (issue #355). "
            "Every row carries its own host block, so the registry "
            "records which machine and which ceiling stands behind "
            "each axis.")


def _commit_refresh(axes: list[str], hosts_note: str) -> str:
    """Branch, stage the regenerated pages, commit; returns the branch name."""
    branch = f"standings-refresh-{time.strftime('%Y%m%d')}"
    subprocess.run(["git", "checkout", "-b", branch], check=True, cwd=REPO)
    subprocess.run(["git", "add", "-A", "benchmarks/", "docs/method/",
                    "README.md"], check=True, cwd=REPO)
    subprocess.run(["git", "commit", "-m",
                    f"{_refresh_title(axes)}\n\n"
                    f"Axes: {','.join(axes)}. Same-pod refresh via "
                    "scripts/standings_refresh.py "
                    "(decision 96); superseded files deleted, registry "
                    "updated, ledger and README regenerated." + hosts_note],
                   check=True, cwd=REPO)
    return branch


def _open_refresh_pr(axes: list[str], branch: str, parity: str, verdict: str,
                     hosts_note: str) -> None:
    """Push the branch and open the supersession PR with both gate tables."""
    subprocess.run(["git", "push", "-u", "origin", branch], check=True, cwd=REPO)
    body = (f"Standings refresh of {','.join(axes)} via "
            f"`scripts/standings_refresh.py` "
            f"(decision 96).{hosts_note}\n\nIngest/train parity (bonsai's "
            f"fused call vs the two-step Dataset form, same pod, interleaved, "
            f"+-{PARITY_BAND_PCT}% band):\n\n{parity}\n\n"
            f"A/B verdict (previous release wheel and the {ANCHOR_VERSION} "
            f"anchor vs HEAD, same pod, interleaved, min of reps, "
            f"+-{AB_BAND_PCT}% band vs old, +-{ANCHOR_BAND_PCT}% vs the "
            f"anchor):\n\n{verdict or 'A/B skipped.'}\n\nA **moved** "
            f"verdict requires a `Standings:`-tagged decision entry before "
            f"merge; the docs-check gate enforces it.")
    subprocess.run(["gh", "pr", "create", "--base", "main", "--title",
                    _refresh_title(axes), "--body", body], check=True,
                   cwd=REPO)


def _run_session(key: str, args: argparse.Namespace, *, plane: str,
                 axes: list[str], out_dir: pathlib.Path, sha: str,
                 pubkey: str, gpus: tuple[str, ...]):
    """One pod, one plane: create, launch the on-pod script, poll, tear down.

    Both planes write into the same results directory. Their file names do
    not collide (one dated file per axis, one A/B file per plane). The
    parity rows are narrower: the pod takes them only when this session
    measures the axis they anchor, so they cannot arrive from a host that
    measured no anchor.
    """
    pod_id = _create_pod(key, pubkey, plane=plane, vcpu=args.cpu_vcpu,
                         gpus=gpus)
    print(f"{plane} pod {pod_id} created for {', '.join(axes)}; waiting for ssh")
    # No HOST_TAG on either plane: the driver knows what it asked for, the
    # pod knows what it got, and naming a row after the request is how a
    # 12-thread run ended up committed under a 16-thread tag.
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
                        f"PREV_VERSION='{args.prev_version}' "
                        f"ANCHOR_VERSION='{ANCHOR_VERSION}' "
                        f"PLANE='{plane}' "
                        "bash /root/standings_refresh_pod.sh "
                        "> /root/refresh.log 2>&1 & echo launched"], check=True)
        _poll_pod_run(ssh, out_dir, ip, port, axes)
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
                 "User-Agent": "bonsai-standings-refresh",
                 "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read().decode()
    return json.loads(body) if body.strip() else {}


def _create_pod(key: str, pubkey: str, *, plane: str, vcpu: int,
                gpus: tuple[str, ...]) -> str:
    """Create a standings pod for one plane, with the mandated PUBLIC_KEY env.

    A GPU pod is bought by device, and its CPU share is whatever the host
    has spare, read off the container rather than assumed. A CPU pod is
    bought by vCPU, so its ceiling is a line on the invoice; that is the
    ``--cpu-plane-host cpupod`` path, not the host of record.

    The GPU create walks the cards in ``gpus`` in order, one create per
    stocked datacenter then unpinned, so a later card is tried only once
    the one before it is exhausted everywhere. The CPU create is unpinned:
    a cpu flavor is not a device with a per-region stock reading.
    """
    name = f"bonsai-standings-{plane}-{time.strftime('%Y%m%d-%H%M')}"
    body = {"name": name, "imageName": IMAGE, "cloudType": "SECURE",
            "ports": ["22/tcp"], "env": {"PUBLIC_KEY": pubkey}}
    if plane == PLANE_CPU:
        body |= {"computeType": "CPU", "cpuFlavorIds": [CPU_FLAVOR],
                 "vcpuCount": vcpu, "containerDiskInGb": CPU_DISK_GB}
        bodies = [body]
    else:
        # One datacenter per attempt: a multi-entry dataCenterIds list can
        # 400 on the create schema. Per card, the unpinned body stays last
        # so an empty or failed stock reading is never worse than not
        # pinning.
        bodies = []
        for gpu in gpus:
            card = body | {"gpuTypeIds": [gpu], "gpuCount": 1,
                           "containerDiskInGb": GPU_DISK_GB}
            bodies += [card | {"dataCenterIds": [dc]}
                       for dc in stocked_datacenters(gpu, key)]
            bodies.append(card)
    deadline = time.time() + CREATE_DEADLINE_S
    attempt = 0
    while True:
        attempt += 1
        for candidate in bodies:
            where = ",".join(candidate.get("dataCenterIds", ["unpinned"]))
            hardware = (candidate.get("gpuTypeIds") or [CPU_FLAVOR])[0]
            try:
                out = _api(f"{REST}/pods", key, candidate)
                if out.get("id"):
                    return out["id"]
            except OSError as e:
                # 500 is that datacenter out of instances; 400 is a
                # datacenter the stock reading lists but the create
                # schema's enum rejects (US-MO-2 and US-NC-2 were). Both
                # mean try the next candidate, neither is fatal.
                print(f"create attempt {attempt} ({hardware}, {where}): {e}",
                      file=sys.stderr)
        if time.time() > deadline:
            raise SystemExit(
                f"no usable {plane} pod after {attempt} passes over "
                f"{len(bodies)} candidates in "
                f"{CREATE_DEADLINE_S // 60} minutes")
        print(f"  every candidate refused; retrying in "
              f"{CREATE_BACKOFF_S}s (stock churns)", flush=True)
        time.sleep(CREATE_BACKOFF_S)


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


def _poll_pod_run(ssh: list[str], out_dir: pathlib.Path, ip: str, port: int,
                  axes: list[str]):
    """Poll the detached run; pull the session directory incrementally.

    Pulling every poll (not just at the end) means a pod that dies late
    still leaves the finished axes on this machine. The whole directory
    comes over, not just *.jsonl, because the quota gate's marker file is
    the evidence that some of those axes must not be published.

    Two limits end the wait, because the pod script prints DONE on every
    exit path including its aborts: absence past a deadline means the pod is
    not coming back, and the `finally` above deletes it. Without them a pod
    that lost its network billed until someone looked (issue #423). Both
    raise, so partial results still reach this machine, since every poll
    already pulled them.

    Parameters
    ----------
    axes : list[str]
        What this session measures, which sizes the deadline: a one-axis
        sweep and a nine-axis release refresh are hours apart, and a cap
        sized for the second is no cap at all for the first.
    """
    scp_base = ["scp", "-i", str(pathlib.Path.home() / ".ssh" / "id_ed25519"),
                *SSH_OPTS, "-P", str(port)]
    deadline = time.time() + POLL_BASE_S + POLL_PER_AXIS_S * max(1, len(axes))
    misses = 0
    while True:
        time.sleep(120)
        subprocess.run([*scp_base, f"root@{ip}:/root/standings/*",
                        str(out_dir) + "/"], capture_output=True)
        tail = subprocess.run([*ssh, "tail -2 /root/refresh.log"],
                              capture_output=True, text=True)
        left = (deadline - time.time()) / 60
        last = tail.stdout.strip().splitlines()[-1:] or [""]
        print(f"  pod ({left:.0f}m left): {last[0][:100]}", flush=True)
        if "STANDINGS_REFRESH_DONE" in tail.stdout:
            subprocess.run([*scp_base, f"root@{ip}:/root/standings/*",
                            str(out_dir) + "/"], check=True)
            return
        misses = misses + 1 if tail.returncode != 0 else 0
        if tail.returncode != 0:
            print(f"  ssh poll failed ({misses}/{POLL_MAX_MISSES})", flush=True)
        if misses >= POLL_MAX_MISSES:
            raise SystemExit(
                f"pod unreachable for {misses} consecutive polls; tearing it "
                "down. Whatever it had measured is in the results directory.")
        if time.time() > deadline:
            raise SystemExit(
                f"no STANDINGS_REFRESH_DONE within the deadline for "
                f"{len(axes)} axes; tearing the pod down. Whatever it had "
                "measured is in the results directory.")


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
        return _absent_parity(allow_absent)
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    live = [r for r in rows if not r.get("skipped")]
    if not live:
        return "Parity check skipped on this host (no visible CUDA device).", True
    cell = f"{live[0]['rows']}x{live[0]['cols']} {live[0]['grower']}"
    lines = [f"| metric ({cell}) | fused | two-step | delta |",
             "|---|--:|--:|--:|"]
    ok = True
    for metric, unit in (("fit_s", "s"), ("peak_rss_gb", "GB")):
        row, moved = _parity_metric_row(live, metric, unit)
        lines.append(row)
        ok = ok and not moved
    if split := _two_step_split(live):
        lines += ["", split]
    lines += ["", f"Verdict: {'PASS' if ok else 'FAIL'} "
                  f"(band +-{PARITY_BAND_PCT}%)."]
    return "\n".join(lines), ok


def _absent_parity(allow_absent: bool) -> tuple[str, bool]:
    """What a missing parity.jsonl reads as, deliberately accepted or not."""
    if allow_absent:
        return ("Parity check absent (no parity.jsonl in this results "
                "dir).", True)
    return ("Parity check FAILED: no parity.jsonl in this results "
            "dir. Absence is not evidence the check does not apply, "
            "it means the check never ran.", False)


def _parity_metric_row(live: list[dict], metric: str,
                unit: str) -> tuple[str, bool]:
    """One metric's fused-vs-two-step row, and whether it left the band."""
    arms = {arm: [r[metric] for r in live
                  if r["arm"] == arm and r.get(metric) is not None]
            for arm in ("fused", "two_step")}
    if not arms["fused"] or not arms["two_step"]:
        return f"| {metric} | n/a | n/a | n/a |", False
    f = statistics.median(arms["fused"])
    t = statistics.median(arms["two_step"])
    d = 100 * (t - f) / f
    moved = abs(d) > PARITY_BAND_PCT
    return (f"| {metric} | {f:.2f}{unit} | {t:.2f}{unit} | "
            f"{d:+.1f}%{' **FAIL**' if moved else ''} |"), moved


def _two_step_split(live: list[dict]) -> str:
    """The ingest/train medians the perf page publishes, or "" if unreported."""
    split = [(r["ingest_s"], r["train_s"]) for r in live
             if r["arm"] == "two_step" and r.get("ingest_s") is not None]
    if not split:
        return ""
    return (f"Two-step split: ingest "
            f"{statistics.median(i for i, _ in split):.2f}s, train "
            f"{statistics.median(t for _, t in split):.2f}s.")


def _ab_verdicts(src: pathlib.Path) -> str:
    """Every plane's A/B verdict table found in ``src``, or empty."""
    return "\n\n".join(
        f"{plane} plane ({name}; {_arm_versions(src / name)}):\n\n"
        f"{_verdict(src / name)}"
        for plane, name in AB_FILES.items() if (src / name).exists())


def _verdict(ab_path: pathlib.Path) -> str:
    """The A/B verdict table, or empty when the file is absent.

    Time and peak RSS both get columns and both can move the verdict:
    memory is a standings claim, and a path change can cost RSS before it
    costs seconds.
    """
    if not ab_path.exists():
        return ""
    return check_standings.ab_table(_ab_rows(ab_path))


def _arm_versions(ab_path: pathlib.Path) -> str:
    versions = check_standings.ab_versions(_ab_rows(ab_path))
    return ", ".join(f"{arm} {versions[arm]}" for arm in check_standings.AB_ARMS
                     if arm in versions)


def _ab_rows(ab_path: pathlib.Path) -> list[dict]:
    return [json.loads(ln) for ln in ab_path.read_text().splitlines()
            if ln.strip()]


if __name__ == "__main__":
    sys.exit(main())
