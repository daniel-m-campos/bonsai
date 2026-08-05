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
    """Run one pod session: create, launch the on-pod script, poll, pull.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed ``measure`` arguments: ``axes``, ``only_stale``,
        ``prev_version``, ``out_dir``, ``keep_pod``.

    Returns
    -------
    int
        0 on success, 1 if ``RUNPOD_API_KEY`` is not set.
    """
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        print("ERROR: export RUNPOD_API_KEY first (runbook section 0)",
              file=sys.stderr)
        return 1
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
    out_dir = pathlib.Path(args.out_dir or
                           f"standings-{time.strftime('%Y%m%d-%H%M')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True, cwd=REPO).stdout.strip()
    pubkey = (pathlib.Path.home() / ".ssh" / "id_ed25519.pub").read_text().strip()

    pod_id = _create_pod(key, pubkey)
    print(f"pod {pod_id} created; waiting for ssh")
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
        axes_arg = ",".join(axes)
        subprocess.run([*ssh, f"nohup env AXES='{axes_arg}' "
                        f"GIT_SHA='{sha}' "
                        f"PREV_VERSION='{args.prev_version}' "
                        "bash /root/standings_refresh_pod.sh "
                        "> /root/refresh.log 2>&1 & echo launched"], check=True)
        _poll_pod_run(ssh, out_dir, ip, port)
    finally:
        if args.keep_pod:
            print(f"pod {pod_id} KEPT per --keep-pod; delete it yourself")
        else:
            _delete_pod(key, pod_id)
            _sweep(key)
    print(f"results in {out_dir}/; next:\n"
          f"  python3 scripts/standings_refresh.py supersede "
          f"--results-dir {out_dir}")
    return 0


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

    branch = f"standings-refresh-{time.strftime('%Y%m%d')}"
    subprocess.run(["git", "checkout", "-b", branch], check=True, cwd=REPO)
    subprocess.run(["git", "add", "-A", "benchmarks/", "docs/method/",
                    "README.md"], check=True, cwd=REPO)
    axes_label = ",".join(axes)
    subprocess.run(["git", "commit", "-m",
                    f"bench(standings): refresh {axes_label}\n\n"
                    "Same-pod refresh via scripts/standings_refresh.py "
                    "(decision 96); superseded files deleted, registry "
                    "updated, ledger and README regenerated."],
                   check=True, cwd=REPO)
    if args.no_pr:
        print(f"committed on {branch}; push and open the PR when ready")
        return 0
    subprocess.run(["git", "push", "-u", "origin", branch], check=True, cwd=REPO)
    body = (f"Standings refresh via `scripts/standings_refresh.py` "
            f"(decision 96).\n\nIngest/train parity (bonsai's fused call vs "
            f"the two-step Dataset form, same pod, interleaved, "
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


def _create_pod(key: str, pubkey: str) -> str:
    """Create the standings pod with the runbook-mandated PUBLIC_KEY env."""
    name = f"bonsai-standings-{time.strftime('%Y%m%d-%H%M')}"
    for attempt in (1, 2):
        try:
            out = _api(f"{REST}/pods", key, {
                "name": name, "imageName": IMAGE, "gpuTypeIds": [GPU],
                "gpuCount": 1, "cloudType": "SECURE", "containerDiskInGb": 80,
                "ports": ["22/tcp"], "env": {"PUBLIC_KEY": pubkey}})
            if out.get("id"):
                return out["id"]
        except OSError as e:
            print(f"create attempt {attempt}: {e}", file=sys.stderr)
        time.sleep(15)
    raise SystemExit("no usable pod after 2 attempts")


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
        return (ip, int(port)) if ip and port else None

    ip, port = _wait_until(mapping, timeout_s=360, what="pod port mapping")

    probe = ["ssh", "-i", str(pathlib.Path.home() / ".ssh" / "id_ed25519"),
             *SSH_OPTS, "-p", str(port), f"root@{ip}", "true"]
    def sshd():
        return subprocess.run(probe, capture_output=True).returncode == 0

    _wait_until(sshd, timeout_s=600, what="sshd")
    return ip, port


def _poll_pod_run(ssh: list[str], out_dir: pathlib.Path, ip: str, port: int):
    """Poll the detached run; pull the jsonl files incrementally.

    Pulling every poll (not just at the end) means a pod that dies late
    still leaves the finished axes on this machine.
    """
    scp_base = ["scp", "-i", str(pathlib.Path.home() / ".ssh" / "id_ed25519"),
                *SSH_OPTS, "-P", str(port)]
    while True:
        time.sleep(120)
        subprocess.run([*scp_base, f"root@{ip}:/root/standings/*.jsonl",
                        str(out_dir) + "/"], capture_output=True)
        tail = subprocess.run([*ssh, "tail -2 /root/refresh.log"],
                              capture_output=True, text=True)
        last = tail.stdout.strip().splitlines()[-1:] or [""]
        print(f"  pod: {last[0][:110]}", flush=True)
        if "STANDINGS_REFRESH_DONE" in tail.stdout:
            subprocess.run([*scp_base, f"root@{ip}:/root/standings/*.jsonl",
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
