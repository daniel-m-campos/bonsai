"""The standings refresh, run locally (decision 96): measure on a rented
pod, then build the supersession PR. Replaces the retired standings-refresh
CI workflow, whose runner-side tail failed on every dispatch while this
exact ritual succeeded by hand.

    export RUNPOD_API_KEY=rpa_...   # never printed; runbook section 0
    python3 scripts/standings_refresh.py measure --prev-version 1.5.4
    python3 scripts/standings_refresh.py supersede --results-dir <dir>

Phases are independent: `measure` rents one L40S, runs
scripts/standings_refresh_pod.sh detached, polls, pulls the jsonl files
into a local directory, and tears the pod down (sweep + verify-empty).
`supersede` works from any results directory: registry update, staging,
render, A/B verdict, branch, and `gh pr create`. A failed supersede is
rerunnable without touching a pod; that separation is the whole point.

It also gates on the pod's fused/two-step parity rows before touching
anything: the published ingest/train split is only honest while bonsai's
two-step form still bins where its fused call does, so a parity failure
stops the supersession instead of shipping a breakdown that describes a
pipeline no cuda grower runs.
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
IMAGE = "ghcr.io/daniel-m-campos/bonsai-ci:cuda12.4"
GPU = "NVIDIA L40S"

# The band the fused and two-step forms must agree inside. Measured at 2.5%
# on one L40S at 4M x 512 and inside repeat noise at 16M; 5% matches the A/B
# band, which is the same pod-noise question.
PARITY_BAND_PCT = 5

AXIS_FILE = {"rows": "rebaseline", "width": "cols-rebaseline",
             "shape": "iso-volume", "frontier": "gpu-pareto-16M",
             "airline": "airline"}

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
    m.add_argument("--axes", default="rows,width,frontier,airline")
    m.add_argument("--prev-version", default="",
                   help="wheel for the A/B old arm (empty skips the A/B)")
    m.add_argument("--out-dir", default=None,
                   help="where the jsonl files land (default: a dated dir)")
    m.add_argument("--keep-pod", action="store_true",
                   help="skip teardown (debugging; delete it yourself)")
    s = sub.add_parser("supersede", help="build the supersession PR from results")
    s.add_argument("--results-dir", required=True)
    s.add_argument("--axes", default="rows,width,frontier,airline")
    s.add_argument("--no-pr", action="store_true",
                   help="stop after commit (inspect before pushing)")
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
        Parsed ``measure`` arguments: ``axes``, ``prev_version``, ``out_dir``,
        ``keep_pod``.

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
        subprocess.run([*ssh, f"nohup env AXES='{args.axes}' GIT_SHA='{sha}' "
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
        Parsed ``supersede`` arguments: ``results_dir``, ``axes``, ``no_pr``.

    Returns
    -------
    int
        0 on success, 1 if a required axis jsonl file is missing.
    """
    src = pathlib.Path(args.results_dir)
    axes = [a.strip() for a in args.axes.split(",")]
    parity, parity_ok = _parity(src / "parity.jsonl")
    print(parity)
    if not parity_ok:
        print("ERROR: fused/two-step parity failed; the ingest/train split "
              "in these rows is not trustworthy. Fix the runner's device "
              "hint and re-measure.", file=sys.stderr)
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
    for axis in axes:
        subprocess.run([sys.executable, "scripts/update_standings.py",
                        "--axis", axis, "--file", files[axis]],
                       check=True, cwd=REPO)
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


# Private Helpers ==================================================================================

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
    """Create the L40S with the runbook-mandated PUBLIC_KEY env."""
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


def _parity(path: pathlib.Path) -> tuple[str, bool]:
    """The fused/two-step parity table, and whether it passes.

    The two arms fit the same anchor cell through bonsai's one-call form
    and through the Dataset + train form the runner reports ingest_s and
    train_s from. Only their agreement makes that split honest: a
    two-step Dataset that lost its device hint bins on the host, which
    costs seconds and host memory, so peak RSS is banded alongside time.

    Parameters
    ----------
    path : pathlib.Path
        The pod's ``parity.jsonl``. A missing or skipped-only file passes
        with a stated caveat, so a refresh measured on a pod without the
        check still supersedes.

    Returns
    -------
    tuple[str, bool]
        The markdown table (or a one-line note) and the pass flag.
    """
    if not path.exists():
        return "Parity check absent (no parity.jsonl in this results dir).", True
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
