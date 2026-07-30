"""Parent-side orchestration for benchmark jobs (spec-driven and legacy).

One child process per (cell, variant, repeat) via the worker protocol in
bonsai.bench.runners: OOM and segfaults become jsonl status rows instead of
killing the sweep. Every row goes through runlog.emit_row (schema v1) with an
optional run label, so ad-hoc campaigns stop inventing schemas.
"""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import re
import subprocess
import sys
import threading

from . import runlog
from .variants import resolve

# GPU variants skip the widest cells by default: 16k+ cols exhausts consumer
# VRAM and kernel grids; the per-host VRAM estimator handles the rest. Both
# gates are policy (a spec can disable them when OOM itself is the datum).
GPU_MAX_COLS = 16_384

PROFILE_RE = re.compile(r"(\w+)=([\d.]+)s")


def parse_profiles(stderr: str) -> dict:
    prof = {}
    for line in stderr.splitlines():
        if line.startswith(("cuda-profile:", "grow-profile:", "ingest-profile:",
                            "fit-profile:", "cuda-upload-decomp:")):
            prefix = line.split(":", 1)[0].removesuffix("-profile")
            for key, val in PROFILE_RE.findall(line):
                prof[f"{prefix}_{key}"] = float(val)
    return prof


def est_host_gb(rows: int, cols: int, n_test: int, lib: str) -> float:
    factor = 4.0 if lib == "catboost" else 3.0
    return (rows + n_test) * cols * 6 * factor / 2**30


def est_dev_gb(rows: int, cols: int) -> float:
    return (rows * cols * 2 + rows * 8 + 512 * 2**20) / 2**30


def timeout_for(cell: dict) -> int:
    cells = cell["rows"] * cell["cols"]
    return int(min(3600, max(900, 90 * cells / 1e8)))


def classify_error(message: str) -> str:
    low = message.lower()
    if any(s in low for s in ("out of memory", "memoryerror", "bad_alloc",
                              "cannot allocate", "oom")):
        return "oom"
    if any(s in low for s in ("unsupported", "max_bin", "border_count",
                              "invalid parameter", "must be")):
        return "unsupported"
    return "error"


def _nvml_query():
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def query(pid: int):
        pid_mb = None
        with contextlib.suppress(pynvml.NVMLError):
            for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                if p.pid == pid and p.usedGpuMemory is not None:
                    pid_mb = p.usedGpuMemory / 2**20
        total_mb = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 2**20
        return pid_mb, total_mb

    return query


def _smi_query(pid: int):
    try:
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        total = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if apps.returncode != 0 or total.returncode != 0:
        return None
    pid_mb = None
    for line in apps.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] == str(pid) and parts[1].isdigit():
            pid_mb = (pid_mb or 0) + int(parts[1])
    head = total.stdout.splitlines()[0].strip() if total.stdout.strip() else ""
    return pid_mb, (int(head) if head.isdigit() else None)


class DeviceMemSampler:
    """Samples device memory while the worker child runs; max over samples.

    Records BOTH the child's per-process usage and the whole-device number
    (allocator slack and context overhead lag the per-pid counter), plus the
    sampling interval, so the row never claims more precision than sampled.
    """

    def __init__(self, pid: int, interval_s: float = 0.25, query=None):
        self.pid, self.interval_s = pid, interval_s
        if query is not None:
            self._query, self.source = query, "injected"
        else:
            try:
                self._query, self.source = _nvml_query(), "nvml"
            except Exception:
                self._query, self.source = _smi_query, "nvidia-smi"
        self._stop = threading.Event()
        self._peak_pid: float | None = None
        self._peak_total: float | None = None
        self._samples = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            got = self._query(self.pid)
            if got is not None:
                pid_mb, total_mb = got
                if pid_mb is not None:
                    self._peak_pid = max(self._peak_pid or 0.0, pid_mb)
                if total_mb is not None:
                    self._peak_total = max(self._peak_total or 0.0, total_mb)
                self._samples += 1
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)

    def result(self) -> dict | None:
        if not self._samples:
            return None
        gb = 1024.0
        return {"peak_gb_pid": (round(self._peak_pid / gb, 2)
                                if self._peak_pid is not None else None),
                "peak_gb_total": (round(self._peak_total / gb, 2)
                                  if self._peak_total is not None else None),
                "samples": self._samples, "interval_s": self.interval_s,
                "source": self.source}


def run_one(spec: dict, timeout: int, sampler: bool = False,
            data_cache: str | None = None) -> dict:
    env = dict(os.environ)
    if spec["variant"].startswith("bonsai"):
        env.update(BONSAI_GROW_PROFILE="1", BONSAI_INGEST_PROFILE="1",
                   BONSAI_CUDA_PROFILE="1", BONSAI_FIT_PROFILE="1")
    if data_cache:
        env["BONSAI_BENCH_DATA_CACHE"] = data_cache
    proc = subprocess.Popen([sys.executable, "-m", "bonsai.bench", "worker"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, env=env)
    sm = DeviceMemSampler(proc.pid) if sampler else None
    try:
        with sm or contextlib.nullcontext():
            stdout, stderr = proc.communicate(input=json.dumps(spec),
                                              timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        out = {"status": "timeout", "message": f"exceeded {timeout}s"}
        if sm:
            out["dev_mem"] = sm.result()
        return out
    dev_mem = sm.result() if sm else None
    result_line = next((ln for ln in stdout.splitlines()
                        if ln.startswith("RESULT ")), None)
    if proc.returncode != 0 or result_line is None:
        if proc.returncode < 0:
            out = {"status": "oom",
                   "message": f"killed by signal {-proc.returncode}"}
        else:
            tail = (stderr.strip().splitlines() or ["no output"])[-1][:300]
            out = {"status": classify_error(tail), "message": tail}
        if sm:
            out["dev_mem"] = dev_mem
        return out
    out = json.loads(result_line.removeprefix("RESULT "))
    out["status"] = "ok"
    out["message"] = None
    prof = parse_profiles(stderr)
    out["profile"] = prof or None
    if sm:
        out["dev_mem"] = dev_mem
    return out


def resume_keys(path: str | pathlib.Path) -> set[tuple]:
    """Keys already measured (pods die: funds, spot reaps). ok/unsupported are
    final; skipped/oom/timeout/error re-attempt — the new host may differ."""
    done = set()
    for line in pathlib.Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("status") in ("ok", "unsupported"):
            c = r["cell"]
            done.add((r["variant"], r["threads"], r["repeat"], c["rows"],
                      c["cols"], c["bins"], c.get("depth"), c.get("iters"),
                      c.get("seed")))
    return done


def _job_key(job: dict, repeat: int) -> tuple:
    c = job["cell"]
    return (job["variant"], job["threads"], repeat, c["rows"], c["cols"],
            c["bins"], c.get("depth"), c.get("iters"), c.get("seed"))


def skip_reason(job: dict, host: dict, gates: dict) -> tuple[str, str] | None:
    cell, variant, threads = job["cell"], job["variant"], job["threads"]
    v = resolve(variant)
    gpu_max_cols = gates.get("gpu_max_cols", GPU_MAX_COLS)
    mem_gate = gates.get("mem_gate", "on") != "off"
    if variant == "lgbm_cuda":
        return ("unsupported", "pip lightgbm lacks CUDA (deferred)")
    if v.device == "cuda" and host["gpu"] is None:
        return ("skipped", "no CUDA device on host")
    if v.device == "cuda" and gpu_max_cols and cell["cols"] > gpu_max_cols:
        return ("skipped", f"cols > {gpu_max_cols} (GPU variant policy)")
    if (v.device == "cuda" and mem_gate and
            est_dev_gb(cell["rows"], cell["cols"]) >
            0.85 * (host["gpu_vram_gb"] or 0)):
        return ("skipped", f"est {est_dev_gb(cell['rows'], cell['cols']):.1f}"
                           f"GB > 0.85x{host['gpu_vram_gb']}GB VRAM")
    if variant.startswith("bonsai") and cell["bins"] > 65_535:
        return ("unsupported", "bonsai bin_id_t is uint16 (max_bin <= 65535)")
    if mem_gate:
        est = est_host_gb(cell["rows"], cell["cols"], cell["n_test"], v.lib)
        if est > 0.8 * host["ram_gb"]:
            return ("skipped", f"est {est:.1f}"
                               f"GB > 0.8x{host['ram_gb']}GB RAM")
    if cell.get("axis") == "threads" and threads > (host["n_vcpu"] or 1):
        return ("skipped", f"threads {threads} > {host['n_vcpu']} vcpus")
    return None


def run_jobs(jobs: list[dict], *, out: str, suite: str, knobs: dict,
             host: dict, run_label: str | None = None, dry_run: bool = False,
             resume_path: str | None = None, timeout_cap: int = 3600,
             gates: dict | None = None, mem_sampler: bool = True,
             data_cache: str | None = None) -> int:
    gates = gates or {}
    out_path = pathlib.Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(cell, variant, threads, repeat, payload):
        payload = dict(payload)
        # The worker reports the versions it imported; fold into host.libs.
        libs = payload.pop("libs", None)
        row_host = (dict(host, libs={**host.get("libs", {}), **libs})
                    if libs else host)
        payload.setdefault("profile", None)
        extra = {"run": run_label} if run_label else {}
        runlog.emit_row(out_path, division="perf", suite=suite, knobs=knobs,
                        host=row_host, timing_mode="in_memory",
                        dataset="synthetic-friedman1", task="reg", cell=cell,
                        variant=variant, threads=threads, repeat=repeat,
                        **extra, **payload)
        print(f"  {variant:>24} t={threads:<3} rows={cell['rows']:>8} "
              f"cols={cell['cols']:>5} bins={cell['bins']:>5} "
              f"-> {payload['status']}"
              + (f" fit={payload['fit_s']}s r2={payload['r2_test']}"
                 if payload["status"] == "ok" else f" ({payload['message']})"))

    done = resume_keys(resume_path) if resume_path else set()

    warmed: set[str] = set()
    for job in jobs:
        cell, variant, threads = job["cell"], job["variant"], job["threads"]
        v = resolve(variant)
        skip = skip_reason(job, host, gates)
        if skip:
            if not dry_run and _job_key(job, 0) not in done:
                emit(cell, variant, threads, 0,
                     {"status": skip[0], "message": skip[1]})
            else:
                print(f"  {variant:>24} {cell['rows']}x{cell['cols']}x"
                      f"{cell['bins']} -> {skip[0]}: {skip[1]}")
            continue
        timeout = min(cell.get("timeout_s") or timeout_for(cell), timeout_cap)
        if dry_run:
            print(f"  {variant:>24} t={threads:<3} {cell['rows']}x"
                  f"{cell['cols']}x{cell['bins']} timeout={timeout}s "
                  f"repeats={job['repeats']}")
            continue
        if v.device == "cuda" and v.lib not in warmed:
            warm_cell = {"axis": "warmup", "rows": 32_768, "cols": 16,
                         "bins": 63, "bins_effective": 63,
                         "depth": cell["depth"], "iters": 5, "lr": cell["lr"],
                         "informative": cell["informative"], "n_test": 1024,
                         "seed": cell["seed"]}
            run_one({"cell": warm_cell, "variant": variant, "threads": 4},
                    timeout=600, data_cache=data_cache)
            warmed.add(v.lib)
        sample = (mem_sampler and v.device == "cuda"
                  and host.get("gpu") is not None)
        for rep in range(job["repeats"]):
            if _job_key(job, rep) in done:
                print(f"  {variant:>24} t={threads:<3} {cell['rows']}x"
                      f"{cell['cols']}x{cell['bins']} rep={rep} -> resume-skip")
                continue
            child = {"cell": cell, "variant": variant, "threads": threads}
            emit(cell, variant, threads, rep,
                 run_one(child, timeout, sampler=sample, data_cache=data_cache))
    return 0
