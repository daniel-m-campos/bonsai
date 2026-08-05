"""Parent-side orchestration for benchmark jobs (spec-driven and legacy).

One child process per (cell, variant, repeat) via the worker protocol in
bonsai.bench.runners: OOM and segfaults become jsonl status rows instead of
killing the sweep. Every row goes through runlog.emit_row (schema v1) with an
optional run label, so ad-hoc campaigns stop inventing schemas.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import json
import os
import pathlib
import re
import subprocess
import sys
import threading

from bonsai.bench import runlog
from bonsai.bench.variants import Device, Lib, resolve

# GPU variants skip the widest cells by default: 16k+ cols exhausts consumer
# VRAM and kernel grids; the per-host VRAM estimator handles the rest. Both
# gates are policy (a spec can disable them when OOM itself is the datum).
GPU_MAX_COLS = 16_384

PROFILE_RE = re.compile(r"(\w+)=([\d.]+)s")

# Backtrace frames and the exit-time profiler lines print AFTER the actual
# exception, so the last stderr line is usually noise; the campaign's first
# OOM was classified "error" with a bare "[bt] (8) ..." message because of it.
_NOISE_RE = re.compile(r"^\s*(\[bt\]|Stack trace|cuda-profile:|grow-profile:|"
                       r"ingest-profile:|fit-profile:|cuda-upload-decomp:)")

_GATE_KEYS = {"mem_gate", "gpu_max_cols"}


# Public Functions =================================================================================

def parse_profiles(stderr: str) -> dict:
    """The exit-time profiler lines as one flat {bucket_key: seconds} dict."""
    prof = {}
    for line in stderr.splitlines():
        if line.startswith(("cuda-profile:", "grow-profile:", "ingest-profile:",
                            "fit-profile:", "cuda-upload-decomp:")):
            prefix = line.split(":", 1)[0].removesuffix("-profile")
            for key, val in PROFILE_RE.findall(line):
                prof[f"{prefix}_{key}"] = float(val)
    return prof


def est_host_gb(rows: int, cols: int, n_test: int, lib: str) -> float:
    """Rough peak host memory for a cell (float32 data x library factor)."""
    factor = 4.0 if lib == Lib.CATBOOST else 3.0
    return (rows + n_test) * cols * 6 * factor / 2**30


def est_dev_gb(rows: int, cols: int) -> float:
    """Rough peak device memory: binned matrix + row state + context."""
    return (rows * cols * 2 + rows * 8 + 512 * 2**20) / 2**30


def timeout_for(cell: dict) -> int:
    """Scales with total cells, iterations, and width (histogram cost grows
    with cols at constant cells: measured 7x at 2^31). No self-cap: the
    caller's timeout_cap is the ceiling, so a spec can raise it."""
    cells = cell["rows"] * cell["cols"]
    est = 90.0 * cells / 1e8
    est *= max(1.0, cell.get("iters", 100) / 100.0)
    est *= 1.0 + cell["cols"] / 16_384.0
    return int(max(900, est))


def classify_error(message: str) -> str:
    """oom / unsupported / error, from keywords anywhere in the text."""
    low = message.lower()
    if any(s in low for s in ("out of memory", "memoryerror", "bad_alloc",
                              "cannot allocate", "oom",
                              "cudaerrormemoryallocation")):
        return "oom"
    if any(s in low for s in ("unsupported", "max_bin", "border_count",
                              "invalid parameter", "must be")):
        return "unsupported"
    return "error"


def error_message(stderr: str) -> str:
    """The last non-noise stderr line, truncated for the row."""
    lines = [ln for ln in stderr.strip().splitlines()
             if ln.strip() and not _NOISE_RE.match(ln)]
    return (lines[-1] if lines else "no output")[:300]


class DeviceMemSampler:
    """Samples device memory while the worker child runs; max over samples.

    Records BOTH the child's per-process usage and the whole-device number
    (allocator slack and context overhead lag the per-pid counter), plus the
    sampling interval, so the row never claims more precision than sampled.
    The `source` field names the backend that produced the numbers ("nvml",
    "nvidia-smi", or "injected"): only NVML attributes memory to the child pid
    inside a container, so a reader can tell a per-process number from a
    device total that stood in for one.
    """

    def __init__(self, pid: int, interval_s: float = 0.25, query=None):
        self.pid, self.interval_s = pid, interval_s
        if query is not None:
            self._query, self.source = query, "injected"
        else:
            self._query, self.source = _default_query()
        self._stop = threading.Event()
        self._peak_pid: float | None = None
        self._peak_total: float | None = None
        self._samples = 0
        self._failed = False
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def __repr__(self) -> str:
        return (f"DeviceMemSampler(pid={self.pid}, source={self.source!r}, "
                f"samples={self._samples})")

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)

    def result(self) -> dict | None:
        """The dev_mem row block, or None when nothing was sampled."""
        if not self._samples:
            return None
        gb = 1024.0
        out = {"peak_gb_pid": (round(self._peak_pid / gb, 2)
                               if self._peak_pid is not None else None),
               "peak_gb_total": (round(self._peak_total / gb, 2)
                                 if self._peak_total is not None else None),
               "samples": self._samples, "interval_s": self.interval_s,
               "source": self.source}
        if self._failed:
            out["stopped_early"] = True
        return out

    def _loop(self):
        while not self._stop.is_set():
            # A query failure (driver reset, MIG event) must not kill the
            # thread silently: mark the truncated window so result() shows it.
            try:
                got = self._query(self.pid)
            except Exception:
                self._failed = True
                return
            if got is not None:
                pid_mb, total_mb = got
                if pid_mb is not None:
                    self._peak_pid = max(self._peak_pid or 0.0, pid_mb)
                if total_mb is not None:
                    self._peak_total = max(self._peak_total or 0.0, total_mb)
                self._samples += 1
            self._stop.wait(self.interval_s)


def run_one(spec: dict, timeout: int, sampler: bool = False,
            data_cache: str | None = None) -> dict:
    """One worker child, one cell: the payload dict for the row.

    Timeouts, signals, and nonzero exits come back as status rows
    (timeout/oom/unsupported/error) instead of exceptions. bonsai children run
    with the profile counters on unless BONSAI_BENCH_NO_PROFILE is set, which
    a campaign measuring wall clock wants: the counters cost time on planes
    whose rounds are short enough for a profile sync to be the round.
    """
    env = dict(os.environ)
    if (resolve(spec[runlog.Row.VARIANT]).lib == Lib.BONSAI
            and not env.get("BONSAI_BENCH_NO_PROFILE")):
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
        # A child wedged in an uninterruptible driver call can survive
        # SIGKILL for a while; never block the sweep on it unboundedly.
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.communicate(timeout=30)
        out = {runlog.Row.STATUS: "timeout", runlog.Row.MESSAGE: f"exceeded {timeout}s"}
        if sm:
            out[runlog.Row.DEV_MEM] = sm.result()
        return out
    dev_mem = sm.result() if sm else None
    result_line = next((ln for ln in stdout.splitlines()
                        if ln.startswith("RESULT ")), None)
    if proc.returncode != 0 or result_line is None:
        if proc.returncode < 0:
            out = {runlog.Row.STATUS: "oom",
                   runlog.Row.MESSAGE: f"killed by signal {-proc.returncode}"}
        else:
            # Classify from the WHOLE stderr (the OOM keyword rarely sits on
            # the last line); report the last non-noise line as the message.
            out = {runlog.Row.STATUS: classify_error(stderr),
                   runlog.Row.MESSAGE: error_message(stderr)}
        if sm:
            out[runlog.Row.DEV_MEM] = dev_mem
        return out
    out = json.loads(result_line.removeprefix("RESULT "))
    out[runlog.Row.STATUS] = "ok"
    out[runlog.Row.MESSAGE] = None
    prof = parse_profiles(stderr)
    out[runlog.Row.PROFILE] = prof or None
    if sm:
        out[runlog.Row.DEV_MEM] = dev_mem
    return out


def resume_keys(path: str | pathlib.Path) -> set[tuple]:
    """Keys already measured (pods die: funds, spot reaps). ok/unsupported are
    final; skipped/oom/timeout/error re-attempt — the new host may differ.
    Host and run label are part of the key so a second host appends its own
    rows instead of resume-skipping against another machine's. Rows without a
    cell dict (older suite schemas) are ignored, not fatal."""
    done = set()
    for line in pathlib.Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        c = r.get(runlog.Row.CELL)
        if r.get(runlog.Row.STATUS) not in ("ok", "unsupported") or not isinstance(c, dict):
            continue
        host = r.get("host")
        host_name = host.get("name") if isinstance(host, dict) else host
        done.add((r.get(runlog.Row.VARIANT), r.get(runlog.Row.THREADS), r.get(runlog.Row.REPEAT),
                  c.get("rows"), c.get("cols"), c.get("bins"), c.get("depth"),
                  c.get("iters"), c.get("seed"), c.get("eval_mode"),
                  host_name, r.get(runlog.Row.RUN)))
    return done


def skip_reason(job: dict, host: dict, gates: dict) -> tuple[str, str] | None:
    """(status, message) when a job must not run on this host, else None."""
    cell, variant, threads = job[runlog.Row.CELL], job[runlog.Row.VARIANT], job[runlog.Row.THREADS]
    v = resolve(variant)
    gpu_max_cols = gates.get("gpu_max_cols", GPU_MAX_COLS)
    mem_gate = gates.get("mem_gate", "on") != "off"
    if v.device == Device.CUDA and host["gpu"] is None:
        return ("skipped", "no CUDA device on host")
    if v.device == Device.CUDA and gpu_max_cols and cell["cols"] > gpu_max_cols:
        return ("skipped", f"cols > {gpu_max_cols} (GPU variant policy)")
    if (v.device == Device.CUDA and mem_gate and
            est_dev_gb(cell["rows"], cell["cols"]) >
            0.85 * (host["gpu_vram_gb"] or 0)):
        return ("skipped", f"est {est_dev_gb(cell['rows'], cell['cols']):.1f}"
                           f"GB > 0.85x{host['gpu_vram_gb']}GB VRAM")
    if v.lib == Lib.BONSAI and cell["bins"] > 65_535:
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
    """Run (or dry-plan) a job list, one row per attempt, resume-aware."""
    gates = gates or {}
    unknown_gates = set(gates) - _GATE_KEYS
    if unknown_gates:
        raise ValueError(f"unknown gate keys: {sorted(unknown_gates)} "
                         f"(known: {sorted(_GATE_KEYS)})")
    out_path = pathlib.Path(out)
    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    sink = _Sink(out_path=out_path, suite=suite, knobs=knobs, host=host,
                 run_label=run_label)
    done = resume_keys(resume_path) if resume_path else set()
    warmed: set[str] = set()
    for job in jobs:
        cell, variant, threads = (job[runlog.Row.CELL], job[runlog.Row.VARIANT],
                                  job[runlog.Row.THREADS])
        v = resolve(variant)
        # CatBoost-GPU caps borders at 254 inside catboost_core; the row must
        # record the bins that actually ran (protocol: bins_effective).
        if v.lib == Lib.CATBOOST and v.device == Device.CUDA and cell["bins"] > 255:
            cell = dict(cell, bins_effective=255)
        skip = skip_reason(job, host, gates)
        if skip:
            _handle_skip(sink, job, cell, skip, dry_run=dry_run, done=done)
            continue
        cell_timeout = cell.get("timeout_s")
        timeout = min(cell_timeout if cell_timeout is not None
                      else timeout_for(cell), timeout_cap)
        if dry_run:
            _print_dry_plan(sink, job, cell, timeout, done)
            continue
        if v.device == Device.CUDA and v.lib not in warmed:
            _warm_gpu(cell, variant, data_cache)
            warmed.add(v.lib)
        sample = mem_sampler and v.device == Device.CUDA
        for rep in range(job["repeats"]):
            if _job_key(job, rep, sink.host_name, run_label) in done:
                print(f"  {variant:>24} t={threads:<3} {cell['rows']}x"
                      f"{cell['cols']}x{cell['bins']} rep={rep} -> resume-skip")
                continue
            child = {runlog.Row.CELL: cell, runlog.Row.VARIANT: variant,
                     runlog.Row.THREADS: threads}
            sink.emit(cell, variant, threads, rep,
                      run_one(child, timeout, sampler=sample,
                              data_cache=data_cache))
    return 0


# Private Helpers ==================================================================================

def _device_index() -> int:
    """The first entry of CUDA_VISIBLE_DEVICES is the device the worker child
    will use; defaulting to 0 on a multi-GPU host would attribute another
    tenant's memory to this run."""
    first = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    return int(first) if first.isdigit() else 0


@functools.lru_cache(maxsize=1)
def _default_query() -> tuple:
    """The (query, source) pair this host samples device memory with.

    NVML is preferred and initialized once: inside a container it attributes
    memory to the child's own pid, while nvidia-smi's compute-apps query
    reports host-namespace pids that match nothing, which is how per-process
    VRAM came back null on the L40S pods. nvidia-smi remains the fallback for
    hosts without `nvidia-ml-py` installed.
    """
    # Any NVML failure (missing package, no driver, MIG topology) means the
    # fallback, never a dead sampler.
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(_device_index())
    except Exception:
        return _smi_query, "nvidia-smi"
    return functools.partial(_nvml_query, pynvml, handle), "nvml"


def _nvml_query(pynvml, handle, pid: int):
    """(pid_mb, total_mb) from NVML for one device handle."""
    pid_mb = None
    for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
        if proc.pid == pid and proc.usedGpuMemory:
            pid_mb = (pid_mb or 0) + proc.usedGpuMemory / 2**20
    return pid_mb, pynvml.nvmlDeviceGetMemoryInfo(handle).used / 2**20


def _smi_query(pid: int):
    """(pid_mb, total_mb) from nvidia-smi, or None when the query fails."""
    dev = ["-i", str(_device_index())]
    try:
        apps = subprocess.run(
            ["nvidia-smi", *dev, "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        total = subprocess.run(
            ["nvidia-smi", *dev, "--query-gpu=memory.used",
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


def _job_key(job: dict, repeat: int, host_name: str | None,
             run_label: str | None) -> tuple:
    """The resume identity of one attempt; must mirror resume_keys().

    eval_mode is part of the identity because the early-stopping suite runs
    the same shape twice at the same iteration count, once with the eval set
    and once without; without it the second arm resume-skips the first.
    Legacy cells carry no eval_mode, so committed rows still match on None.
    """
    c = job[runlog.Row.CELL]
    return (job[runlog.Row.VARIANT], job[runlog.Row.THREADS], repeat, c["rows"], c["cols"],
            c["bins"], c.get("depth"), c.get("iters"), c.get("seed"),
            c.get("eval_mode"), host_name, run_label)


@dataclasses.dataclass(frozen=True)
class _Sink:
    """Where run_jobs writes rows and how it labels them."""

    out_path: pathlib.Path
    suite: str
    knobs: dict
    host: dict
    run_label: str | None

    @property
    def host_name(self) -> str | None:
        return self.host.get("name")

    def emit(self, cell: dict, variant: str, threads: int, repeat: int,
             payload: dict):
        """One row to the jsonl plus the one-line progress print."""
        payload = dict(payload)
        # The worker reports the versions it imported; fold into host.libs.
        libs = payload.pop("libs", None)
        row_host = (dict(self.host, libs={**self.host.get("libs", {}), **libs})
                    if libs else self.host)
        payload.setdefault(runlog.Row.PROFILE, None)
        extra = {runlog.Row.RUN: self.run_label} if self.run_label else {}
        runlog.emit_row(self.out_path, division="perf", suite=self.suite,
                        knobs=self.knobs, host=row_host,
                        timing_mode="in_memory",
                        dataset="synthetic-friedman1", task="reg", cell=cell,
                        variant=variant, threads=threads, repeat=repeat,
                        **extra, **payload)
        print(f"  {variant:>24} t={threads:<3} rows={cell['rows']:>8} "
              f"cols={cell['cols']:>5} bins={cell['bins']:>5} "
              f"-> {payload[runlog.Row.STATUS]}"
              + (f" fit={payload[runlog.Row.FIT_S]}s r2={payload[runlog.Row.R2_TEST]}"
                 if payload[runlog.Row.STATUS] == "ok" else f" ({payload[runlog.Row.MESSAGE]})"))


def _handle_skip(sink: _Sink, job: dict, cell: dict, skip: tuple[str, str], *,
                 dry_run: bool, done: set[tuple]):
    """Record a gated job: a status row normally, a print when dry/duplicate."""
    variant, threads = job[runlog.Row.VARIANT], job[runlog.Row.THREADS]
    key = _job_key(job, 0, sink.host_name, sink.run_label)
    if not dry_run and key not in done:
        sink.emit(cell, variant, threads, 0,
                  {runlog.Row.STATUS: skip[0], runlog.Row.MESSAGE: skip[1]})
        return
    print(f"  {variant:>24} {cell['rows']}x{cell['cols']}x"
          f"{cell['bins']} -> {skip[0]}: {skip[1]}")


def _print_dry_plan(sink: _Sink, job: dict, cell: dict, timeout: int,
                    done: set[tuple]):
    """The dry-run line for one job, with its resume-skip count."""
    variant, threads = job[runlog.Row.VARIANT], job[runlog.Row.THREADS]
    already = sum(1 for rep in range(job["repeats"])
                  if _job_key(job, rep, sink.host_name, sink.run_label) in done)
    print(f"  {variant:>24} t={threads:<3} {cell['rows']}x"
          f"{cell['cols']}x{cell['bins']} timeout={timeout}s "
          f"repeats={job['repeats']}"
          + (f" resume-skip={already}/{job['repeats']}" if already else ""))


def _warm_gpu(cell: dict, variant: str, data_cache: str | None):
    """One tiny unrecorded fit so CUDA context/JIT cost stays out of rep 0."""
    warm_cell = {"axis": "warmup", "rows": 32_768, "cols": 16,
                 "bins": 63, "bins_effective": 63,
                 "depth": cell["depth"], "iters": 5, "lr": cell["lr"],
                 "informative": cell["informative"], "n_test": 1024,
                 "seed": cell["seed"]}
    run_one({runlog.Row.CELL: warm_cell, runlog.Row.VARIANT: variant,
             runlog.Row.THREADS: 4}, timeout=600, data_cache=data_cache)
