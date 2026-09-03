"""Result-row schema (v1) and environment capture for every benchmark jsonl.

Rows are append-only and additive: files may mix pre-schema rows with v1
rows, and readers must tolerate extra keys. Every row records enough to
reproduce it: the command, the knobs (hashed for grouping), the git sha, and
the host down to library versions.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys
from typing import Final

SCHEMA_VERSION = 1
DIVISIONS = ("quality", "perf")
TIMING_MODES = ("in_memory", "pipeline")

CGROUP_V2_CPU_MAX = "/sys/fs/cgroup/cpu.max"
CGROUP_V1_QUOTA = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
CGROUP_V1_PERIOD = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
CGROUP_V2_MEMORY_MAX = "/sys/fs/cgroup/memory.max"
CGROUP_V1_MEMORY_LIMIT = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
# cgroup v1 spells "no memory limit" as a sentinel near 2**63 rather than a
# word, so any limit at or above this one is no limit at all.
CGROUP_V1_MEMORY_UNLIMITED = 2**62

class Row:
    """Field names shared by the result rows every suite emits."""

    CELL: Final = "cell"
    VARIANT: Final = "variant"
    THREADS: Final = "threads"
    REPEAT: Final = "repeat"
    RUN: Final = "run"
    STATUS: Final = "status"
    MESSAGE: Final = "message"
    FIT_S: Final = "fit_s"
    INGEST_S: Final = "ingest_s"
    TRAIN_S: Final = "train_s"
    PREDICT_S: Final = "predict_s"
    CONTRIBS_S: Final = "contribs_s"
    CONTRIBS_ROWS_PER_S: Final = "contribs_rows_per_s"
    CONTRIBS_ADDITIVITY: Final = "contribs_additivity"
    R2_TRAIN: Final = "r2_train"
    R2_TEST: Final = "r2_test"
    AUC_TEST: Final = "auc_test"
    PEAK_RSS_GB: Final = "peak_rss_gb"
    PROFILE: Final = "profile"
    DEV_MEM: Final = "dev_mem"


def git_sha() -> str:
    """The commit these rows describe, or "unknown" if nothing states it.

    `BONSAI_BENCH_GIT_SHA` wins when set, because the fallback asks git about
    the *current directory*: a runner launched from anywhere but the checkout
    records "unknown" and the rows become unattributable, which is what the
    pod script's checkout assertion exists to prevent.

    Returns
    -------
    str
        The stated sha, the short HEAD sha, or "unknown".
    """
    stated = os.environ.get("BONSAI_BENCH_GIT_SHA", "").strip()
    if stated:
        return stated
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


def knobs_hash(knobs: dict) -> str:
    """Canonical 8-hex-char digest of a knobs dict.

    Key order must not matter (sort_keys) and the encoding must never
    change: committed rows carry these values and resume matching
    compares them.
    """
    canon = json.dumps(knobs, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()[:8]


def lib_versions() -> dict:
    """Versions of the already-imported reference libraries only."""
    libs = {}
    for name in ("bonsai", "xgboost", "lightgbm", "catboost", "numpy"):
        mod = sys.modules.get(name)
        if mod is not None:
            libs[name] = getattr(mod, "__version__", "unknown")
    return libs


def cpu_quota() -> float | None:
    """The cores this container may use, or None when nothing caps it.

    Two mechanisms cap a container and a host can carry either, so the
    tighter of the two is the readable ceiling. A GPU pod caps CPU
    bandwidth with a CFS quota well below what the machine advertises
    (13.6 cores against 128, issue #355) and a spin-wait barrier burns
    exactly the bandwidth the cap withholds. A rented CPU pod caps by
    cpuset instead: its CFS quota reads `max`, and the purchased vCPU
    count appears as that many CPUs in the affinity mask, so a quota read
    alone would call it unlimited.

    Returns
    -------
    float or None
        Cores the cgroup allows, or None on a host neither mechanism caps
        (bare metal, or macOS, where the controller files do not exist).
    """
    caps = [cap for cap in (_cfs_quota_cores(), _cpuset_cores())
            if cap is not None]
    return min(caps) if caps else None


def usable_cpus() -> int:
    """CPUs this process may run on, which is what `nproc` reports."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def usable_ram_gb() -> int:
    """RAM this container may use, the machine's when nothing caps it.

    /proc/meminfo and `free` report the machine even inside a container, so
    a memory gate reading them alone cannot fail where it matters: a 4GB
    container reads the host's 1TB and every cell looks affordable right up
    to the OOM kill. The cgroup limit is the number to compare against, and
    the smaller of the two is the honest answer on any host.

    Returns
    -------
    int
        Whole GB the container may allocate.
    """
    limit = _cgroup_memory_limit_bytes()
    total = _machine_ram_bytes()
    return round(min(total, limit) / 2**30 if limit else total / 2**30)


def detect_host(name: str | None = None) -> dict:
    """The host block every row embeds.

    Returns
    -------
    dict
        name, gpu, driver, gpu_vram_gb, cpu_model, n_vcpu, ram_gb, os,
        python, libs (the imported reference-library versions), and the
        two fields a CPU-plane row is only readable with: cpu_quota (the
        cgroup ceiling in cores, null when nothing caps the host) and
        omp_wait_policy.

    ``n_vcpu`` and ``ram_gb`` both describe the container rather than the
    machine, because on a rented pod the two differ by orders of magnitude
    (16 CPUs against 256, 4GB against 1TB) and the smaller number is the
    one a thread count or a cell size has to be read against.
    """
    gpu, vram = None, None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            gpu, vram = _gpu_name_and_vram_gb(out.stdout.strip().splitlines()[0])
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    driver = None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                              "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            driver = out.stdout.strip().splitlines()[0].strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    if sys.platform == "darwin":
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                             capture_output=True, text=True).stdout.strip()
    else:
        cpu = ""
        for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    ram = usable_ram_gb()
    return {"name": name or platform.node(), "gpu": gpu, "driver": driver,
            "gpu_vram_gb": vram, "cpu_model": cpu, "n_vcpu": usable_cpus(),
            "cpu_quota": cpu_quota(),
            "omp_wait_policy": os.environ.get("OMP_WAIT_POLICY"),
            "ram_gb": ram, "os": platform.platform(),
            "python": platform.python_version(), "libs": lib_versions()}


def emit_row(path: str | pathlib.Path, *, division: str, suite: str,
             knobs: dict | None = None, host: dict | None = None,
             timing_mode: str | None = None, **fields) -> dict:
    """Append one schema-v1 row; returns the row. Extra keyword fields pass
    through verbatim so suite-specific columns (cell, kind, ...) survive."""
    if division not in DIVISIONS:
        raise ValueError(f"unknown division {division!r} (known: {DIVISIONS})")
    if timing_mode is not None and timing_mode not in TIMING_MODES:
        raise ValueError(
            f"unknown timing_mode {timing_mode!r} (known: {TIMING_MODES})")
    row = {
        "schema": SCHEMA_VERSION,
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "division": division,
        "suite": suite,
        "script": getattr(sys.modules.get("__main__"), "__file__", None),
        "cmd": " ".join([sys.executable, *sys.argv]),
        "timing_mode": timing_mode,
        "host": host if host is not None else detect_host(),
    }
    if knobs is not None:
        row["knobs"] = knobs
        row["knobs_hash"] = knobs_hash(knobs)
    row.update(fields)
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def peak_rss_gb() -> float:
    """This process's peak RSS. ru_maxrss: bytes on macOS, KiB on Linux."""
    import resource
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(ru / (2**30 if sys.platform == "darwin" else 2**20), 2)


def repo_root() -> pathlib.Path | None:
    """The enclosing source checkout, if this package is imported from one.

    Walks up from the package rather than assuming a fixed depth: under
    PYTHONPATH=build/python a fixed parents[3] lands in the build tree,
    whose benchmarks/ directory make clean deletes (campaign rows died
    that way once).
    """
    for p in pathlib.Path(__file__).resolve().parents:
        if (p / "pyproject.toml").is_file() and (p / "benchmarks").is_dir():
            return p
    return None


# Private Functions ================================================================================

def _gpu_name_and_vram_gb(smi_line: str) -> tuple[str, float | None]:
    """Parse one ``name,memory.total`` line from nvidia-smi.

    An integrated GPU (Jetson) shares the host's RAM and reports its memory
    as ``[N/A]``, so the size is None there rather than a parse error.
    """
    name, _, memory_mib = smi_line.partition(",")
    try:
        vram = round(float(memory_mib) / 1024, 1)
    except ValueError:
        vram = None
    return name.strip(), vram


def _cfs_quota_cores() -> float | None:
    """The CFS bandwidth ceiling in cores; None when it reads unlimited."""
    v2 = pathlib.Path(CGROUP_V2_CPU_MAX)
    if v2.is_file():
        quota, _, period = v2.read_text().strip().partition(" ")
        if quota == "max":
            return None
        return round(int(quota) / int(period), 2)
    quota_file, period_file = pathlib.Path(CGROUP_V1_QUOTA), pathlib.Path(CGROUP_V1_PERIOD)
    if not quota_file.is_file() or not period_file.is_file():
        return None
    quota_us = int(quota_file.read_text().strip())
    if quota_us <= 0:
        return None
    return round(quota_us / int(period_file.read_text().strip()), 2)


def _cpuset_cores() -> float | None:
    """CPUs in the affinity mask, or None when the mask is the whole machine."""
    allowed = usable_cpus()
    return None if allowed >= (os.cpu_count() or allowed) else float(allowed)


def _cgroup_memory_limit_bytes() -> int | None:
    """The cgroup memory ceiling in bytes; None when it reads unlimited."""
    v2 = pathlib.Path(CGROUP_V2_MEMORY_MAX)
    if v2.is_file():
        value = v2.read_text().strip()
        return None if value == "max" else int(value)
    v1 = pathlib.Path(CGROUP_V1_MEMORY_LIMIT)
    if not v1.is_file():
        return None
    limit = int(v1.read_text().strip())
    return None if limit >= CGROUP_V1_MEMORY_UNLIMITED else limit


def _machine_ram_bytes() -> int:
    """Physical RAM the machine owns, container limits ignored."""
    if sys.platform == "darwin":
        out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                             capture_output=True, text=True).stdout
        return int(out.strip())
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
