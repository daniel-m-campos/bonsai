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
    PREDICT_S: Final = "predict_s"
    R2_TRAIN: Final = "r2_train"
    R2_TEST: Final = "r2_test"
    AUC_TEST: Final = "auc_test"
    PEAK_RSS_GB: Final = "peak_rss_gb"
    PROFILE: Final = "profile"
    DEV_MEM: Final = "dev_mem"


def git_sha() -> str:
    """The short HEAD sha, or "unknown" outside a git checkout."""
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


def detect_host(name: str | None = None) -> dict:
    """The host block every row embeds.

    Returns
    -------
    dict
        name, gpu, gpu_vram_gb, cpu_model, n_vcpu, ram_gb, os, python,
        and libs (the imported reference-library versions).
    """
    gpu, vram = None, None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            g, v = out.stdout.strip().splitlines()[0].split(",")
            gpu, vram = g.strip(), round(float(v) / 1024, 1)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    if sys.platform == "darwin":
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                             capture_output=True, text=True).stdout.strip()
        ram = round(int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                       capture_output=True,
                                       text=True).stdout) / 2**30)
    else:
        cpu = ""
        for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
        ram = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
                    / 2**30)
    return {"name": name or platform.node(), "gpu": gpu, "gpu_vram_gb": vram,
            "cpu_model": cpu, "n_vcpu": os.cpu_count(), "ram_gb": ram,
            "os": platform.platform(), "python": platform.python_version(),
            "libs": lib_versions()}


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
