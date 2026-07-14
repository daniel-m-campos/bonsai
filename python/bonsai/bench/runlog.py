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

SCHEMA_VERSION = 1
DIVISIONS = ("quality", "perf")
TIMING_MODES = ("in_memory", "pipeline")


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


def knobs_hash(knobs: dict) -> str:
    canon = json.dumps(knobs, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()[:8]


def lib_versions() -> dict:
    libs = {}
    for name in ("bonsai", "xgboost", "lightgbm", "catboost", "numpy"):
        mod = sys.modules.get(name)
        if mod is not None:
            libs[name] = getattr(mod, "__version__", "unknown")
    return libs


def detect_host(name: str | None = None) -> dict:
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
    assert division in DIVISIONS, division
    if timing_mode is not None:
        assert timing_mode in TIMING_MODES, timing_mode
    row = {
        "schema": SCHEMA_VERSION,
        "ts": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
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
