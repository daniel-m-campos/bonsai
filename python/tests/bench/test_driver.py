"""Tests for bonsai.bench.driver (run_jobs, gates, resume, sampler)."""

from __future__ import annotations

import json
import tempfile

import pytest
from bonsai.bench import runlog


def test_driver_resume_and_emit():
    import pathlib

    from bonsai.bench import driver

    legacy_row = {"status": "ok", "variant": "bonsai_depthwise", "threads": 16,
                  "repeat": 0, "cell": {"rows": 1_000_000, "cols": 100,
                                        "bins": 255, "depth": 8, "iters": 100,
                                        "seed": 42}}
    err_row = dict(legacy_row, status="error", repeat=1)
    with tempfile.TemporaryDirectory() as td:
        prior = pathlib.Path(td) / "prior.jsonl"
        prior.write_text(json.dumps(legacy_row) + "\n"
                         + json.dumps(err_row) + "\n")
        done = driver.resume_keys(prior)
        assert ("bonsai_depthwise", 16, 0, 1_000_000, 100, 255, 8, 100, 42,
                None, None) in done
        assert len(done) == 1  # the error row re-attempts

        host = {"name": "t", "gpu": None, "gpu_vram_gb": None, "cpu_model": "t",
                "n_vcpu": 8, "ram_gb": 64.0, "os": "t", "python": "3",
                "libs": {"numpy": "0"}}
        cell = {"axis": "cell", "rows": 1000, "cols": 10, "bins": 255,
                "bins_effective": 255, "depth": 8, "iters": 100, "lr": 0.1,
                "informative": 20, "n_test": 200, "seed": 42}
        out = pathlib.Path(td) / "out.jsonl"
        # Stub the child: the emit path is what is under test.
        real_run_one = driver.run_one
        driver.run_one = lambda spec, timeout, **kw: {
            "status": "ok", "message": None, "fit_s": 1.0, "predict_s": 0.1,
            "r2_train": 0.9, "r2_test": 0.9, "peak_rss_gb": 0.1,
            "libs": {"xgboost": "9.9.9"}, "profile": None}
        try:
            driver.run_jobs(
                [{"cell": cell, "variant": "xgb_hist", "threads": 4, "repeats": 1},
                 {"cell": cell, "variant": "xgb_cuda", "threads": 4, "repeats": 1}],
                out=str(out), suite="test", knobs={"a": 1}, host=host,
                run_label="unit")
        finally:
            driver.run_one = real_run_one
        rows = [json.loads(ln) for ln in out.read_text().splitlines()]
        ok, skip = rows
        assert ok["schema"] == 1 and ok["suite"] == "test"
        assert ok["run"] == "unit" and ok["repeat"] == 0
        assert ok["host"]["libs"] == {"numpy": "0", "xgboost": "9.9.9"}
        assert "libs" not in ok  # folded into host, never a stray column
        assert ok["knobs_hash"] == runlog.knobs_hash({"a": 1})
        assert skip["variant"] == "xgb_cuda" and skip["status"] == "skipped"
        assert skip["repeat"] == 0  # no CUDA on this host


def test_mem_sampler():
    import time as _time

    from bonsai.bench import driver

    seen = iter([(100.0, 2048.0), (1536.0, 4096.0), (512.0, 3072.0)])

    def fake_query(pid):
        assert pid == 1234
        return next(seen, None) or (None, None)

    with driver.DeviceMemSampler(1234, interval_s=0.01, query=fake_query) as sm:
        _time.sleep(0.1)
    got = sm.result()
    assert got["peak_gb_pid"] == 1.5 and got["peak_gb_total"] == 4.0
    assert got["samples"] >= 3 and got["interval_s"] == 0.01
    assert got["source"] == "injected"
    # A query that never answers yields no result, not zeros.
    with driver.DeviceMemSampler(1, interval_s=0.01, query=lambda p: None) as sm2:
        _time.sleep(0.03)
    assert sm2.result() is None
    # nvidia-smi CSV parsing: our pid, a foreign pid, and the device total.
    real = driver.subprocess.run

    class FakeProc:
        def __init__(self, stdout):
            self.stdout, self.returncode = stdout, 0

    def fake_run(cmd, **kw):
        if "--query-compute-apps=pid,used_memory" in cmd:
            return FakeProc("999, 512\n4321, 1024\n999, 256\n")
        return FakeProc("7168\n")

    driver.subprocess.run = fake_run
    try:
        pid_mb, total_mb = driver._smi_query(999)
    finally:
        driver.subprocess.run = real
    assert pid_mb == 768 and total_mb == 7168


def test_error_classification():
    from bonsai.bench import driver

    oom_stderr = ("Traceback (most recent call last):\n"
                  "xgboost.core.XGBoostError: cudaErrorMemoryAllocation: "
                  "out of memory\n"
                  "  [bt] (7) /lib/libxgboost.so(+0x1) [0x1]\n"
                  "  [bt] (8) /lib/libxgboost.so(+0x2) [0x2]\n"
                  "fit-profile: total=1.2s\n")
    assert driver.classify_error(oom_stderr) == "oom"
    msg = driver.error_message(oom_stderr)
    assert "out of memory" in msg and "[bt]" not in msg
    assert driver.error_message("  [bt] (8) only frames\n") == "no output"


def test_driver_gates_and_effective_bins():
    import pathlib

    from bonsai.bench import driver

    host = {"name": "h", "gpu": "FakeGPU", "gpu_vram_gb": 999.0,
            "cpu_model": "t", "n_vcpu": 8, "ram_gb": 999.0, "os": "t",
            "python": "3", "libs": {}}
    cell = {"axis": "cell", "rows": 1000, "cols": 10, "bins": 1023,
            "bins_effective": 1023, "depth": 8, "iters": 100, "lr": 0.1,
            "informative": 5, "n_test": 200, "seed": 42,
            "min_data_in_leaf": 20, "lambda_l2": 1.0}
    seen = []
    real = driver.run_one
    driver.run_one = lambda spec, timeout, **kw: (
        seen.append(spec) or {
            "status": "ok", "message": None, "fit_s": 1.0, "predict_s": 0.1,
            "r2_train": 0.9, "r2_test": 0.9, "peak_rss_gb": 0.1,
            "profile": None})
    try:
        with tempfile.TemporaryDirectory() as td:
            out = pathlib.Path(td) / "o.jsonl"
            driver.run_jobs(
                [{"cell": dict(cell), "variant": "catboost_gpu", "threads": 4,
                  "repeats": 1}],
                out=str(out), suite="test", knobs={}, host=host,
                run_label="u", mem_sampler=False)
            fit_specs = [s for s in seen if s["cell"]["iters"] == 100]
            assert fit_specs[0]["cell"]["bins_effective"] == 255  # 254+1 cap
            row = [json.loads(ln) for ln in out.read_text().splitlines()][-1]
            assert row["cell"]["bins_effective"] == 255
            assert row["cell"]["bins"] == 1023
            with pytest.raises(ValueError) as e:
                driver.run_jobs([], out=str(out), suite="t", knobs={},
                                host=host, gates={"gpu_max_col": None})
                assert "unknown gate keys" in str(e.value)
    finally:
        driver.run_one = real


def test_resume_is_host_scoped():
    import pathlib

    from bonsai.bench import driver

    host_a = {"name": "pod-a", "gpu": None, "gpu_vram_gb": None,
              "cpu_model": "t", "n_vcpu": 8, "ram_gb": 64.0, "os": "t",
              "python": "3", "libs": {}}
    host_b = dict(host_a, name="workrig")
    cell = {"axis": "cell", "rows": 1000, "cols": 10, "bins": 255,
            "bins_effective": 255, "depth": 8, "iters": 100, "lr": 0.1,
            "informative": 5, "n_test": 200, "seed": 42}
    job = {"cell": cell, "variant": "xgb_hist", "threads": 4, "repeats": 1}
    real = driver.run_one
    driver.run_one = lambda spec, timeout, **kw: {
        "status": "ok", "message": None, "fit_s": 1.0, "predict_s": 0.1,
        "r2_train": 0.9, "r2_test": 0.9, "peak_rss_gb": 0.1, "profile": None}
    try:
        with tempfile.TemporaryDirectory() as td:
            out = pathlib.Path(td) / "o.jsonl"
            driver.run_jobs([dict(job)], out=str(out), suite="t", knobs={},
                            host=host_a, run_label="r", mem_sampler=False)
            # Same host + label resume-skips; another host appends.
            driver.run_jobs([dict(job)], out=str(out), suite="t", knobs={},
                            host=host_a, run_label="r", mem_sampler=False,
                            resume_path=str(out))
            assert len(out.read_text().splitlines()) == 1
            driver.run_jobs([dict(job)], out=str(out), suite="t", knobs={},
                            host=host_b, run_label="r", mem_sampler=False,
                            resume_path=str(out))
            assert len(out.read_text().splitlines()) == 2
    finally:
        driver.run_one = real


def test_timeout_and_sampler_resilience():
    import time as _time

    from bonsai.bench import driver

    assert driver.timeout_for({"rows": 1000, "cols": 8, "iters": 100}) == 900
    base = {"rows": 16_777_216, "cols": 128, "iters": 100}  # 2^31 cells
    assert (driver.timeout_for(dict(base, iters=300))
            > driver.timeout_for(dict(base)))
    # Same cell count, wider aspect: the histogram term must raise it.
    assert (driver.timeout_for({"rows": 32_768, "cols": 65_536, "iters": 100})
            > driver.timeout_for(dict(base)))

    calls = {"n": 0}

    def flaky(pid):
        calls["n"] += 1
        if calls["n"] > 2:
            raise RuntimeError("driver reset")
        return (100.0, 200.0)

    with driver.DeviceMemSampler(1, interval_s=0.01, query=flaky) as sm:
        _time.sleep(0.1)
    got = sm.result()
    assert got["samples"] == 2 and got.get("stopped_early") is True
