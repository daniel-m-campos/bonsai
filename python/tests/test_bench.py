"""bonsai.bench invariants: run with PYTHONPATH=build/python python
python/tests/test_bench.py (wired into make python-test)."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile

import numpy as np
from bonsai.bench import metrics, params, runlog, synth

# Captured from scripts/bench_scaling.py::gen_data BEFORE the move to
# bonsai.bench.synth (2026-07-14): the generator must stay byte-stable or
# every perf-division result loses comparability.
GEN_DATA_GOLDENS = {
    (10_000, 20, 42, 1000, 20): ["33172850a22efea3", "72dd8c4d554f8e15",
                                 "2a69346b81ee31cb", "699d900c1abda049"],
    (5000, 7, 0, 500, 5): ["4052a735cd2e4edf", "bf4a1f3766b784c6",
                           "62e923970a54e7a9", "2e46ac59454ce2d1"],
}


def test_gen_data_bytestable():
    for args, want in GEN_DATA_GOLDENS.items():
        got = [hashlib.sha256(a.tobytes()).hexdigest()[:16]
               for a in synth.gen_data(*args)]
        assert got == want, (args, got, want)


def test_reference_param_mappings():
    x = params.xgb_core(learning_rate=0.05, max_depth=6, min_data_in_leaf=20,
                        lambda_l2=1.0, max_bin=255, seed=42)
    assert x["min_child_weight"] == 20 and x["tree_method"] == "hist"
    lgb = params.lgbm_core(learning_rate=0.1, max_depth=8, num_leaves=256,
                           min_data_in_leaf=20, lambda_l2=1.0, max_bin=255,
                           seed=42)
    assert lgb["num_leaves"] == 256 and lgb["verbose"] == -1
    # max_bin arrives in BIN semantics; catboost_core owns the borders
    # fencepost (bins - 1) and the GPU 254-border cap (fairness review
    # 2026-07-30: call-site translations had drifted three ways).
    cb_gpu = params.catboost_core(learning_rate=0.1, max_depth=8, lambda_l2=1.0,
                                  max_bin=1023, seed=42, device="cuda")
    cb_cpu = params.catboost_core(learning_rate=0.1, max_depth=8, lambda_l2=1.0,
                                  max_bin=1023, seed=42, device="cpu")
    assert cb_gpu["border_count"] == 254 and cb_cpu["border_count"] == 1022
    cb_255 = params.catboost_core(learning_rate=0.1, max_depth=8, lambda_l2=1.0,
                                  max_bin=255, seed=42, device="cpu")
    assert cb_255["border_count"] == 254  # 255 bins, matching the others
    assert params.num_leaves_campaign(6) == 63
    assert params.num_leaves_full(6) == 64
    # the shim keeps the documented import path alive
    sys.path.insert(0, "scripts")
    import reference_params as rp
    assert rp.xgb_core is params.xgb_core


def test_bonsai_core_pairs():
    # The exact dotted keys run_bonsai used to hand-build; a drift here
    # changes every perf row silently.
    pairs = params.bonsai_core(learning_rate=0.1, max_depth=8, num_leaves=256,
                               min_data_in_leaf=20, lambda_l2=1.0, max_bin=255,
                               seed=42, n_iters=100, n_threads=16,
                               grower="depthwise")
    assert dict(pairs) == {
        "dispatch.grower_name": "depthwise",
        "dispatch.objective_name": "mse",
        "booster.n_iters": "100",
        "booster.learning_rate": "0.1",
        "booster.random_seed": "42",
        "tree.max_depth": "8",
        "tree.max_leaves": "256",
        "tree.min_data_in_leaf": "20",
        "tree.lambda_l2": "1.0",
        "bin_mapper.max_bin": "255",
        "parallel.n_threads": "16",
    }
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in pairs)
    logloss = params.bonsai_core(learning_rate=0.1, max_depth=8, num_leaves=256,
                                 min_data_in_leaf=20, lambda_l2=1.0, max_bin=255,
                                 seed=42, n_iters=100, n_threads=16,
                                 grower="oblivious", objective="logloss")
    assert dict(logloss)["dispatch.objective_name"] == "logloss"


def test_variant_registry():
    from bonsai.bench import airline, grinsztajn, scaling, variants

    # Committed rows pin these exact spellings forever.
    assert variants.SCALING == tuple(scaling.VARIANTS) == (
        "bonsai_depthwise", "bonsai_leafwise", "bonsai_oblivious",
        "bonsai_cuda_depthwise", "bonsai_cuda_oblivious", "xgb_hist",
        "xgb_cuda", "lgbm_cpu", "lgbm_cuda", "catboost_cpu", "catboost_gpu")
    assert variants.AIRLINE == tuple(airline.VARIANTS) == (
        "bonsai_depthwise", "bonsai_oblivious", "bonsai_cuda_depthwise",
        "bonsai_cuda_oblivious", "bonsai_ts_depthwise", "bonsai_ts_oblivious",
        "bonsai_ts_cuda_depthwise", "bonsai_ts_cuda_oblivious", "xgb_hist",
        "xgb_cuda", "lgbm_cpu", "catboost_cpu", "catboost_gpu")
    assert grinsztajn.VARIANTS == variants.GRINSZTAJN == (
        "bonsai_dw", "bonsai_lw", "bonsai_obl", "xgb", "lgbm", "catboost")
    for n in (*variants.SCALING, *variants.AIRLINE):
        assert variants.resolve(n).name == n
    # Historical alias spellings resolve to the intended canonical arm.
    for alias, canon in {"bonsai_dw": "bonsai_depthwise",
                         "bonsai_lw": "bonsai_leafwise",
                         "bonsai_obl": "bonsai_oblivious",
                         "xgb": "xgb_hist", "lgbm": "lgbm_cpu",
                         "catboost": "catboost_cpu",
                         "bonsai_cpu": "bonsai_depthwise",
                         "bonsai_leaf_cpu": "bonsai_leafwise",
                         "bonsai_gpu": "bonsai_cuda_depthwise",
                         "bonsai_obl_gpu": "bonsai_cuda_oblivious",
                         "xgb_cpu": "xgb_hist",
                         "xgb_gpu": "xgb_cuda"}.items():
        assert variants.resolve(alias).name == canon
    # No alias shadows a canonical name.
    aliases = {a for v in variants.REGISTRY.values() for a in v.aliases}
    assert not aliases & set(variants.REGISTRY)
    # The (lib, device) views survive the derivation.
    assert scaling.VARIANTS["bonsai_cuda_oblivious"] == ("bonsai", "cuda")
    assert scaling.VARIANTS["lgbm_cuda"] == ("lgbm", "cuda")
    assert airline.VARIANTS["bonsai_ts_cuda_depthwise"] == ("bonsai", "cuda")
    try:
        variants.resolve("nope")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown variant must be rejected")


def test_spec_expansion():
    from bonsai.bench import spec as spec_mod

    s = {"name": "iso-test", "suite": "iso-volume",
         "defaults": {"depth": 8, "iters": 100, "lr": 0.1, "bins": 255,
                      "informative": 20, "seed": 42},
         "cells": [{"gen": "iso_volume", "log2_cells": 31,
                    "cols": [128, 2048, 65536]},
                   {"rows": 1_000_000, "cols": 100}],
         "variants": ["bonsai_cuda_depthwise", "xgb_cuda", "bonsai_depthwise"],
         "threads": [16],
         "repeats": {"default": 2, "cpu": 1}}
    cells = spec_mod.cells_of(s)
    assert [(c["rows"], c["cols"]) for c in cells[:3]] == [
        (16_777_216, 128), (1_048_576, 2048), (32_768, 65_536)]
    assert all(c["rows"] * c["cols"] == 1 << 31 for c in cells[:3])
    assert cells[0]["axis"] == "iso_volume" and cells[0]["aspect"] == 131072.0
    assert cells[3]["axis"] == "cell" and cells[3]["n_test"] == 200_000
    jobs = spec_mod.expand(s)
    assert len(jobs) == 4 * 3
    by = {(j["variant"], j["cell"]["cols"]): j for j in jobs}
    assert by[("bonsai_cuda_depthwise", 128)]["repeats"] == 2
    assert by[("bonsai_depthwise", 128)]["repeats"] == 1  # cpu policy
    # variant_iters cross-products the listed variant only.
    s2 = dict(s, variant_iters={"xgb_cuda": [50, 100]},
              cells=[{"rows": 1000, "cols": 10}], repeats=1)
    assert [(j["variant"], j["cell"]["iters"]) for j in spec_mod.expand(s2)] == [
        ("bonsai_cuda_depthwise", 100), ("xgb_cuda", 50), ("xgb_cuda", 100),
        ("bonsai_depthwise", 100)]
    try:
        spec_mod.cells_of({"cells": [{"gen": "iso_volume", "log2_cells": 31,
                                      "cols": [1000]}]})
    except ValueError:
        pass
    else:
        raise AssertionError("non-dividing cols must be rejected")


def test_bundled_specs():
    from bonsai.bench import spec as spec_mod

    names = spec_mod.bundled_specs()
    assert "iso-volume-2026-08" in names and "gpu-pareto-16M" in names
    s = spec_mod.load_spec("iso-volume-2026-08")  # bare name, no repo path
    assert s["suite"] == "iso-volume" and len(spec_mod.expand(s)) > 0
    pareto = spec_mod.load_spec("gpu-pareto-16M.json")  # suffix tolerated
    assert len(spec_mod.expand(pareto)) == 22  # the four iteration ladders
    try:
        spec_mod.load_spec("no-such-spec")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("unknown spec name must be rejected")


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


def test_data_cache():
    import pathlib

    from bonsai.bench import runners, synth

    cell = {"rows": 3000, "cols": 12, "seed": 7, "n_test": 500,
            "informative": 5}
    direct = synth.gen_data(cell["rows"], cell["cols"], cell["seed"],
                            cell["n_test"], cell["informative"])
    with tempfile.TemporaryDirectory() as td:
        first = runners.cached_gen_data(cell, td)
        files = sorted(p.name for p in pathlib.Path(td).iterdir())
        assert len(files) == 4 and not any(".tmp" in f for f in files)
        second = runners.cached_gen_data(cell, td)
        for d, a, b in zip(direct, first, second):
            assert np.array_equal(d, a) and np.array_equal(d, b)
        assert isinstance(second[0], np.memmap)
        assert second[0].dtype == np.float32


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


def test_variant_canonicalization_and_ts_guard():
    from bonsai.bench import runners
    from bonsai.bench import spec as spec_mod

    s = {"name": "t", "cells": [{"rows": 1000, "cols": 8}],
         "variants": ["bonsai_dw", "xgb"]}
    jobs = spec_mod.expand(s)
    assert [j["variant"] for j in jobs] == ["bonsai_depthwise", "xgb_hist"]
    cell = spec_mod.make_cell({}, rows=512, cols=4)
    try:
        runners.worker({"cell": cell, "variant": "bonsai_ts_depthwise",
                        "threads": 1})
    except RuntimeError as e:
        assert "airline" in str(e)
    else:
        raise AssertionError("ts arm must be rejected outside airline")


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
            try:
                driver.run_jobs([], out=str(out), suite="t", knobs={},
                                host=host, gates={"gpu_max_col": None})
            except ValueError as e:
                assert "unknown gate keys" in str(e)
            else:
                raise AssertionError("gate typos must be rejected")
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


def test_cli_plan_is_lazy():
    from bonsai.bench import cli

    with tempfile.TemporaryDirectory() as td:
        import pathlib
        p = pathlib.Path(td) / "s.json"
        p.write_text(json.dumps(
            {"name": "t", "cells": [{"rows": 1000, "cols": 10}],
             "variants": ["bonsai_depthwise", "xgb_hist"]}))
        assert cli.main(["plan", "--spec", str(p)]) == 0
    for heavy in ("xgboost", "lightgbm", "catboost"):
        assert heavy not in sys.modules, f"{heavy} imported by plan"


def test_metrics_against_sklearn():
    rng = np.random.default_rng(0)
    y = rng.random(500)
    pred = y + rng.normal(0, 0.1, 500)
    yb = (rng.random(500) > 0.5).astype(float)
    scores = yb * 0.6 + rng.random(500) * 0.4
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        roc_auc_score,
    )
    assert abs(metrics.r2(y, pred) - r2_score(y, pred)) < 1e-12
    assert abs(metrics.rmse(y, pred) - mean_squared_error(y, pred) ** 0.5) < 1e-12
    assert abs(metrics.mae(y, pred) - mean_absolute_error(y, pred)) < 1e-12
    assert abs(metrics.auc(yb, scores) - roc_auc_score(yb, scores)) < 1e-12
    # the numpy fallback must agree with sklearn, including under ties
    tied = np.round(scores, 1)
    import unittest.mock as mock
    with mock.patch.dict(sys.modules, {"sklearn.metrics": None, "sklearn": None}):
        fallback = metrics.auc(yb, tied)
    assert abs(fallback - roc_auc_score(yb, tied)) < 1e-12


def test_runlog_roundtrip():
    with tempfile.NamedTemporaryFile("r", suffix=".jsonl") as f:
        knobs = {"b": 2, "a": 1}
        row = runlog.emit_row(f.name, division="perf", suite="scaling",
                              knobs=knobs, timing_mode="in_memory",
                              variant="bonsai_dw", value=1.0, metric="r2",
                              status="ok")
        back = json.loads(open(f.name).read().splitlines()[-1])
        assert back == json.loads(json.dumps(row))
        for key in ("schema", "ts", "git_sha", "division", "suite", "cmd",
                    "timing_mode", "host", "knobs", "knobs_hash"):
            assert key in back, key
        # hash is canonical: key order must not matter
        assert runlog.knobs_hash({"a": 1, "b": 2}) == back["knobs_hash"]
    try:
        runlog.emit_row("/tmp/x.jsonl", division="nope", suite="s")
    except AssertionError:
        pass
    else:
        raise AssertionError("bad division must be rejected")


def test_bench_import_is_lazy():
    """Importing bonsai.bench must not drag in the reference libraries."""
    for heavy in ("xgboost", "lightgbm", "catboost", "openml"):
        assert heavy not in sys.modules, f"{heavy} imported eagerly"


if __name__ == "__main__":
    test_bench_import_is_lazy()
    test_gen_data_bytestable()
    test_reference_param_mappings()
    test_bonsai_core_pairs()
    test_variant_registry()
    test_spec_expansion()
    test_bundled_specs()
    test_driver_resume_and_emit()
    test_error_classification()
    test_variant_canonicalization_and_ts_guard()
    test_driver_gates_and_effective_bins()
    test_resume_is_host_scoped()
    test_timeout_and_sampler_resilience()
    test_mem_sampler()
    test_data_cache()
    test_cli_plan_is_lazy()
    test_metrics_against_sklearn()
    test_runlog_roundtrip()
    print("all bench tests passed")
