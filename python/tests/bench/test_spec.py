"""Tests for bonsai.bench.spec (loading, generators, expansion)."""

from __future__ import annotations

import pytest
from bonsai.bench import spec as spec_mod

_GPU_ARMS = ["bonsai_cuda_depthwise", "bonsai_cuda_leafwise",
             "bonsai_cuda_levelwise", "xgb_cuda", "lgbm_cuda", "catboost_gpu"]
_CPU_ARMS = ["bonsai_depthwise", "bonsai_leafwise", "bonsai_levelwise",
             "xgb_hist", "lgbm_cpu", "catboost_cpu"]


def test_spec_expansion():
    s = {"name": "cell-test", "suite": "cells",
         "defaults": {"depth": 8, "iters": 100, "lr": 0.1, "bins": 255,
                      "informative": 20, "seed": 42},
         "cells": [{"rows": 16_777_216, "cols": 128},
                   {"rows": 1_048_576, "cols": 2048},
                   {"rows": 32_768, "cols": 65_536},
                   {"rows": 1_000_000, "cols": 100}],
         "variants": ["bonsai_cuda_depthwise", "xgb_cuda", "bonsai_depthwise"],
         "threads": [16],
         "repeats": {"default": 2, "cpu": 1}}
    cells = spec_mod.cells_of(s)
    assert [(c["rows"], c["cols"]) for c in cells[:3]] == [
        (16_777_216, 128), (1_048_576, 2048), (32_768, 65_536)]
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
    with pytest.raises(ValueError):
        spec_mod.cells_of({"cells": [{"cols": 1000}]})


def test_spec_expansion_exclude_variants():
    s = {"name": "exclude-test", "cells": [
            {"rows": 1000, "cols": 10},
            {"rows": 2000, "cols": 20, "exclude_variants": ["xgb_cuda"]}],
         "variants": ["bonsai_cuda_depthwise", "xgb_cuda"],
         "threads": [16], "repeats": 1}
    jobs = spec_mod.expand(s)
    assert len(jobs) == 3  # 2 variants at cols=10, 1 at cols=20
    by_cols = {(j["cell"]["cols"], j["variant"]) for j in jobs}
    assert (20, "xgb_cuda") not in by_cols
    assert (20, "bonsai_cuda_depthwise") in by_cols
    # exclude_variants never rides into the emitted cell dict.
    assert all("exclude_variants" not in j["cell"] for j in jobs)


# The redesigned standings axes (decision 103): scenario name -> the cell it
# measures, the arms it runs, and how many jobs that expands to.
STANDINGS_AXES = {
    "gpu-tall": ((16_777_216, 128), _GPU_ARMS, 6),
    "gpu-wide": ((131_072, 16_384), _GPU_ARMS, 6),
    "gpu-extreme": ((16_777_216, 1024), _GPU_ARMS, 6),
    "cpu-tall": ((2_097_152, 128), _CPU_ARMS, 6),
    "cpu-wide": ((16_384, 16_384), _CPU_ARMS, 6),
}


def test_bundled_specs():
    names = spec_mod.bundled_specs()
    assert names == ["cpu-tall", "cpu-wide", "gpu-early-stop", "gpu-extreme",
                     "gpu-tall", "gpu-wide", "scaling-bins", "scaling-cols",
                     "scaling-rows", "scaling-threads"]
    s = spec_mod.load_spec("gpu-tall")  # bare name, no repo path
    assert s["suite"] == "gpu-tall" and len(spec_mod.expand(s)) == 6
    wide = spec_mod.load_spec("gpu-wide.json")  # suffix tolerated
    assert len(spec_mod.expand(wide)) == 6
    with pytest.raises(FileNotFoundError):
        spec_mod.load_spec("no-such-spec")


@pytest.mark.parametrize("name", sorted(STANDINGS_AXES))
def test_standings_axis_expansion(name):
    """Each scenario axis is one cell across its plane's six arms."""
    cell, arms, n_jobs = STANDINGS_AXES[name]
    s = spec_mod.load_spec(name)
    assert s["suite"] == name and s["variants"] == arms
    cells = spec_mod.cells_of(s)
    assert [(c["rows"], c["cols"]) for c in cells] == [cell]
    assert cells[0]["depth"] == 8 and cells[0]["iters"] == 100
    jobs = spec_mod.expand(s)
    assert len(jobs) == n_jobs
    assert {j["variant"] for j in jobs} == set(arms)


def test_gpu_extreme_runs_with_the_memory_gate_off():
    """The extreme axis exists to publish the OOM boundary, so the driver's
    memory gates must not pre-empt it: a skipped row is not a measurement."""
    s = spec_mod.load_spec("gpu-extreme")
    assert s["gates"] == {"mem_gate": "off", "gpu_max_cols": None}
    assert s["timeout_cap"] == 10800
    assert s["repeats"] == {"default": 1}


def test_gpu_early_stop_expansion():
    """Three eval-mode arms at the gpu-tall cell, six GPU variants each."""
    s = spec_mod.load_spec("gpu-early-stop")
    cells = spec_mod.cells_of(s)
    assert [c["eval_mode"] for c in cells] == ["off", "eval", "stop"]
    assert all((c["rows"], c["cols"]) == (16_777_216, 128) for c in cells)
    stop = cells[-1]
    assert (stop["iters"], stop["lr"], stop["patience"]) == (2000, 0.05, 50)
    jobs = spec_mod.expand(s)
    assert len(jobs) == 3 * 6
    assert all(j["repeats"] == 2 for j in jobs)
