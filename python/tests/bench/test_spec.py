"""Tests for bonsai.bench.spec (loading, generators, expansion)."""

from __future__ import annotations

import pytest
from bonsai.bench import spec as spec_mod


def test_spec_expansion():
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
    with pytest.raises(ValueError):
        spec_mod.cells_of({"cells": [{"gen": "iso_volume", "log2_cells": 31,
                                      "cols": [1000]}]})


def test_bundled_specs():
    from bonsai.bench import spec as spec_mod

    names = spec_mod.bundled_specs()
    assert "iso-volume-2026-08" in names and "gpu-pareto-16M" in names
    s = spec_mod.load_spec("iso-volume-2026-08")  # bare name, no repo path
    assert s["suite"] == "iso-volume" and len(spec_mod.expand(s)) > 0
    pareto = spec_mod.load_spec("gpu-pareto-16M.json")  # suffix tolerated
    assert len(spec_mod.expand(pareto)) == 32  # the six iteration ladders
    with pytest.raises(FileNotFoundError):
        spec_mod.load_spec("no-such-spec")
