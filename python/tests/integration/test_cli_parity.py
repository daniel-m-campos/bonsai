"""Integration: the python module and the CLI agree on the toy dataset."""

from __future__ import annotations

import pathlib
import subprocess
import tempfile

import bonsai
import numpy as np
from conftest import CH_PARAMS, CH_TOML, CLI


def test_parity_with_cli(toy_train, toy_test):
    """Same config through the native module and the CLI must agree."""
    Xtr, ytr = toy_train
    Xte, _ = toy_test
    py_pred = bonsai.BonsaiRegressor(**CH_PARAMS).fit(Xtr, ytr).predict(Xte)

    with tempfile.TemporaryDirectory() as td:
        model = pathlib.Path(td) / "m.msgpack"
        preds = pathlib.Path(td) / "p.csv"
        subprocess.run(
            [CLI, "fit", "-c", CH_TOML, "--set", "dispatch.grower_name=depthwise",
             "--model", model],
            check=True, capture_output=True,
        )
        subprocess.run(
            [CLI, "predict", "-c", CH_TOML, "--model", model, "--out", preds],
            check=True, capture_output=True,
        )
        cli_pred = np.loadtxt(preds, skiprows=1, dtype=np.float32)

    np.testing.assert_allclose(py_pred, cli_pred, rtol=0, atol=2e-4)
