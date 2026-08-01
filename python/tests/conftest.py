"""Shared fixtures and loaders for the bonsai python test suite."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
TRAIN_CSV = REPO / "tests/data/california_housing_train.csv"
TEST_CSV = REPO / "tests/data/california_housing_test.csv"
CH_TOML = REPO / "configs/california_housing.toml"
CLI = REPO / "build/src/bonsai"

CH_PARAMS = dict(
    n_iters=200,
    learning_rate=0.05,
    max_depth=6,
    grower="depthwise",
    params={
        "tree.min_data_in_leaf": 20,
        "tree.min_child_hess": 0.001,
        "bin_mapper.max_bin": 255,
    },
)


def load_csv(path):
    """(X, y) float32 arrays from a label-first csv."""
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    return data[:, 1:], data[:, 0]  # label is column 0


@pytest.fixture(scope="session")
def toy_train():
    """California-housing train arrays (X, y)."""
    return load_csv(TRAIN_CSV)


@pytest.fixture(scope="session")
def toy_test():
    """California-housing test arrays (X, y)."""
    return load_csv(TEST_CSV)
