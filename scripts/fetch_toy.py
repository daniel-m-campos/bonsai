#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=1.26",
#     "pandas>=2.2",
#     "scikit-learn>=1.4",
# ]
# ///
"""Fetch California Housing as the toy dataset for end-to-end runs.

Writes tests/data/california_housing_{train,test}.csv with the label
as the first column (matches DataConfig.label_column = 0 default).

Run via uv:
    uv run scripts/fetch_toy.py
"""

import csv
import pathlib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "tests" / "data"


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    bundle = fetch_california_housing(as_frame=True)
    df = bundle.frame  # last column is the target ("MedHouseVal")

    target_col = bundle.target.name
    feature_cols = [c for c in df.columns if c != target_col]

    df = df[[target_col, *feature_cols]].rename(columns={target_col: "label"})

    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train_path = DATA_DIR / "california_housing_train.csv"
    test_path = DATA_DIR / "california_housing_test.csv"

    train.to_csv(train_path, index=False, quoting=csv.QUOTE_MINIMAL)
    test.to_csv(test_path, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"wrote {len(train)} rows -> {train_path}")
    print(f"wrote {len(test)} rows  -> {test_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
