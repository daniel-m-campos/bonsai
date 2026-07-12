#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas>=2.2",
# ]
# ///
"""Fetch the UCI Year Prediction MSD dataset.

Writes tests/data/year_prediction_msd_{train,test}.csv with `label`
(release year, 1922-2011) as the first column and 90 timbre features
following. Uses the UCI-recommended split (first 463,715 rows train,
last 51,630 test) to avoid the producer effect.

Idempotent: skips download + split if outputs already exist.

Run via uv:
    uv run scripts/fetch_year_msd.py
"""

import csv
import io
import pathlib
import urllib.request
import zipfile

import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "tests" / "data"
URL = "https://archive.ics.uci.edu/static/public/203/yearpredictionmsd.zip"
ARCHIVE_MEMBER = "YearPredictionMSD.txt"

TRAIN_ROWS = 463_715
TEST_ROWS = 51_630
N_FEATURES = 90


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_path = DATA_DIR / "year_prediction_msd_train.csv"
    test_path = DATA_DIR / "year_prediction_msd_test.csv"

    if train_path.exists() and test_path.exists():
        print(f"already present: {train_path}")
        print(f"already present: {test_path}")
        return 0

    print(f"downloading {URL} (~200 MB compressed, ~440 MB uncompressed) ...", flush=True)
    with urllib.request.urlopen(URL) as resp:
        zip_bytes = resp.read()

    print(f"extracting {ARCHIVE_MEMBER} ...", flush=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        with zf.open(ARCHIVE_MEMBER) as fp:
            df = pd.read_csv(fp, header=None)

    # Column 0 is the year (regression target); columns 1-90 are timbre features.
    df.columns = ["label", *[f"f{i}" for i in range(N_FEATURES)]]

    if len(df) != TRAIN_ROWS + TEST_ROWS:
        raise RuntimeError(
            f"unexpected row count: got {len(df)}, "
            f"expected {TRAIN_ROWS + TEST_ROWS}"
        )

    train = df.iloc[:TRAIN_ROWS]
    test = df.iloc[TRAIN_ROWS:]

    train.to_csv(train_path, index=False, quoting=csv.QUOTE_MINIMAL)
    test.to_csv(test_path, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"wrote {len(train):>7} rows -> {train_path}")
    print(f"wrote {len(test):>7} rows -> {test_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
