#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Fetch a HIGGS subset for the binary-classification benchmark.

Streams the UCI HIGGS gzip (11M rows) and stops after the first
550,000 rows (~130 MB of the 2.6 GB transferred), writing
tests/data/higgs_{train,test}.csv with `label` first and f0..f27
following. Train = first 500k, test = next 50k.

Idempotent: skips if outputs exist. Run: uv run scripts/fetch_higgs.py
"""

import gzip
import pathlib
import urllib.request

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "tests" / "data"
URL = "https://archive.ics.uci.edu/static/public/280/higgs.zip"
# The zip holds HIGGS.csv.gz; UCI also serves the gz directly:
GZ_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"

N_TRAIN = 500_000
N_TEST = 50_000
N_FEATURES = 28


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_path = DATA_DIR / "higgs_train.csv"
    test_path = DATA_DIR / "higgs_test.csv"
    if train_path.exists() and test_path.exists():
        print(f"already present: {train_path}")
        return 0

    header = "label," + ",".join(f"f{i}" for i in range(N_FEATURES)) + "\n"
    print(f"streaming {GZ_URL} (stops after {N_TRAIN + N_TEST} rows) ...",
          flush=True)
    req = urllib.request.Request(GZ_URL, headers={"User-Agent": "bonsai-fetch"})
    with (
        urllib.request.urlopen(req) as resp,  # noqa: S310
        gzip.GzipFile(fileobj=resp) as gz,
        open(train_path, "w") as ftr,
        open(test_path, "w") as fte,
    ):
        ftr.write(header)
        fte.write(header)
        for i, raw in enumerate(gz):
            if i >= N_TRAIN + N_TEST:
                break
            line = raw.decode()
            (ftr if i < N_TRAIN else fte).write(line)
    print(f"wrote {N_TRAIN} rows -> {train_path}")
    print(f"wrote {N_TEST} rows -> {test_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
