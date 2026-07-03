#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Fetch UCI Covertype for the multiclass benchmark.

581,012 rows, 54 numeric features, 7 forest cover classes. Label moves
to column 0 and is shifted 1..7 -> 0..6. Train = first 500k rows,
test = the rest. Idempotent. Run: uv run scripts/fetch_covtype.py
"""

import gzip
import pathlib
import urllib.request

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "tests" / "data"
URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
       "covtype/covtype.data.gz")
N_TRAIN = 500_000
N_FEATURES = 54


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_path = DATA_DIR / "covtype_train.csv"
    test_path = DATA_DIR / "covtype_test.csv"
    if train_path.exists() and test_path.exists():
        print(f"already present: {train_path}")
        return 0

    header = "label," + ",".join(f"f{i}" for i in range(N_FEATURES)) + "\n"
    print(f"downloading {URL} (~11 MB) ...", flush=True)
    req = urllib.request.Request(URL, headers={"User-Agent": "bonsai-fetch"})
    n = 0
    with (
        urllib.request.urlopen(req) as resp,  # noqa: S310
        gzip.GzipFile(fileobj=resp) as gz,
        open(train_path, "w") as ftr,
        open(test_path, "w") as fte,
    ):
        ftr.write(header)
        fte.write(header)
        for raw in gz:
            cols = raw.decode().strip().split(",")
            label = int(cols[-1]) - 1  # 1..7 -> 0..6
            line = f"{label}," + ",".join(cols[:-1]) + "\n"
            (ftr if n < N_TRAIN else fte).write(line)
            n += 1
    print(f"wrote {min(n, N_TRAIN)} train / {max(0, n - N_TRAIN)} test rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
