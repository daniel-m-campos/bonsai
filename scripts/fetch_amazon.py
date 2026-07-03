#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["scikit-learn>=1.4", "pandas>=2.2"]
# ///
"""Fetch the Amazon employee access dataset (OpenML 4135) for the
categorical benchmark: 32,769 rows, 9 high-cardinality categorical
features (integer IDs: RESOURCE ~7k distinct, MGR_ID ~4k, ...), binary
target ACTION. Split 80/20 stratified, label first.

Run: uv run scripts/fetch_amazon.py
"""

import pathlib

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "tests" / "data"


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_path = DATA_DIR / "amazon_train.csv"
    test_path = DATA_DIR / "amazon_test.csv"
    if train_path.exists() and test_path.exists():
        print(f"already present: {train_path}")
        return 0

    print("fetching OpenML Amazon_employee_access (id 4135) ...", flush=True)
    ds = fetch_openml(data_id=4135, as_frame=True, parser="auto")
    X = ds.data.astype("int64")
    y = ds.target.astype("int64")

    df = pd.concat([y.rename("label"), X], axis=1)
    train, test = train_test_split(df, test_size=0.2, random_state=42,
                                   stratify=df["label"])
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    for name, col in X.items():
        print(f"  {name}: {col.nunique()} distinct")
    print(f"wrote {len(train)} train / {len(test)} test rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
