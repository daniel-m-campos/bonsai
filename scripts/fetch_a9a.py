#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Fetch a9a (Adult, preprocessed to 123 binary features, LIBSVM format)
for the sparse-input benchmark. 32,561 train / 16,281 test rows.
Labels -1/+1 are rewritten to 0/1. Run: uv run scripts/fetch_a9a.py"""

import pathlib
import urllib.request

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "tests" / "data"
BASE = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"


def fetch(name: str, out: pathlib.Path) -> None:
    req = urllib.request.Request(BASE + name, headers={"User-Agent": "bonsai"})
    with urllib.request.urlopen(req) as resp:  # noqa: S310
        text = resp.read().decode()
    lines = []
    for line in text.splitlines():
        if not line:
            continue
        label, _, rest = line.partition(" ")
        lines.append(("1" if label.strip() == "+1" else "0") + " " + rest)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} rows -> {out}")


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train = DATA_DIR / "a9a_train.libsvm"
    test = DATA_DIR / "a9a_test.libsvm"
    if train.exists() and test.exists():
        print(f"already present: {train}")
        return 0
    fetch("a9a", train)
    fetch("a9a.t", test)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
