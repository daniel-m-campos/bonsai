"""The dataset registry: every dataset bonsai benchmarks on, with provenance.

Self-contained by design (stdlib + numpy only, no sibling imports): the repo
fetch scripts load this file directly via importlib so data can be fetched
before the native module is built.

Tiers:
- test-pin        CI-load-bearing; fetched by the standalone scripts the CI
                  cache keys hash (scripts/fetch_toy.py, fetch_amazon.py).
- quality-external  third-party-selected accuracy suites (Grinsztajn) and
                  public single datasets used by quality studies.
- quality-smoke   the internal ten-dataset campaign (fast local regression
                  check; not citable as standings, decision 68).
- perf-scale      large real datasets for latency/throughput ladders.
- perf-synthetic  the Friedman-1 generator (bonsai.bench.synth; provenance
                  in its docstring and docs/method/benchmark-protocol.md).
"""

from __future__ import annotations

import dataclasses
import gzip
import os
import pathlib
import urllib.request


def data_root() -> pathlib.Path:
    """tests/data in a repo checkout; BONSAI_BENCH_DATA or a user cache dir
    when running from an installed wheel."""
    env = os.environ.get("BONSAI_BENCH_DATA")
    if env:
        return pathlib.Path(env)
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "tests" / "data"
        if cand.is_dir():
            return cand
    return pathlib.Path.home() / ".cache" / "bonsai-bench"


@dataclasses.dataclass(frozen=True)
class Dataset:
    name: str
    tier: str
    task: str
    source: str
    license_note: str
    split: str
    files: tuple[str, ...]


REGISTRY = {
    "tiny": Dataset(
        "tiny", "test-pin", "reg", "committed in-repo (tests/data/tiny.csv)",
        "project-generated", "single file", ("tiny.csv",)),
    "california": Dataset(
        "california", "test-pin", "reg",
        "sklearn fetch_california_housing (StatLib)", "public domain",
        "80/20 train_test_split(random_state=42) by scripts/fetch_toy.py",
        ("california_housing_train.csv", "california_housing_test.csv")),
    "amazon": Dataset(
        "amazon", "test-pin", "binary", "OpenML data id 4135",
        "Kaggle competition data, research use",
        "80/20 stratified random_state=42 by scripts/fetch_amazon.py",
        ("amazon_train.csv", "amazon_test.csv")),
    "grinsztajn": Dataset(
        "grinsztajn", "quality-external", "mixed",
        "OpenML suites 297/298/299/304 (Grinsztajn et al. 2022)",
        "per-dataset OpenML licenses",
        "10k-row train cap, 3 seeded splits (bonsai.bench.grinsztajn)",
        ()),
    "campaign10": Dataset(
        "campaign10", "quality-smoke", "mixed",
        "ten OpenML datasets (decisions 56-57; benchmarks/quality-campaign-2026-07.md)",
        "per-dataset OpenML licenses",
        "fetched at runtime via OpenML by the campaign tooling", ()),
    "a9a": Dataset(
        "a9a", "quality-external", "binary",
        "LIBSVM binary collection (Adult, 123 binary features)",
        "UCI Adult derivative", "upstream fixed train/test split",
        ("a9a_train.libsvm", "a9a_test.libsvm")),
    "covtype": Dataset(
        "covtype", "quality-external", "multiclass",
        "UCI Covertype", "UCI, research use",
        "first 500k rows train, remainder test",
        ("covtype_train.csv", "covtype_test.csv")),
    "higgs": Dataset(
        "higgs", "perf-scale", "binary",
        "UCI HIGGS (first 550k of 11M rows)", "UCI, CC BY 4.0",
        "first 500k train, next 50k test",
        ("higgs_train.csv", "higgs_test.csv")),
    "year_msd": Dataset(
        "year_msd", "perf-scale", "reg",
        "UCI YearPredictionMSD", "UCI, research use",
        "UCI-recommended split: first 463,715 train, last 51,630 test "
        "(avoids the producer effect)",
        ("year_prediction_msd_train.csv", "year_prediction_msd_test.csv")),
    "friedman1": Dataset(
        "friedman1", "perf-synthetic", "reg",
        "bonsai.bench.synth.gen_data (generalized Friedman 1991)",
        "project-generated",
        "deterministic in (seed, rows, cols); see synth.py provenance", ()),
}


def paths(name: str) -> list[pathlib.Path]:
    return [data_root() / f for f in REGISTRY[name].files]


def is_fetched(name: str) -> bool:
    ps = paths(name)
    return bool(ps) and all(p.exists() for p in ps)


# ---- fetchers (stdlib streaming; pandas only where the source demands it) --

_UA = {"User-Agent": "bonsai"}


def _fetch_a9a(root: pathlib.Path) -> None:
    base = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"
    for src, dst in (("a9a", "a9a_train.libsvm"), ("a9a.t", "a9a_test.libsvm")):
        req = urllib.request.Request(base + src, headers=_UA)
        with urllib.request.urlopen(req) as resp:
            text = resp.read().decode()
        lines = []
        for line in text.splitlines():
            if not line:
                continue
            label, _, rest = line.partition(" ")
            lines.append(("1" if label.strip() == "+1" else "0") + " " + rest)
        (root / dst).write_text("\n".join(lines) + "\n")


def _fetch_covtype(root: pathlib.Path) -> None:
    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "covtype/covtype.data.gz")
    n_train, n_feat = 500_000, 54
    header = "label," + ",".join(f"f{i}" for i in range(n_feat)) + "\n"
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req) as resp, gzip.open(resp, "rt") as gz, \
            (root / "covtype_train.csv").open("w") as tr, \
            (root / "covtype_test.csv").open("w") as te:
        tr.write(header)
        te.write(header)
        for i, line in enumerate(gz):
            cols = line.rstrip("\n").split(",")
            # label moves to column 0 and shifts 1..7 -> 0..6
            row = str(int(cols[-1]) - 1) + "," + ",".join(cols[:-1]) + "\n"
            (tr if i < n_train else te).write(row)


def _fetch_higgs(root: pathlib.Path) -> None:
    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "00280/HIGGS.csv.gz")
    n_train, n_test, n_feat = 500_000, 50_000, 28
    header = "label," + ",".join(f"f{i}" for i in range(n_feat)) + "\n"
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req) as resp, gzip.open(resp, "rt") as gz, \
            (root / "higgs_train.csv").open("w") as tr, \
            (root / "higgs_test.csv").open("w") as te:
        tr.write(header)
        te.write(header)
        for i, line in enumerate(gz):
            if i >= n_train + n_test:
                break
            (tr if i < n_train else te).write(line)


def _fetch_year_msd(root: pathlib.Path) -> None:
    import io
    import zipfile
    url = "https://archive.ics.uci.edu/static/public/203/yearpredictionmsd.zip"
    n_train, n_feat = 463_715, 90
    header = "label," + ",".join(f"f{i}" for i in range(n_feat)) + "\n"
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req) as resp:
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf, \
            zf.open("YearPredictionMSD.txt") as member, \
            io.TextIOWrapper(member) as text, \
            (root / "year_prediction_msd_train.csv").open("w") as tr, \
            (root / "year_prediction_msd_test.csv").open("w") as te:
        tr.write(header)
        te.write(header)
        for i, line in enumerate(text):
            (tr if i < n_train else te).write(line)


_FETCHERS = {"a9a": _fetch_a9a, "covtype": _fetch_covtype,
             "higgs": _fetch_higgs, "year_msd": _fetch_year_msd}


def fetch(name: str, force: bool = False) -> list[pathlib.Path]:
    """Fetch a registry dataset into data_root(); idempotent. test-pin
    datasets are owned by the standalone CI scripts; grinsztajn/campaign10
    fetch at suite runtime via OpenML; friedman1 is generated, not fetched."""
    ds = REGISTRY[name]
    if name not in _FETCHERS:
        raise ValueError(
            f"{name} ({ds.tier}) is not fetched here: {ds.split}")
    root = data_root()
    root.mkdir(parents=True, exist_ok=True)
    if not force and is_fetched(name):
        return paths(name)
    _FETCHERS[name](root)
    return paths(name)


if __name__ == "__main__":
    import sys
    for arg in sys.argv[1:] or ["--list"]:
        if arg == "--list":
            for d in REGISTRY.values():
                state = "fetched" if is_fetched(d.name) else (
                    "n/a" if not d.files else "missing")
                print(f"{d.name:12s} {d.tier:17s} {d.task:10s} [{state}] "
                      f"{d.source}")
        else:
            print(fetch(arg))
