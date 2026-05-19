# Benchmarks

## Reference-library comparison

bonsai vs xgboost, lightgbm, catboost on a CSV regression dataset. All
four libraries see the same hyperparameters by reading them out of the
bonsai TOML config.

The Python sidecar is two single-file scripts with [PEP 723][pep723]
inline metadata. They run via [uv][uv] with no manual virtualenv setup.
**Python 3.12 is required** (pinned in each script's header; uv enforces
it).

[pep723]: https://peps.python.org/pep-0723/
[uv]: https://docs.astral.sh/uv/

### Setup

Install uv once:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Run

From the repo root:

```
# 1. Fetch the toy dataset (writes tests/data/california_housing_{train,test}.csv).
uv run scripts/fetch_toy.py

# 2. Build bonsai.
make build

# 3. Run all four libraries and write the report.
uv run scripts/compare.py --config configs/california_housing.toml
```

Results land in `benchmarks/results/<config-stem>.{json,md}`. The Python
script also prints the markdown table to stdout when it finishes.

The first `uv run` of each script downloads its declared dependencies
into uv's shared cache. Subsequent runs reuse the cache and are fast.

### Microbenchmarks

`bench_split.cpp` here is a google-benchmark-style microbench of the
histogram splitter, separate from the end-to-end Python harness.
