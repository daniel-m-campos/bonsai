# tests/data: the shared data root

This directory serves two masters, and the split matters:

| tier | files | fetched by | notes |
|---|---|---|---|
| test-pin (CI-load-bearing) | `tiny.csv` (committed), `california_housing_*.csv`, `amazon_*.csv` | `scripts/fetch_toy.py`, `scripts/fetch_amazon.py` (standalone; CI cache keys hash these scripts) | C++ eval-baseline pin, Python binding tests, encoder quality pins. Do not move or rename. |
| quality-external | `a9a_*.libsvm`, `covtype_*.csv` | `python3 scripts/fetch_<name>.py` (registry wrappers) | benchmark-only, gitignored |
| perf-scale | `higgs_*.csv`, `year_prediction_msd_*.csv` | same | large (365/399 MB), fetched on demand |

The machine-readable truth is the registry: `python/bonsai/bench/datasets.py`
(run `python3 python/bonsai/bench/datasets.py --list` pre-build via the fetch
shims, or `python -m bonsai.bench.datasets --list` with the module built).
Everything except `tiny.csv` is gitignored; the Grinsztajn standings suite
and the internal campaign fetch straight from OpenML at runtime and never
land here. Dataset provenance and tier definitions:
docs/method/benchmark-protocol.md.
