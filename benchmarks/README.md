# Benchmarks

This directory is the evidence: one study writeup per closed round, the committed rows behind them under `results/`, and the standings registry `standings.json`.

Three pages carry everything else, and this file deliberately repeats none of it:

- How to run a suite, a custom ladder, or a pod campaign: [Running the benchmarks](../docs/use/benchmarks.md).
- The normative rules (divisions, suites, primary metrics, timing modes, the result-row schema, the standings policy): [the benchmark protocol](../docs/method/benchmark-protocol.md).
- Renting, accepting, and driving a GPU pod: [the RunPod runbook](../docs/ops/runpod-runbook.md).

One path lives nowhere else: the CLI-compare sidecar, which runs bonsai against XGBoost, LightGBM, and CatBoost on a CSV dataset with all four reading their hyperparameters out of the same bonsai TOML config. It is a [PEP 723](https://peps.python.org/pep-0723/) script, so [uv](https://docs.astral.sh/uv/) supplies its own Python 3.12 and dependencies:

```bash
uv run scripts/fetch_toy.py                                   # toy dataset
make build                                                    # the bonsai binary
uv run scripts/compare.py --config configs/california_housing.toml
```

Results land in `results/<config-stem>.{json,md}`, and the markdown table also prints to stdout.
