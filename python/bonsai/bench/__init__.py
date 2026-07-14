"""bonsai.bench: the benchmarking harness behind bonsai's published claims.

Install the reference libraries with ``pip install bonsai-gbt[bench]``, then
reproduce the standings table yourself::

    python -m bonsai.bench.grinsztajn out.jsonl
    python -m bonsai.bench.grinsztajn out.jsonl --report

Two divisions, both run "closed" (matched knobs, no per-model tuning):
``quality`` (accuracy standings; timing never citable) and ``perf``
(latency/throughput/memory; every row labels its timing_mode). The normative
protocol lives in docs/method/benchmark-protocol.md on the documentation site.

Importing this package pulls numpy only; xgboost/lightgbm/catboost/openml/
sklearn are imported lazily by the variants and suites that need them.
"""
