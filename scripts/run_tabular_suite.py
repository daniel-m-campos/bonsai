"""Shim: the Grinsztajn suite lives in bonsai.bench.grinsztajn (decision 69).

    PYTHONPATH=build/python python3 scripts/run_tabular_suite.py out.jsonl
    PYTHONPATH=build/python python3 scripts/run_tabular_suite.py out.jsonl --report
or, equivalently: python -m bonsai.bench.grinsztajn out.jsonl
"""
from bonsai.bench.grinsztajn import main

if __name__ == "__main__":
    main()
