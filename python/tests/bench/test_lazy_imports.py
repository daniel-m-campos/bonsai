"""Import-laziness contract of bonsai.bench."""

from __future__ import annotations

import subprocess
import sys


def test_bench_import_is_lazy():
    """Importing bonsai.bench must not drag in the reference libraries.

    A fresh interpreter makes the check order-independent.
    """
    code = ("import sys, bonsai.bench\n"
            "for heavy in ('xgboost', 'lightgbm', 'catboost', 'openml'):\n"
            "    assert heavy not in sys.modules, heavy\n")
    subprocess.run([sys.executable, "-c", code], check=True)
