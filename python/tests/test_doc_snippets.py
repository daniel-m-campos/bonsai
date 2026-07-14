"""Executes the ```python fences in user-facing docs so the documented API
cannot drift from the real one: run with PYTHONPATH=build/python python
python/tests/test_doc_snippets.py (wired into make python-test).

Snippets are illustrative, not self-contained, so each runs against a prelude
namespace supplying the conventional names (X_train, y_train, grid, ...) on
synthetic data, in a temp cwd so model.msgpack writes land nowhere real.
"""

from __future__ import annotations

import os
import pathlib
import re
import tempfile

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
DOCS = [REPO / "README.md", *sorted((REPO / "docs" / "use").glob("*.md"))]
FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _prelude() -> dict:
    import bonsai

    rng = np.random.default_rng(0)
    n = 800
    X = rng.random((n, 6), dtype=np.float32)
    y = (X[:, 0] * 2 + rng.normal(0, 0.2, n)).astype(np.float32)
    half = n // 2
    return {
        # Docs conventionally import once in the first fence; later fences
        # are fragments that assume it.
        "bonsai": bonsai, "np": np,
        "X": X, "y": y,
        "X_train": X[:half], "y_train": y[:half],
        "X_valid": X[half:], "y_valid": y[half:],
        "X_test": X[half:], "Xv": X[half:], "yv": y[half:],
        "w": np.ones(half, dtype=np.float32),
        "grid": [
            [("booster.n_iters", "5")],
            [("booster.n_iters", "5"), ("tree.max_depth", "3")],
        ],
    }


def test_doc_python_snippets():
    total = 0
    for doc in DOCS:
        fences = FENCE.findall(doc.read_text())
        assert fences, f"{doc.relative_to(REPO)}: expected python fences"
        for i, code in enumerate(fences):
            # Speed: cap illustrative iteration counts; the point is that the
            # API calls resolve and run, not the model quality.
            code = code.replace("n_iters=200", "n_iters=8")
            ns = _prelude()
            cwd = os.getcwd()
            with tempfile.TemporaryDirectory() as td:
                os.chdir(td)
                try:
                    exec(compile(code, f"{doc.name}:fence{i}", "exec"), ns)
                except Exception as e:
                    raise AssertionError(
                        f"{doc.relative_to(REPO)} python fence #{i} failed: {e!r}\n{code}"
                    ) from e
                finally:
                    os.chdir(cwd)
            total += 1
    print(f"doc snippets: {total} python fences executed across {len(DOCS)} files")


if __name__ == "__main__":
    test_doc_python_snippets()
    print("all doc snippet tests passed")
