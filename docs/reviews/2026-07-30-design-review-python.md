# Design review (lite): the Python package

2026-07-30. Trigger: the first external user (Daniel, at work) pip-installed cleanly but had to read source to run the benchmarks, and the source he landed in was an 849-line `__init__.py`. Scope: `python/bonsai/` only; the native module and `bench/` internals are out of scope. The restructure this review recommends ships in the same PR.

## Findings

**1. `__init__.py` was an implementation home, not a facade.** 849 lines holding input coercion, the xgboost-compat translation tables, and all three estimator classes at package top level; it grew 590 to 849 in a single PR (the xgboost drop-in surface) and every future compat addition would land in the same file. Resolution: split into `_coerce.py` (input coercion), `_compat.py` (xgboost translation tables and argument shapes), and `estimators.py` (the three classes); `__init__.py` shrinks to a ~50-line public surface. Flat modules, not a subpackage: the CMake dev-tree copy globs `python/bonsai/*.py` flat (CMakeLists ~291-310), the wheel takes the directory wholesale (`wheel.packages`), and ~1000 package lines do not warrant a second level. Public API unchanged (same nine `__all__` names); old pickles resolve through the re-export and were verified to load and predict bit-identically across the split.

**2. The alias keyword list lives in three places, deliberately.** The base `__init__` and both subclass `__init__`s each spell out the full xgboost-alias parameter list, because sklearn's `clone`/`get_params` contract and the signature guard test (`test_bindings.py`, "maintained in three places") require explicit introspectable signatures. Accepted cost, kept as is; the guard test enforces sync.

**3. CI gap: the compat tests were not in `make python-test`.** The target ran four test files; `test_xgb_compat.py` (added with the drop-in surface) was only exercised when someone ran pytest by hand. Fixed in this PR: the target now runs all five.

**4. The trigger itself: benchmark running was undocumented where users look.** `benchmarks/README.md` calls itself the operational how-to-run companion yet never shows the `python -m bonsai.bench.<suite>` form; `bonsai.bench.airline` was runnable but documented nowhere; no page put the installed-wheel path (`pip install "bonsai-gbt[bench]"`) beside the source-tree path (`make python` + `PYTHONPATH=build/python`); the dataset fetcher (`python -m bonsai.bench.datasets --list`) was buried in a provenance paragraph. Fixed in this PR: `docs/use/benchmarks.md` (the two worlds side by side plus a suite table), an airline row in the benchmark-protocol suite table, and pointers from benchmarks/README, install, and the API tour.

**5. What is already right, kept.** `encoding.py` is the module-hygiene model (standalone, numpy-only, one class, own `__all__`); `bench/` is a proper subpackage with per-suite `main()`s and a lazy-import contract pinned by `test_bench.py` (importing `bonsai.bench` must not pull xgboost/lightgbm/catboost); sklearn imports stay lazy inside `__sklearn_tags__` so `import bonsai` never needs sklearn.

## Non-findings

Looked at and left alone: the duck-typed sklearn contract (no `BaseEstimator` subclassing; keeps sklearn out of the runtime deps and is test-pinned), the `params=` dotted-key escape hatch (the config system is the API, aliases are conveniences), and `bench/`'s flat module layout (each suite is one file with its own entry point; splitting further would separate code from its single caller).
