"""Declarative benchmark specs: cell lists and job expansion.

A spec is a JSON file (bundled under bench/specs/ for committed campaigns) that
names its cells, variants, threads, and repeats; expand() turns it into the
flat job list the driver executes.
"""

from __future__ import annotations

import json
import pathlib

from bonsai.bench import params
from bonsai.bench.variants import resolve

_SPEC_KEYS = {"name", "suite", "defaults", "cells", "variants", "threads",
              "repeats", "gates", "timeout_cap"}

# Cell knob defaults when a spec omits them: the scaling regime, single-
# sourced from params so the two cannot drift.
_CELL_DEFAULTS = {**{k: params.SCALING[k]
                     for k in ("bins", "depth", "iters", "lr", "seed",
                               "min_data_in_leaf", "lambda_l2")},
                  "informative": 20,
                  # off by default so only the SHAP suite pays for the
                  # extra phase; a spec's defaults block or a cell can
                  # set it true.
                  "contribs": False}


def bundled_specs() -> list[str]:
    """Names of the specs shipped inside the wheel (bench/specs/*.json)."""
    from importlib import resources
    d = resources.files(__package__) / "specs"
    return sorted(f.name.removesuffix(".json") for f in d.iterdir()
                  if f.name.endswith(".json"))


def load_spec(path: str | pathlib.Path) -> dict:
    """Load and validate a spec from a path or bundled name.

    Raises
    ------
    ValueError
        On unknown top-level keys, a missing name/cells/variants, or an
        unknown variant spelling.
    FileNotFoundError
        When neither a file nor a bundled spec matches.
    """
    spec = json.loads(_spec_text(path))
    unknown = set(spec) - _SPEC_KEYS
    if unknown:
        raise ValueError(f"unknown spec keys: {sorted(unknown)}")
    for req in ("name", "cells", "variants"):
        if req not in spec:
            raise ValueError(f"spec is missing {req!r}")
    for v in spec["variants"]:
        resolve(v)
    return spec


def make_cell(defaults: dict, **over) -> dict:
    """One fully-defaulted cell dict; raises ValueError without rows/cols.

    The defaults block is knob-checked: a typo'd key there would otherwise
    ride into every cell silently, which is the drift class params.py's
    docstring documents. Per-cell keys stay open (rows/cols/task/eval
    knobs are legitimately cell-level).
    """
    unknown = set(defaults) - set(_CELL_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown key(s) in spec defaults: {sorted(unknown)}; "
                         f"legal: {sorted(_CELL_DEFAULTS)}")
    c = {**_CELL_DEFAULTS, **defaults, **over}
    if "rows" not in c or "cols" not in c:
        raise ValueError(f"cell needs rows and cols: {c}")
    c.setdefault("axis", "cell")
    c.setdefault("n_test", min(c["rows"] // 5, 500_000))
    c["bins_effective"] = c["bins"]
    return c


def cells_of(spec: dict) -> list[dict]:
    """The spec's concrete cells, each fully defaulted."""
    defaults = spec.get("defaults", {})
    return [make_cell(defaults, **entry) for entry in spec["cells"]]


def expand(spec: dict, *, variants: list[str] | None = None,
           repeats: int | None = None) -> list[dict]:
    """Flat job list: [{cell, variant, threads, repeats}], cells outer so a
    sweep finishes one shape across all arms before moving on (same-shape
    rows stay adjacent in the output)."""
    # Canonicalize: aliases (bonsai_dw, xgb, ...) validate AND normalize, so
    # emitted rows and resume keys carry one spelling per arm.
    chosen = [resolve(v).name for v in (variants or spec["variants"])]
    policy = repeats if repeats is not None else spec.get("repeats", 1)
    threads = spec.get("threads", [16])
    return [{"cell": dict(cell), "variant": variant, "threads": t,
             "repeats": _repeats_for(variant, policy)}
            for cell in cells_of(spec)
            for variant in chosen
            for t in threads]


def _spec_text(name_or_path: str | pathlib.Path) -> str:
    """A filesystem path wins; a bare name resolves to a bundled spec
    (bench/specs/<name>.json), so wheel installs run the committed
    campaigns without a repo checkout."""
    p = pathlib.Path(name_or_path)
    if p.exists():
        return p.read_text()
    from importlib import resources
    stem = str(name_or_path).removesuffix(".json")
    res = resources.files(__package__) / "specs" / f"{stem}.json"
    if res.is_file():
        return res.read_text()
    raise FileNotFoundError(
        f"no spec file or bundled spec named {str(name_or_path)!r}; "
        "bundled names: " + ", ".join(bundled_specs()))


def _repeats_for(variant: str, policy: dict | int) -> int:
    """Per-variant repeat count from an int or {device/default: n} policy."""
    if isinstance(policy, int):
        return policy
    device = resolve(variant).device
    return int(policy.get(device, policy.get("default", 1)))
