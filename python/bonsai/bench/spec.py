"""Declarative benchmark specs: cell lists, ladder generators, job expansion.

A spec is a JSON file (bundled under bench/specs/ for committed campaigns) that
names its cells, variants, threads, and repeats; expand() turns it into the
flat job list the driver executes. Generators cover the recurring ladder
shapes so campaigns stop hand-writing driver scripts.
"""

from __future__ import annotations

import json
import pathlib

from bonsai.bench import params
from bonsai.bench.variants import resolve

_SPEC_KEYS = {"name", "suite", "defaults", "cells", "variants", "threads",
              "repeats", "gates", "timeout_cap", "variant_iters"}

# Cell knob defaults when a spec omits them: the scaling regime, single-
# sourced from params so the two cannot drift.
_CELL_DEFAULTS = {**{k: params.SCALING[k]
                     for k in ("bins", "depth", "iters", "lr", "seed",
                               "min_data_in_leaf", "lambda_l2")},
                  "informative": 20}


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


def bundled_specs() -> list[str]:
    from importlib import resources
    d = resources.files(__package__) / "specs"
    return sorted(f.name.removesuffix(".json") for f in d.iterdir()
                  if f.name.endswith(".json"))


def load_spec(path: str | pathlib.Path) -> dict:
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
    c = {**_CELL_DEFAULTS, **defaults, **over}
    if "rows" not in c or "cols" not in c:
        raise ValueError(f"cell needs rows and cols: {c}")
    c.setdefault("axis", "cell")
    c.setdefault("n_test", min(c["rows"] // 5, 500_000))
    c["bins_effective"] = c["bins"]
    return c


def gen_iso_volume(entry: dict, defaults: dict) -> list[dict]:
    """Cells of constant rows x cols volume, sweeping the aspect ratio."""
    cells_total = 1 << entry["log2_cells"]
    out = []
    for cols in entry["cols"]:
        if cells_total % cols:
            raise ValueError(f"cols {cols} does not divide 2^{entry['log2_cells']}")
        rows = cells_total // cols
        over = {k: v for k, v in entry.items()
                if k not in ("gen", "log2_cells", "cols")}
        out.append(make_cell(defaults, rows=rows, cols=cols, axis="iso_volume",
                             aspect=rows / cols, **over))
    return out


_GENERATORS = {"iso_volume": gen_iso_volume}


def cells_of(spec: dict) -> list[dict]:
    defaults = spec.get("defaults", {})
    out = []
    for entry in spec["cells"]:
        if "gen" in entry:
            out.extend(_GENERATORS[entry["gen"]](entry, defaults))
        else:
            out.append(make_cell(defaults, **entry))
    return out


def _repeats_for(variant: str, policy: dict | int) -> int:
    if isinstance(policy, int):
        return policy
    device = resolve(variant).device
    return int(policy.get(device, policy.get("default", 1)))


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
    variant_iters = spec.get("variant_iters", {})
    jobs = []
    for cell in cells_of(spec):
        for variant in chosen:
            for t in threads:
                iters_ladder = variant_iters.get(variant)
                for iters in (iters_ladder or [cell["iters"]]):
                    c = dict(cell, iters=iters) if iters_ladder else dict(cell)
                    jobs.append({"cell": c, "variant": variant, "threads": t,
                                 "repeats": _repeats_for(variant, policy)})
    return jobs
