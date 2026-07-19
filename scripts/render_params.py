"""Render docs/use/parameters.md from the committed config extraction.

The source of truth is the config section descriptors and the
default-constructed structs under `include/bonsai/config/`. The `bonsai
params` command dumps that default `Config` as TOML straight from the real
structs (`src/cli/params.cpp` calls `config::dump_toml`). This generator
never parses C++: it reads the committed JSON extraction of that dump and
merges a hand-maintained effect line per knob, so every parameter carries a
type, a default, and an effect on the model.

    bonsai params | python3 scripts/render_params.py --extract  # refresh JSON
    python3 scripts/render_params.py                            # rewrite page
    python3 scripts/render_params.py --check                    # CI: fail on drift

Run the extract step through `make params-json`; it needs the built CLI and
Python 3.11+ (`tomllib`, imported lazily so the check path stays 3.9-clean).
The check step reads only the committed JSON, stdlib-only on every supported
Python, so CI never builds C++ to verify the page. A knob with no effect line
is a hard error listing the names, so a new knob forces documentation.

Freshness: CI checks page-against-JSON, not JSON-against-structs (that needs
a C++ build). Re-run `make params-json` after adding or renaming a knob. The
round-trip-safe TOML dump omits unset `std::optional` fields, so those are
declared explicitly in OPTIONAL_KNOBS below and will not appear from the dump
on their own.
"""

from __future__ import annotations

import json
import pathlib
import struct
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
SRC = REPO / "docs" / "use" / "parameters.src.json"
OUT = REPO / "docs" / "use" / "parameters.md"

# Section reading order (most-tuned first); any section absent here is
# appended in sorted order so a new section still renders.
SECTION_ORDER = [
    "dispatch", "booster", "tree", "sampler", "bin_mapper",
    "objective", "metrics", "data", "parallel",
]

SECTION_INTRO = {
    "dispatch": "Selects the algorithm: which loss, which tree grower, which row sampler.",
    "booster": "The boosting loop: how many trees, how strongly each contributes, when to stop.",
    "tree": "Per-tree shape and regularization, the primary overfitting controls.",
    "sampler": "Row subsampling, active only for the `bernoulli` and `goss` samplers.",
    "bin_mapper": "Histogram binning: how feature values become the bins that splits search over.",
    "objective": "Extra parameters read only by the matching objective.",
    "metrics": "Which metrics to report during fit and eval; no effect on the trained model.",
    "data": "Dataset IO for the CLI; the Python API passes arrays instead of paths.",
    "parallel": "Compute placement: CPU threads and CUDA device. No effect on the model bits.",
}

# One effect-on-model line per dotted knob (type and default come from the
# extraction). Keep each under 25 words, concrete about the model effect.
EFFECTS = {
    # dispatch
    "dispatch.objective_name": "Training loss: mse, mae, huber, quantile, poisson, logloss, or softmax. Sets the gradients, the link, and the default metrics.",
    "dispatch.grower_name": "Tree-growth strategy: depthwise, leafwise, or oblivious (plus cuda_ variants). Leafwise cuts loss faster per node; oblivious builds symmetric trees.",
    "dispatch.sampler_name": "Row-sampling strategy: all_rows, bernoulli (uniform subsample), or goss (keeps large-gradient rows). Trades a little accuracy for training speed.",
    # booster
    "booster.n_iters": "Number of boosting rounds. More rounds lower training loss but risk overfitting; pair with a smaller learning_rate.",
    "booster.learning_rate": "Shrinks each tree's contribution. Lower values need more rounds but generalize better, the classic accuracy-versus-iterations trade.",
    "booster.random_seed": "Seeds the stochastic boosting choices, such as DART drops. Fixing it makes a run reproducible bit for bit.",
    "booster.log_intervals": "Progress-logging frequency during fit. 0 is silent; higher prints more metric ticks. No effect on the trained model.",
    "booster.early_stopping_rounds": "Stop when the first validation objective stalls this many rounds, keeping the best iteration. 0 disables it; needs a valid set.",
    "booster.dart_drop_rate": "DART: probability of dropping each existing tree per round. Above 0 enables DART regularization; incompatible with early stopping.",
    # tree
    "tree.min_child_hess": "Minimum summed hessian per leaf. Higher values block splits on thin data, reducing overfitting; the hessian-weighted analog of min_child_weight.",
    "tree.min_gain_to_split": "Minimum loss reduction to accept a split. Raising it prunes low-value splits and shrinks trees, curbing overfitting.",
    "tree.lambda_l2": "L2 penalty on leaf weights. Higher values shrink leaf outputs toward zero, smoothing predictions and reducing variance.",
    "tree.lambda_l1": "L1 penalty on leaf weights. Above 0 pushes small leaf outputs to exactly zero, a sparser, more regularized fit.",
    "tree.max_depth": "Maximum tree depth. Deeper trees capture more feature interactions but overfit and cost more; the primary complexity knob.",
    "tree.min_data_in_leaf": "Minimum training rows per leaf. Higher values prevent tiny leaves fit to noise, a strong overfitting guard on small data.",
    "tree.max_leaves": "Leaf cap for leafwise growth. 0 is unbounded (depth-capped). Fewer leaves regularize; more leaves fit finer structure.",
    "tree.feature_fraction": "Fraction of features sampled per tree. Below 1 decorrelates trees and speeds training, often improving generalization.",
    "tree.feature_seed": "Seeds the per-tree feature_fraction draws. Fixing it keeps feature subsampling reproducible across runs.",
    "tree.monotone_constraints": "Per-feature monotone direction: +1 increasing, -1 decreasing, 0 free. Forces predictions to respect known monotonic relationships. Node-splitting growers only.",
    "tree.interaction_constraints": "Feature groups allowed to interact on one tree path. Restricts which features co-split, encoding domain structure and limiting interactions.",
    # sampler
    "sampler.subsample": "bernoulli sampler: fraction of rows drawn per tree. Below 1 adds randomness and speed, trading accuracy for less overfitting.",
    "sampler.top_rate": "goss sampler: fraction of large-gradient rows always kept. Higher retains more of the informative rows goss prioritizes.",
    "sampler.other_rate": "goss sampler: fraction sampled from the remaining small-gradient rows. Higher samples more, nearer full-data accuracy but slower.",
    # bin_mapper
    "bin_mapper.max_bin": "Histogram bins per feature. More bins mean finer splits and slower training; below about 64 accuracy degrades on continuous features.",
    "bin_mapper.n_samples": "Rows sampled to compute bin edges. More gives more accurate quantiles at higher binning cost; the default already stabilizes edges.",
    "bin_mapper.seed": "Seeds the row sample used for bin-edge quantiles. Fixing it keeps bin boundaries reproducible across runs.",
    "bin_mapper.min_data_in_bin": "Minimum rows per histogram bin. Higher values merge sparse bins, coarsening splits and guarding against noise-driven cuts.",
    # objective
    "objective.huber_delta": "huber objective: residual half-width of the L2 zone. Smaller behaves more like L1 (robust to outliers), larger like L2.",
    "objective.quantile_alpha": "quantile objective: the target quantile in (0, 1). 0.5 is the median; higher tilts predictions toward that upper quantile.",
    "objective.n_classes": "softmax objective: number of classes, labels 0 to K-1. Must match the data; sets the trees-per-round multiplier.",
    # metrics
    "metrics.fit": "Metric names reported during fit. Empty uses the objective's defaults. Reporting only, no effect on the model.",
    "metrics.eval": "Metric names reported during eval. Empty uses the objective's defaults. Reporting only, no effect on the model.",
    # data
    "data.train": "Path to the training dataset. The CLI reads it; the Python API passes arrays instead.",
    "data.test": "Path to a test dataset scored after fit. CLI-only, like train.",
    "data.valid": "Paths to validation datasets. The first drives early stopping; all are scored each logging tick.",
    "data.format": "Input file format: csv or libsvm. Selects the parser and must match the files.",
    "data.header": "Whether CSV inputs have a header row. The wrong setting misreads the first row as data or as names.",
    "data.label_column": "Zero-based index of the label column in CSV inputs. Points the parser at the target.",
    "data.weight_column": "Zero-based index of a per-row weight column. -1 means unweighted; a valid index weights the loss per row.",
    "data.ignore_columns": "Zero-based column indices to drop before training. Excludes ids or leaks without editing the file.",
    "data.missing_nan": "Treat NaN as missing, routed to each split's learned default branch. Off means NaN is an ordinary value.",
    "data.missing_sentinel": "Extra numeric value treated as missing, alongside NaN. Unset by default; set it to flag a placeholder like -999.",
    "data.libsvm_n_features": "libsvm only: force the feature count. 0 infers from the max index; set it when a split's max index is lower.",
    # parallel
    "parallel.n_threads": "CPU threads for training. 0 auto-detects hardware threads, capped at 16. More threads speed fit; model bits are unchanged.",
    "parallel.device_id": "CUDA device for cuda_ growers. Placement only: ignored by CPU growers and deliberately not stored in the model.",
}

# Knobs the round-trip-safe TOML dump cannot represent, declared by hand.
# `std::optional` fields default to nullopt and are skipped by dump_toml, so
# they never appear in the extraction; type and default are read off the
# struct. A new optional knob added here (or to the struct) is the one drift
# case CI cannot catch from the JSON alone; see the module docstring.
OPTIONAL_KNOBS = {
    "data.missing_sentinel": {"type": "float (optional)", "default": "unset"},
}


def _clean_float(x: float) -> float:
    """Shortest decimal that round-trips to the same float32 the struct holds.

    `dump_toml` widens float32 fields to double, so a 0.05F default prints as
    0.05000000074505806. Recover the intended short value by finding the
    fewest significant digits that pack back to the identical float32.
    """
    target = struct.pack("<f", x)
    for prec in range(1, 10):
        s = f"{x:.{prec}g}"
        if struct.pack("<f", float(s)) == target:
            return float(s)
    return x


def extract() -> int:
    import tomllib  # 3.11+; only the extract path (make params-json) needs it

    data = tomllib.loads(sys.stdin.read())
    out: dict[str, dict] = {}
    for section in sorted(data):
        cleaned: dict[str, object] = {}
        for key in sorted(data[section]):
            value = data[section][key]
            cleaned[key] = _clean_float(value) if isinstance(value, float) else value
        out[section] = cleaned
    SRC.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    n = sum(len(v) for v in out.values())
    print(f"wrote {SRC.relative_to(REPO)} ({n} keys from bonsai params)")
    return 0


def type_label(value) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    return "?"


def fmt_default(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return '""' if value == "" else f'"{value}"'
    if isinstance(value, list):
        return "[]" if not value else json.dumps(value)
    if isinstance(value, float):
        r = repr(value)
        return r if ("." in r or "e" in r) else r + ".0"
    return str(value)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "|" + "|".join("---" for _ in headers) + "|"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def sections_from_src() -> tuple[dict[str, list[tuple]], list[str]]:
    """Return {section: [(leaf, dotted, type, default), ...]} plus all dotted
    keys, merging the hand-declared optional knobs into their sections."""
    data = json.loads(SRC.read_text())
    sections: dict[str, list[tuple]] = {}
    for sec, keys in data.items():
        sections[sec] = [
            (leaf, f"{sec}.{leaf}", type_label(v), fmt_default(v))
            for leaf, v in keys.items()
        ]
    for dotted, meta in OPTIONAL_KNOBS.items():
        sec, leaf = dotted.split(".", 1)
        sections.setdefault(sec, []).append(
            (leaf, dotted, meta["type"], meta["default"]))
    for sec in sections:
        sections[sec].sort(key=lambda row: row[0])
    dotted = [row[1] for rows in sections.values() for row in rows]
    return sections, dotted


HEADER = """<!-- GENERATED by scripts/render_params.py from docs/use/parameters.src.json. Edit the generator or run `make params-json`, not this page. -->

# Parameters

This page lists every training knob, with its type, default, and effect on the model. Set any parameter three ways: on the CLI with `--set tree.max_depth=8`, under `[tree]` in a TOML config, or in Python with `params=[("tree.max_depth", 8)]`. Run `bonsai params` to print the whole default config as TOML.

The table below is generated from `docs/use/parameters.src.json`, extracted from the default `Config` by `make params-json` (which runs `bonsai params`). CI checks the page against that JSON. Re-run `make params-json` after adding or renaming a knob so the extraction tracks the structs.
"""


def render() -> str:
    sections, _ = sections_from_src()
    ordered = [s for s in SECTION_ORDER if s in sections]
    ordered += sorted(s for s in sections if s not in SECTION_ORDER)
    parts = [HEADER]
    for sec in ordered:
        block = [f"## {sec}"]
        if sec in SECTION_INTRO:
            block.append("\n" + SECTION_INTRO[sec])
        rows = [
            [f"`{leaf}`", typ, f"`{default}`", EFFECTS[dotted]]
            for leaf, dotted, typ, default in sections[sec]
        ]
        block.append("\n" + md_table(["parameter", "type", "default", "effect"], rows))
        parts.append("\n".join(block) + "\n")
    return "\n".join(parts)


def coverage_error() -> str | None:
    _, dotted = sections_from_src()
    known = set(dotted)
    missing = sorted(d for d in dotted if d not in EFFECTS)
    stale = sorted(e for e in EFFECTS if e not in known)
    problems = []
    if missing:
        problems.append(
            "knobs with no effect line (add one to EFFECTS in "
            f"scripts/render_params.py): {missing}")
    if stale:
        problems.append(
            "effect lines for knobs not in the extraction (remove them or "
            f"re-run make params-json): {stale}")
    return "; ".join(problems) if problems else None


def main() -> int:
    if "--extract" in sys.argv:
        return extract()
    err = coverage_error()
    if err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1
    text = render()
    n = sum(len(rows) for rows in sections_from_src()[0].values())
    if "--check" in sys.argv:
        if not OUT.exists() or OUT.read_text() != text:
            print("ERROR: docs/use/parameters.md is stale; run "
                  "python3 scripts/render_params.py", file=sys.stderr)
            return 1
        print(f"parameters reference: in sync ({n} knobs)")
        return 0
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(REPO)} ({n} knobs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
