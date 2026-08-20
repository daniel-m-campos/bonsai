"""One source of truth for benchmark knobs, over one source of truth for
name translation.

A one-knob drift here has produced a false experimental conclusion twice
(max_bin 255 vs 256, decision 55's follow-up; min_child_weight 1 vs 20,
decision 68's correction), so every harness and probe imports these instead
of re-deriving them.

Each ``*_core`` builder states the cell in bonsai's own dotted registry
keys, checked by ``Params`` at call time, and hands them to ``interop`` to
be renamed. What stays here is what a translation cannot know, namely the
campaign's deliberate choices: which bonsai knob rides which reference knob
when the two are not equivalent, and the per-library settings that only a
benchmark wants (silenced logs, the GPU border cap, histogram method).
"""

from __future__ import annotations

from bonsai import interop
from bonsai.bench.variants import Device
from bonsai.params import Params

# The campaign regime: quality-division suites (Grinsztajn standings, the
# internal smoke campaign, admission-gate probes). Decisions 56 and 68.
CAMPAIGN = dict(iters=200, lr=0.05, depth=6, bins=255, min_data_in_leaf=20,
                lambda_l2=1.0, seed=42)

# The scale regime: perf-division synthetic sweeps. Decision 46.
SCALING = dict(iters=100, lr=0.1, depth=8, bins=255, min_data_in_leaf=20,
               lambda_l2=1.0, seed=42)

# The early-stopping regime, layered on SCALING for the time-to-stop arm: a
# cap high enough that the valid set, not the cap, ends the fit, a patience
# wide enough that noise does not, and the lower rate those two imply.
# eval_frac is the held-out slice carved off the TRAIN side (the 90/10 split
# the early-stopping guide already benchmarks), so the test split stays
# untouched for the final metric.
EARLY_STOP = dict(iters=2000, lr=0.05, patience=50, eval_frac=0.1)

# bonsai dotted keys for the campaign regime (the estimator kwargs carry the
# rest); used by quality suites and probes. Values come from CAMPAIGN, keys
# are validated against the registry, so neither can drift alone.
BONSAI_CAMPAIGN_PARAMS = Params.from_dict({
    "tree.min_data_in_leaf": CAMPAIGN["min_data_in_leaf"],
    "tree.lambda_l2": CAMPAIGN["lambda_l2"],
    "bin_mapper.max_bin": CAMPAIGN["bins"],
}).to_dict()


def num_leaves_campaign(depth: int) -> int:
    """The campaign/compare convention: (1 << depth) - 1 leaves."""
    return (1 << depth) - 1


def num_leaves_full(depth: int) -> int:
    """The scaling/gpu-bench convention: full 1 << depth leaves."""
    return 1 << depth


def num_leaves_of(cell: dict) -> int:
    """A cell's leaf budget: an explicit num_leaves, else the full-depth
    convention. Uncapped-depth cells must name it, because 1 << depth is
    not a budget once the depth cap is lifted."""
    return cell.get("num_leaves") or num_leaves_full(cell["depth"])


def bonsai_core(*, learning_rate, max_depth, num_leaves, min_data_in_leaf,
                lambda_l2, max_bin, seed, n_iters, n_threads, grower,
                objective="mse",
                early_stopping_rounds=0) -> list[tuple[str, str]]:
    """Dotted-key pairs for bonsai.train, mirroring the reference mappings
    below so harnesses stop hand-building (and drifting) the pair list.

    early_stopping_rounds is bonsai's own patience config key (verified
    against docs/use/parameters.md and the module docstring in
    src/python/module.cpp: `train(pairs, dataset, eval_set=(Xv, yv))` is what
    enables per-iter eval, and booster.early_stopping_rounds is what arms the
    stop). 0 omits the key entirely rather than writing the default, so a
    fixed-iteration arm's pair list is byte-identical to a run predating
    early stopping.
    """
    pairs = [
        ("dispatch.grower_name", grower),
        ("dispatch.objective_name", objective),
        ("booster.n_iters", str(n_iters)),
        ("booster.learning_rate", str(learning_rate)),
        ("booster.random_seed", str(seed)),
        ("tree.max_depth", str(max_depth)),
        ("tree.max_leaves", str(num_leaves)),
        ("tree.min_data_in_leaf", str(min_data_in_leaf)),
        ("tree.lambda_l2", str(lambda_l2)),
        ("bin_mapper.max_bin", str(max_bin)),
        ("parallel.n_threads", str(n_threads)),
    ]
    if early_stopping_rounds:
        pairs.append(("booster.early_stopping_rounds",
                      str(early_stopping_rounds)))
    return pairs


def xgb_core(*, learning_rate, max_depth, min_data_in_leaf, lambda_l2, max_bin,
             seed) -> dict:
    """xgboost params matched to the shared knob names.

    The campaign's row floor rides xgboost's hessian floor: bonsai's
    min_data_in_leaf is written to tree.min_child_hess, which interop renames
    to min_child_weight. That is a benchmark choice, not an equivalence (a
    hessian floor is a row count only under squared error), and it is the
    decision-68 correction: leaving it at xgboost's default 1 produced a
    false conclusion once. tree_method is the histogram method bonsai always
    uses, and has no bonsai key to be renamed from.
    """
    core = interop.to_xgboost(Params.from_dict({
        "booster.learning_rate": learning_rate,
        "tree.max_depth": max_depth,
        "tree.min_child_hess": min_data_in_leaf,
        "tree.lambda_l2": lambda_l2,
        "bin_mapper.max_bin": max_bin,
        "booster.random_seed": seed,
    }))
    return {**core, "tree_method": "hist"}


def lgbm_core(*, learning_rate, max_depth, num_leaves, min_data_in_leaf,
              lambda_l2, max_bin, seed) -> dict:
    """lightgbm params matched to the shared knob names, silenced logs."""
    core = interop.to_lightgbm(Params.from_dict({
        "booster.learning_rate": learning_rate,
        "tree.max_depth": max_depth,
        "tree.max_leaves": num_leaves,
        "tree.min_data_in_leaf": min_data_in_leaf,
        "tree.lambda_l2": lambda_l2,
        "bin_mapper.max_bin": max_bin,
        "booster.random_seed": seed,
    }))
    return {**core, "verbose": -1}


def catboost_core(*, learning_rate, max_depth, lambda_l2, max_bin, seed,
                  device) -> dict:
    """catboost params matched to the shared knob names.

    max_bin arrives in BIN semantics (what bonsai/xgboost/lightgbm count);
    interop applies the bins-to-borders fencepost, because CatBoost's
    border_count counts SPLITS. What is a benchmark rule rather than a
    translation stays here: GPU caps border_count at 254 (= 255 bins,
    matching the campaign/scale default exactly).
    """
    core = interop.to_catboost(Params.from_dict({
        "booster.learning_rate": learning_rate,
        "tree.max_depth": max_depth,
        "tree.lambda_l2": lambda_l2,
        "bin_mapper.max_bin": max_bin,
        "booster.random_seed": seed,
    }))
    if device == Device.CUDA:
        core["border_count"] = min(core["border_count"], 254)
    return core


def xgb_early_stop(rounds: int) -> dict:
    """Extra xgb.train kwargs that arm early stopping; {} when rounds is 0.

    xgboost takes patience at the call site, not in the params dict
    (verified against xgboost 3.3's `xgb.train` signature: `evals` and
    `early_stopping_rounds` are keyword-only arguments there), so this is a
    call-site builder rather than a `xgb_core` knob. The runner supplies
    `evals`, because only it holds the validation DMatrix.
    """
    return {"early_stopping_rounds": rounds} if rounds else {}


def lgbm_early_stop(rounds: int) -> dict:
    """Kwargs for lgb.early_stopping(...); {} when rounds is 0.

    lightgbm's documented mechanism is the callback, not a params key
    (verified against lightgbm 4.6's `lgb.early_stopping(stopping_rounds,
    first_metric_only, verbose, min_delta)` signature). Returned as kwargs
    so params.py stays free of a lightgbm import: `bonsai.bench` must import
    without dragging in the reference libraries.
    """
    return {"stopping_rounds": rounds, "verbose": False} if rounds else {}


def catboost_early_stop(rounds: int, *, has_eval_set: bool) -> dict:
    """Extra CatBoost constructor params for an eval-set arm.

    Two translations live here rather than at the call site. Patience is
    CatBoost's overfitting detector, `od_type="Iter"` plus `od_wait`
    (verified against catboost 1.2.10). And CatBoost is the only library
    that shrinks the model to its best iteration merely because an eval set
    was passed: measured on 1.2.10, 200 requested iterations came back as a
    140-tree model with no detector armed at all. A fixed-iteration arm must
    therefore say `use_best_model=False`, or it reports the metric of a
    model shorter than the one it was timed training.
    """
    if not has_eval_set:
        return {}
    if not rounds:
        return {"use_best_model": False}
    return {"od_type": "Iter", "od_wait": rounds}
