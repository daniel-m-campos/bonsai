---
name: feature-admission
description: >
  bonsai's gate for admitting a new feature into the core: measure the
  benefit at zero core cost first, price the complexity on named axes, and
  only then write C++. Use whenever a capability is proposed (a new split
  type, objective, encoding, sampler, data format), especially when the
  pitch is "the other libraries have it". Invoke via /feature-admission.
---

A feature is admitted by measurement, not by ecosystem envy. The burden of proof is on the feature; the default answer is "not in the core". Worked example: decision 58 + `benchmarks/categorical-tradeoff-2026-07.md` (the probe script itself is closed evidence and lives in git history), where this method killed a fully-designed engine feature (drafted in full, then declined; the draft lives in git history) and shipped a 100-line encoder that beat it.

Three tests decide it. The benefit is shown by the cheapest possible prototype, at zero cost to the core. The benefit moves standings on a fixed evaluation suite, not a hand-picked example. The kill criteria were written down before the experiment ran.

Two worked gates, both cheap, both decisive:

- **Categoricals.** The most-requested structural feature, present in all
three reference libraries. The cheapest prototype was not C++: toggle LightGBM's own categorical support on and off, and feed bonsai a 40-line ordered-target-statistics preprocessing step, across three real datasets. LightGBM's native toggle HURT on one of the three (kick, -0.018 AUC against plain ordinals); bonsai plus preprocessing beat LightGBM-native on the hardest (amazon, 0.8590 against 0.8572). The drafted C++ design was declined and the encoder shipped (decision 58).
- **Ordered boosting.** The hypothesis that CatBoost's accuracy at scale came
from ordered boosting was killed without implementing anything: benchmark CatBoost against itself, `Ordered` against `Plain`, same data and budget. Identical accuracy at roughly 7x the cost, and CatBoost defaults it off past ~50k rows anyway.

That is the gate's cheapest form: when the feature exists in a reference implementation, the reference IS the prototype. Measure it there before building it here.

The output worth the most is the decline: a recorded, measured "no" with the conditions that would reopen it, which converts a recurring debate into a lookup. Price on named axes before anyone writes C++: core lines of code, new configuration knobs, whether existing behavior stays bit-identical, and whether the documentation story survives.

## Step 1: prototype the benefit at zero core cost

Before any C++, demonstrate the ceiling with whichever of these is cheapest:

- **Reference-library toggle**: if LightGBM, XGBoost, or CatBoost ship the feature, toggle it on/off in *their* config at campaign-matched knobs (`scripts/reference_params.py`). Their own toggle prices the feature honestly, e.g. Lightgbm's native categorical splits measured **negative** on kick.
- **Preprocessing prototype**: if the feature can be simulated by transforming inputs (encodings, binning schemes, feature crosses), write it in Python and feed stock bonsai.
- **Config-only ablation**: if it's expressible as existing knobs (e.g. simulating a constraint via data manipulation), do that.
- **Python harness**: for objectives/losses, prototype the gradient/hessian in a custom-objective harness of a reference library first.

If none of these can demonstrate the benefit, say so explicitly in the decision entry: "cannot be prototyped outside the core" is itself a finding (true for e.g. a ranking objective, false for most encodings).

## Step 2: the benefit criterion: standings moved

Measure on **multiple realistic datasets** (the quality-suite pattern: `scripts/compare.py`, 200 iters / lr 0.05 / depth 6 / 255 bins / seed 42, task-appropriate metrics), never a single flagship dataset. The bar:

- The feature must **change who wins** (a best-library cell flips on the standing table), or unlock a workload class bonsai currently cannot serve at all.
- Deltas under the chance band (±0.001 AUC-ish, ±2% rmse-ish at these sizes, decision 55) are noise, not benefit.
- Report the full distribution including losses. A feature that wins one dataset and loses another is a coin flip; coin flips are declined.

## Step 3: price the cost on named axes

- **Core lines**: `wc -l` the files touched (the split/tree/SHAP/model core is ~1,400 lines total, and its smallness is the pedagogy moat; growing it needs proportional benefit).
- **Hot-path branches**: does the existing numeric/default path pay anything? Gate: `scripts/model_hash.py` byte-identity for untouched configs (the quality-gates ritual).
- **Config surface**: new knobs multiply the dispatch product and the docs; count them.
- **Model format**: version bumps ripple to every loader.
- **Guide impact**: can the feature be a *leaf* chapter (additive, like guide 13), or does it complicate chapters 1–4? Core-chapter changes are the most expensive line item.
- **Pre-registered kill criterion**: write down, before building, the measurement that would revert the feature.

## Step 4: verdict, recorded

- Write the probe as a committed script (`scripts/probe_*.py`) and the evidence as `benchmarks/<name>-tradeoff-<date>.md` + raw JSON in `benchmarks/results/` (gitignored, so `git add -f`).
- Record the call in `docs/decisions.md`: **adopted** (with the measured table), **declined by measurement** (keep any design doc, marked priced-but-declined, like doc 17), or **deferred pending probe X**.
- A refutation is a deliverable (doc 16's rule). Killing a feature with data is a win, not a failure.

## The escape hatch

Evidence can reopen a declined feature: the decision entry must name what measurement would change the answer (for categoricals: crossed-TS preprocessing failing to close the CatBoost gap AND a user workload where that gap is load-bearing).
