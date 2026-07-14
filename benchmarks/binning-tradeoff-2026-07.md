# Per-feature bin budgets: priced and declined (2026-07-14)

The hypothesis: per-feature bin budgets (more bins on informative features, fewer on noise; ultimately user-supplied edges) could close decision 55's +0.001 r² cut-quality residual vs xgboost and serve users with domain-mandated binning. lightgbm ships `max_bin_by_feature`; catboost ships `per_float_feature_quantization`; the probe prices the idea with zero bonsai C++ changes.

Method (`scripts/probe_binning.py`, campaign knobs, seed 42, `bin_mapper.n_samples = n_rows` for exactness): bonsai budgets are emulated by pre-discretizing each feature to its budget (the issue-#61 one-cut-per-distinct rule reproduces the budgets exactly, verified); lightgbm prices the feature natively via its own toggle. Policies share uniform-255's total budget except `headroom` (top-10 features by gain importance get 1023, no cap). `inverse` allocates against importance as the causal control.

Results (test r² / AUC deltas vs bonsai uniform-255; raw in `results/binning-probe-2026-07.json`):

| dataset | uniform | importance | inverse (control) | headroom | lgbm toggle Δ |
|---|--:|--:|--:|--:|--:|
| synth 20-informative-of-100 | 0.8694 | +0.0003 | −0.0836 | +0.0011 | +0.0010 |
| california | 0.8260 | +0.0011 | +0.0014 | +0.0016 | +0.0000 |
| adult | 0.9298 | −0.0011 | −0.0126 | −0.0012 | +0.0000 |
| kick | 0.7830 | −0.0096 | −0.0192 | −0.0103 | −0.0024 |
| year MSD | 0.3012 | −0.0012 | −0.0042 | −0.0003 | +0.0005 |

Findings:

1. **No upside at the default.** Even on the synthetic with known signal structure and unlimited extra budget, the best gain (+0.0011) sits at the chance band. No standings cell flips anywhere. lightgbm's own per-feature toggle agrees: ≤ +0.001 at best, negative on kick.
2. **Real downside when misallocated.** The inverse control costs up to −0.084 r², and even the importance-GUIDED policy loses on adult and kick because a 50-iteration importance probe misallocates. The asymmetry is the verdict: uniform 255 is at saturation; reallocation can only break things.
3. **Decision 55's residual does not close.** Headroom on MSD moves −0.0003; the cut-quality gap is not a resolution-allocation problem.
4. **The emulation is the feature.** Users with genuine domain-mandated bins (regulatory buckets, known discontinuities) can bin exactly today: pre-discretize to bin ids and stock bonsai reproduces the scheme bit-exactly via the distinct-value rule. `scripts/probe_binning.py`'s `ManualBinner` is the reference recipe.

Verdict: automatic accuracy-motivated budgets **declined by measurement**. Explicit user-supplied edges are a separate, unmeasured axis: the emulation is train-time exact but keeps the binning outside the model artifact (predict requires the transform forever), which is a deployment question this probe does not adjudicate; see decision 67 and architecture doc 18 for the API design and its admission gate.
