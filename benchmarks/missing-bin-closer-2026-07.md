# The missing-bin closer: probe evidence (issue #155, decision 74)

Fitted bins leaked finite training values into the NaN sentinel three ways: the stride path's top tail, the greedy path's final group, and rows above the 200k bin sample's maximum. Leaked rows trained as missing (routed by `default_left`) but predicted by raw threshold, a train/predict skew. The fix appends a `FLT_MAX` top-band cut before the `+inf` sentinel in every fitting path, the same closer `from_edges` (decision 73) uses.

Probe: `scripts/probe_missing_bin.py`, A/B via a build-time toggle, campaign knobs (200 iters, lr 0.05, depth 6, 255 bins, seed 42), both growers, per-arm child processes.

## Mechanism synthetics (one per leak)

| case | grower | off r² | on r² | delta |
|---|---|--:|--:|--:|
| capped-max heavy value (greedy path) | depthwise | 0.483 | 0.988 | **+0.505** |
| capped-max heavy value (greedy path) | oblivious | 0.483 | 0.988 | **+0.504** |
| rare top-tail signal (stride path) | depthwise | 0.594 | 0.755 | **+0.161** |
| rare top-tail signal (stride path) | oblivious | 0.593 | 0.770 | **+0.176** |
| signal above sampled max (1M rows) | depthwise | 0.959 | 0.968 | +0.008 |
| signal above sampled max (1M rows) | oblivious | 0.959 | 0.968 | +0.009 |

The capped-sensor case is the realistic catastrophe: a 10%-heavy cap value carrying signal loses half the variance when it trains as missing. Clipped telemetry, capped amounts, and 999-coded maxima are common shapes the quality suite happens not to contain.

## Real suite

| dataset | metric | depthwise delta | oblivious delta |
|---|---|--:|--:|
| california | r² | +0.0011 | −0.0010 |
| amazon | AUC | −0.0020 | −0.0005 |
| higgs (500k rows) | AUC | +0.0001 | −0.0001 |
| airline 0.1m | AUC | +0.0006 | −0.0006 |

The amazon −0.002 was checked against a null yardstick: the off-arm's own sensitivity to `max_bin` 253/254/255/256 spans 0.0034 (0.8179-0.8212), so the closer's delta is inside the dataset's ordinary bin-configuration churn (decision-55 threshold-placement noise on high-cardinality ID codes). Seeds do not perturb this config; the null perturbation is the honest error bar.

## Verdict

Adopted (decision 74) on robustness and invariant grounds with the standings-neutral suite recorded honestly: the sentinel bin is NaN-only on every path, the train/predict skew is gone, the cost is a budget slot that was already reserved. Canonical hashes moved (serial `09dbf47353033362`, sampled `ca7174cb1560221e`); California pin 0.71725→0.7153 (−0.27%, in-band).
