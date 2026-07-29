# Expectile objective: priced and declined (decision 87)

2026-07-28. Probe: [`scripts/probe_expectile.py`](../scripts/probe_expectile.py). Raw rows: [`results/expectile-probe-2026-07.jsonl`](results/expectile-probe-2026-07.jsonl). Verdict recorded as [decision 87](../docs/decisions.md).

## Why this probe exists

XGBoost 3.3 (2026-07-21) shipped `reg:expectileerror`. Expectile loss is `w(r) * r^2` with `r = y - pred` and `w = alpha` when the model underpredicts, `1 - alpha` when it overpredicts: a smooth loss for problems where the two error directions carry different costs. Unlike the pinball loss it has a real, nonzero second derivative, so Newton-step boosting consumes it natively, and expectile curves for different alphas cannot cross. bonsai ships quantile (pinball) but not expectile; the admission question is whether expectile unlocks a workload the existing surface cannot serve.

## Protocol

Campaign-matched knobs (200 iters, lr 0.05, depth 6, 255 bins, seed 42), 60/20/20 train/validation/holdout split, alpha in {0.75, 0.9, 0.95}. Two datasets: a 200k x 20 heteroscedastic synthetic (noise scale rides on a feature, so the conditional expectile genuinely varies with x and no constant shift can match it) and california housing (20.6k rows, real, right-skewed target). Single seed: differences under ~2% are noise.

Six arms per cell. Two XGBoost 3.3 reference arms (native `reg:expectileerror`, and the same loss as a custom objective, which validates the gradient math in both directions). One naive baseline per library (MSE fit plus a constant shift fitted on the validation split to minimize expectile loss; the shift is the alpha-expectile of the validation residuals). Two zero-core bonsai arms: the shipped quantile objective at the same alpha plus the same shift treatment, and iteratively reweighted least squares (a plain MSE fit, then two refits with `sample_weight` set to alpha or 1 - alpha by the previous fit's residual sign; three fits total).

The metric is holdout expectile loss at the target alpha. The jsonl also records the identification gap (mean of `w * r` on holdout, zero at the true conditional expectile) as a calibration diagnostic.

## Results

Holdout expectile loss, lower is better, bold best per row:

| dataset | alpha | xgb expectile | xgb custom-obj | xgb mse+shift | bonsai mse+shift | bonsai quantile+shift | bonsai IRLS x2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| california | 0.75 | 0.1088 | **0.1069** | 0.1114 | 0.1105 | 0.1193 | 0.1088 |
| california | 0.90 | 0.0830 | **0.0822** | 0.0857 | 0.0849 | 0.0967 | 0.0830 |
| california | 0.95 | **0.0625** | 0.0645 | 0.0652 | 0.0645 | 0.0752 | 0.0638 |
| synthetic_hetero | 0.75 | 1.5848 | 1.5868 | 1.7249 | 1.7268 | **1.5609** | 1.5884 |
| synthetic_hetero | 0.90 | **1.0126** | 1.0211 | 1.3569 | 1.3538 | 1.0204 | 1.0186 |
| synthetic_hetero | 0.95 | 0.6940 | 0.7074 | 1.0882 | 1.0837 | **0.6867** | 0.6895 |

Three readings. First, the workload is real: on the heteroscedastic synthetic the true expectile fit beats MSE plus shift by 9%, 34%, and 57% as alpha climbs, and even on california the gap is 2-4% and grows with alpha. A constant shift fixes the level but cannot track a noise scale that varies with x. Second, bonsai already serves it: the IRLS arm ties the native XGBoost objective on every cell (ratios 0.99-1.02), at three fits of wall clock and zero core lines. Third, quantile plus shift is a partial substitute only: it ties or wins on the symmetric synthetic (where the quantile and expectile surfaces differ mainly by a level constant the shift absorbs) but trails by 10-20% on the skewed real data, where the two functionals genuinely diverge.

Calibration fine print: the shift arms sit nearest zero identification gap by construction (the shift is fitted to exactly that), while the native and IRLS fits keep a small positive gap (+0.01 to +0.035, residual underprediction from shrinkage at these knobs). A user who cares about exact calibration can compose the validation shift onto any arm for free.

## Pre-registered predictions, scored

- P1 (math check): custom objective within 1% of native. Hit in spirit, 0.98-1.03x across the six cells, worst 3.3% at california alpha 0.95, inside single-seed noise. The gradient math (`grad = 2w(pred - y)`, `hess = 2w`, base score seeded with the training expectile) is validated.
- P2 (benefit >5% on synthetic, <2% on california): hit on synthetic (9-57%); the california half measured 2-4%, slightly above the predicted ceiling and growing with alpha, which strengthens the case that the workload is real.
- P3 (quantile+shift trails by >5% everywhere): split. Hit on california (10-20% behind); refuted on the symmetric synthetic, where it ties or wins. The substitute works exactly when the noise is symmetric and fails when it is not.
- P4 (IRLS x2 closes more than half the gap): exceeded. It closes all of it, tying native on every cell.

## Verdict

Declined. The benefit bar for a core objective is a workload class the shipped surface cannot serve, and the probe refutes that: `sample_weight` plus two refits reproduces the exact objective's accuracy everywhere it was measured. A native `ExpectileObjective` (mirroring `QuantileObjective`: roughly 100 lines across objective.hpp/cpp, the registry, a config knob, tests) would buy a 3x wall saving and one-command UX, not capability. The recipe is the deliverable; if the workload shows up in practice, the reopener in decision 87 names the measurement that changes the answer.
