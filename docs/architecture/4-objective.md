# 4. Objective

> **Status:** Phase 1 design, ratified in [`../decisions.md`](../decisions.md) entries 21–25.

## Why MSE and binary logloss together

The proposal ([`../proposal.md` §1](https://github.com/daniel-m-campos/bonsai/blob/main/docs/proposal.md)) commits to both in Phase 1 for the same reason `3-tree.md` lands two trees: two implementations of an open concept catch single-impl assumptions that one wouldn't. Concretely, MSE has constant unit hessian (`h = 1` per row), so any code path that conflates "row count" with `sum_hess` keeps working under MSE and quietly breaks under logloss (`h = p * (1 - p)`, in (0, 0.25]). Shipping both exercises `min_child_hess` (decision 20), the leaf-value formula `-G / (H + λ)` (decision 14 / `3-tree.md`), and the determinism contract (decision 7) under both flat and variable hessian regimes.

The cost is one extra concrete `Objective` type plus a logloss-aware test path. Sample weighting is out of scope for the concept itself (see "What's not here").

## The `Objective` concept

```cpp
template <typename T>
concept Objective = requires(floats_view preds, floats_view targets,
                             floats_out grad, floats_out hess) {
    { T::compute(preds, targets, grad, hess) } -> std::same_as<void>;
    { T::eval(preds, targets) }                -> std::same_as<float>;
};
```

Two static functions. No instance state — matches `SplitFinder`'s dispatch shape (decision 14) so a `Booster<Gr, Obj, ...>` can fix both at compile time.

**`compute`** — fills `grad` and `hess` per row from raw scores and targets. Inputs and outputs are all length `n_rows` 1D spans (decision 23 pins single-output for Phase 1). Output is **written**, not accumulated (decision 24): callers don't need to zero the buffers first.

**`eval`** — returns the mean per-row loss as a scalar `float`. Used for training-log lines and (eventually) early stopping. No metric concept split for MVP — adding a separate `Metric` later is non-breaking since the concept doesn't expose it.

`preds` are **raw scores** (decision 24) — any real value in (−∞, ∞), never pre-transformed by the booster. Objectives that need a probability (logloss) apply the link inverse internally to derive grad/hess/loss; the booster keeps its accumulator on the raw-score scale and applies the link a second time only at predict time, on the outside. This matches xgboost/LightGBM and keeps the boosting math additive on the score scale.

### Why `void` and out-spans, not return-by-value

The Booster pre-allocates a single `(grad, hess)` buffer pair sized to `n_rows` and reuses it across iterations. Returning a fresh buffer per call would allocate `O(n_rows · n_iters)` and defeat cache reuse. Out-spans also let the booster overlay sample-weight multiplication or row-subsampling (decision 12, sampler) on the same buffer in place.

### Signatures, by component

| Caller | Sees |
|---|---|
| `Booster<...>::update_one_iter` | both `T::compute` (every iter) and `T::eval` (logging cadence) |
| Grower / SplitFinder | nothing — they consume the `grad`/`hess` spans, not the objective |
| Predict path | nothing — link inverse (sigmoid for logloss) lives in the booster's predict, not on the objective (see `5-booster.md`) |

## `MSEObjective`

```cpp
struct MSEObjective
{
    static void compute(floats_view preds, floats_view targets,
                        floats_out grad, floats_out hess);
    static float eval(floats_view preds, floats_view targets);
};
```

Loss: `L = ½ Σ (p − y)²`. Per-row:

- `grad[i] = p[i] − y[i]`
- `hess[i] = 1.0F`
- `eval` returns mean squared error: `(1/n) Σ (p[i] − y[i])²`. (Mean rather than half-mean, matching xgboost's `rmse` metric without the square root — RMSE is a wrapper concern.)

Note the asymmetry: `grad = p − y` is the derivative of the un-normalized `½ Σ (p − y)²`, while `eval` reports the normalized `(1/n) Σ (p − y)²`. The two scalars differ by a constant factor of `2/n`. This is the xgboost convention — the eval metric is a human-readable training-log number, and the regularization knobs (λ in the leaf formula `-G/(H+λ)`, decision 14) are tuned against the gradient scale, not the eval scale.

The constant unit hessian is the regime where `min_child_hess` collapses onto `min_data_in_leaf`-style row-count gating. That's a useful baseline for the parity tests.

## `LogLossObjective`

```cpp
struct LogLossObjective
{
    static void compute(floats_view scores, floats_view labels,
                        floats_out grad, floats_out hess);
    static float eval(floats_view scores, floats_view labels);
};
```

Binary cross-entropy with the link folded into the math. targets are {0, 1}. Per-row, with `p = sigmoid(score)`:

- `grad[i] = p − y`
- `hess[i] = p · (1 − p)` (in `[0, 0.25]` — open at 0 in real math, but
  collapses to exactly `0` in float once `|score|` is large enough for
  `exp` to saturate; `min_child_hess` (decision 20) catches it downstream)
- `eval` returns mean cross-entropy on probabilities: `(1/n) Σ −[y·log(p) + (1−y)·log(1−p)]`.

**Sigmoid stability.** Compute via `std::log1p(std::exp(−|x|))` and
sign-branch — the textbook numerically stable form. No explicit score clipping needed; the IEEE behavior of `exp` saturates to 0/∞
gracefully at large `|score|`, and `p · (1 − p)` underflows to 0,
which `min_child_hess` (decision 20, default 1.0) catches before that row's grad/hess could destabilize a split. Eval uses the same stable form to avoid `log(0)` near saturation.

**Label validation** is the booster's job, not the objective's (decision 22): `Booster` checks `targets ⊂ {0, 1}` at fit time. `compute` assumes well-formed inputs.

## What `Objective` does **not** own

Three things that look like Objective concerns but live elsewhere:

1. **Initial score / bias.** The first tree starts from a constant prediction (mean of targets for MSE; log-odds of positive rate for logloss). This is the booster's responsibility — it owns the prediction accumulator and decides whether the bias comes from config or is computed from targets. The decision shape lives in `5-booster.md`.
2. **Link inverse / `transform`.** Sigmoid at predict time is the booster's `predict` path, not an objective method. Trees store raw-score leaves (decision 10 / 3-tree.md "Leaf values"); the booster sums them and applies the link only at the outermost call.
3. **Sample weights.** A per-row `weights` span (when present on the `Dataset`) multiplies `grad` and `hess` after `compute` returns. Wrapping every objective with weight logic inside `compute` would force every future objective to remember the multiplication; doing it once in the booster keeps `Objective` impls focused on loss math. Pin in decision 25.

The first two are "where does the loss geometry live" calls that the booster has to make anyway (e.g. choosing initial score from config vs. targets). The third is a single 2-line loop on the buffer the objective just wrote.

## Determinism

`compute` and `eval` are pure functions of their inputs — no thread- locals, no mutable globals, no allocation. Cross-thread-count reproducibility (decision 7) holds trivially: same inputs, same outputs, in any order. The booster, not the objective, decides how to parallelize the loop (e.g. `OpenMPBackend::for_each_row`); the choice doesn't change per-row results.

For determinism of the floating-point sum inside `eval`: the booster pins reduction order at fixed thread count, per the determinism contract. Within `eval` a serial accumulator is fine for Phase 1; parallel `eval` is `7-parallel.md`'s problem.

## Validation, errors

No constructors (concepts are stateless). The booster validates targets at fit time and throws `ConfigError` on label-range violation for logloss. `compute` itself does no input checking — its preconditions are documented as "targets in {0,1} for logloss; raw scores otherwise."

## Testing

Per the test-naming convention in [`README.md`](README.md):

- `MSEObjective: compute writes grad = p − y and hess = 1`
- `MSEObjective: eval returns mean squared error`
- `LogLossObjective: compute matches sigmoid math for canonical inputs`
- `LogLossObjective: compute is numerically stable for extreme scores`
- `LogLossObjective: eval matches reference cross-entropy`
- `LogLossObjective: hess in (0, 0.25] across the score range`

Parity tests against xgboost/LightGBM live at the booster level (end-to-end), not here.

## What's not here

- **Multi-class / softmax.** Decision 23 pins single-output for Phase 1. Multi-class is a Phase 4 extension; the natural shape is a K-output concept overload returning `grad`/`hess` per class. Signature change is contained: it touches `Objective`, `Booster`, and `Tree` (which would need K-output leaves). Not free, but well- isolated.
- **Custom user-supplied objectives.** Out of scope for Phase 1. Adding one means satisfying the concept; no registry needed since dispatch is static at the `Booster` template parameter.
- **Quantile, Huber, MAE** — since shipped (decision 35): objectives became Config-constructed instances so parameterized losses (`[objective] huber_delta / quantile_alpha`) carry state; MSE/LogLoss kept static methods (statics satisfy instance-call syntax) and gained trivial ctors. Constant-hessian caveat: MAE/quantile leaves are gradient means, not residual medians — leaf renewal (feature_gap row 10) is the known follow-up. **Tweedie, Cox** remain future.
- **Metric concept.** Bundled into `Objective::eval` for MVP (single-metric, the training loss). Splitting into a `Metric` concept happens when early stopping needs auxiliary metrics (e.g. AUC alongside logloss).
- **Sample weighting.** Booster-side concern (decision 25); see `5-booster.md`.
- **Initial score / link inverse.** Booster-side (decision 22); see `5-booster.md`.

## Cross-references

- [`../decisions.md`](../decisions.md) entries 21–25 (this doc's ratifying decisions); entries 7 (determinism), 14 (static dispatch shape), 20 (regularization knobs) for context.
- [`../proposal.md` §1, §3](https://github.com/daniel-m-campos/bonsai/blob/main/docs/proposal.md) for the Phase 1 scope commitment to MSE + logloss and the static-dispatch ethos.
- [`3-tree.md`](3-tree.md) for the leaf-value formula `-G / (H + λ)` that consumes the grad/hess this concept produces.
- `5-booster.md` (TBD) for initial score, link inverse, sample weighting, and the `update_one_iter` loop that drives `compute`.
