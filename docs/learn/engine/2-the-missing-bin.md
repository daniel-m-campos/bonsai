# E2. The missing bin

Decision 73 shipped a new capability. An acceptance test failed on the first try. The failure exposed a train/predict routing skew that every fitted model had been carrying, and closing it moved the standings. This case is why an acceptance test is worth more than the feature that prompted it.

## The setting

Decision 73 shipped user-supplied bin edges: `bonsai.Dataset(X, y, bin_edges={col: edges})` bins named features at domain cut points you supply. This is the artifact-construction door for domain-mandated bins. The plumbing added no model-format change, no hot-path branch, and no CUDA change: a `from_edges` mapper beside the loader path.

Domain edges are a reproducibility statement. The bin boundaries live inside the artifact, so a saved model bins new data exactly as it trained, with no external transform to drift. That is the point of putting the edges in the Dataset instead of a preprocessing script.

The feature cleared bonsai's admission gate on roadmap signal, not on an accuracy claim. Decision 67 had asked for a workload needing domain bins in the artifact, and the owner ranked it onto the roadmap. Accuracy stayed a non-claim; decision 67 had already measured bin choice as saturated.

Background on binning is in [guide chapter 2](../../guide/2-binning-and-histograms.md). The short version: each feature is discretized into at most 255 quantile buckets once, and the last bin is reserved for missing values (NaN). Splits are scored by scanning the bins, and the missing bin is excluded from that scan.

## The test that failed

Every admitted capability needs an acceptance test. This is not a unit test. It exercises the whole path a user takes: construct with domain bands, fit, predict on raw values, save, load, and predict again byte-identically. It asserts three things: within-band invariance, cross-band separation, and right-inclusive edge membership.

The first implementation appended only the `+inf` sentinel, and the test failed. Bands above and below the last supplied edge were inseparable. No tree could split them apart.

## A convention, not a bug

The root cause was not a coding error. It was an engine convention meeting a new use.

The split scan never offers the last real bin as a candidate. For a fitted column that cut is degenerate: the observed maximum defines it, so a split there separates nothing. The convention had gone unquestioned because fitted columns never advertised a band above their maximum. User edges did, on the first fit. For user edges the band above the last edge is a domain statement, so it must be splittable.

The fix appends a `FLT_MAX` cut to close the top band as a real bin, then the `+inf` sentinel to keep the missing bin NaN-only. With k edges you get k+1 splittable bands, and NaN still routes to its own bin. The edges are validated at their own named entry, so the loader's trusted path stays branch-free. They must be finite, below FLT_MAX, strictly increasing, and non-empty. Bad column indices and duplicates throw a config error.

## The skew it exposed

Then the same closer was applied to every fitting path, not just user edges (decision 74, issue #155). The reason was a train/predict routing skew that fitted columns had carried all along.

Three leaks fed finite values into the NaN sentinel. The stride path's top tail is worth about one mean bin of rows. The greedy path's final group is an entire heavy run when the maximum is a capped value. The third is every row above the 200k-row bin sample's maximum.

Those rows trained as missing, routed by the learned `default_left` branch. But at prediction time they were routed by a raw threshold comparison, landing right of everything. The same row took different branches in training and deployment.

The early-stopping path already scored raw values, so it sat on the deployment side of the skew. Only the training route was wrong, which is why the skew hid: the two scoring paths disagreed silently. This is the core lesson in one sentence: a correctness asymmetry was hiding behind a quality metric that looked flat.

The result is one invariant, stated in a sentence: no finite value ever bins as missing, on any path. The same closer serves user edges and fitted columns, so the sentinel bin means the same thing everywhere. A tree may now split real values away from missing at the last cut, a candidate that did not exist before.

## The mechanism, then the field

The fix was measured two ways. Synthetics isolate each leak; the real suite measures the field.

The probe runs both growers at campaign knobs: 200 iterations, learning rate 0.05, depth 6, 255 bins, seed 42, each arm in its own process.

The synthetics are decisive, and the capped case is the realistic catastrophe. The synthetic is a sensor clipped at its ceiling: the cap value is 10% of the rows and carries signal. Training it as missing loses half the variance, so r² is 0.483. With the closer the cap bins as a real top band, and r² is 0.988. The two lighter leaks show the same sign: a rare top-tail signal gains +0.16, and signal above the sampled max at 1M rows gains +0.008.

Clipped sensors, capped amounts, and 999-coded maxima are common shapes the quality suite happens not to contain.

The real suite is chance-band flat. Higgs moves ±0.0001, airline ±0.0006, and California ±0.001 with opposite signs per grower. Amazon's −0.002 sits inside that dataset's own 0.0034 spread across `max_bin` 253 to 256 with no closer at all. That null-perturbation spread is the honest error bar, and the delta is inside it.

So the fix is adopted on robustness, not accuracy. It removes a works-versus-doesn't failure on capped columns, at the cost of a budget slot that was already reserved.

The models still change wherever a leak existed. The canonical hashes moved, and the California pin shifted 0.71725 to 0.7153, inside the band. An all-NaN column now emits a real bin plus the sentinel, so a splittable bin exists even when nothing was observed.

## The surprise

The Grinsztajn standings were re-run to confirm the change was neutral. The expectation was chance-band. The result was not.

Mean rank went from 1.73 to 1.44. Outright wins went from 27 to 36 of 55. Second-or-better went from 44 to 50, and last-place finishes from 1 to 0. Head-to-head against LightGBM went from 37 to 46 wins, and against XGBoost from 42 to 48.

## Why small nudges moved ranks

A rank jump this size from a robustness fix demands an honest decomposition. The per-dataset values mostly did not move: 43 of 55 stayed inside 0.001. Eight improved beyond it, four regressed, and the worst regression was `house_16H` at −0.0040.

Three things multiplied those small nudges into rank movement. The suite caps every task at 10k rows. At that size the greedy binning path is nearly universal, with a mean bin of about 39 rows, so 461 of 495 fits changed. The suite is full of photo-finish second places. And the closer's small nudges broke those near-ties systematically one way.

The durable facts are the head-to-head counts and the zero last places, not the mean-rank delta. Rank tables amplify near-ties, so a rank delta without the per-dataset distribution is not evidence. The decision-55 residual also narrowed: `year` recovered +0.0036 of its +0.0066 gap to XGBoost.

## What it teaches

- **Acceptance tests find what unit tests structurally cannot.** The skew never showed on small fixtures, because they never produce a near-empty top band. The test that caught it fits with domain bands and predicts on raw values, end to end, with no external transform.
- **A correctness asymmetry can hide behind a quality metric.** On the flat real suite the closer looked like noise. The synthetics showed a capped column losing half its variance, r² 0.483 to 0.988, because the failure is works-versus-doesn't, not a tuning delta.
- **Distrust rank deltas without the per-dataset distribution.** Mean rank moved from 1.73 to 1.44 on value changes that were mostly inside 0.001. The move was real but small, amplified by a suite of near-ties; the head-to-head counts are what transfers.

## The record

- Decisions: [73](../../decisions.md) (explicit bin edges, and the failing acceptance test) and [74](../../decisions.md) (the FLT_MAX closer on every fitting path).
- Evidence: [the missing-bin closer probe](../../../benchmarks/missing-bin-closer-2026-07.md), with the synthetics table, the real suite, and the standings re-validation.
- Background: [guide chapter 2](../../guide/2-binning-and-histograms.md) on binning, the missing bin, and the right-inclusive edge convention.
