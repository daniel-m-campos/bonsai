# 21: Component-timing standings: the DAG constants become the ledger

> **Status:** design (issue #323, sub-issue of #318). Doc 16 made placement costs a first-class design object with measured constants; this doc makes those constants standing measurements with the same supersession discipline as every other published number. Nothing here changes what a fit computes.

## The problem this solves

The standings answer "who wins this cell" with end-to-end grids, and the grids are why a refresh costs hours: every cell of every axis re-fits everything, even though most changes move exactly one node of doc 16's DAG. The 2026-08-04 refresh spent 60 percent of its wall clock on wide-CPU cells that no GPU-side change can move (#320 addresses the scheduling), but the deeper issue is that end-to-end grids are the wrong instrument for pricing a change: they answer "did anything move" at the cost of re-measuring everything, where a component axis answers "what moved and by how much" at the cost of one profiled anchor run.

The macro decomposition already landed: every perf row now carries `ingest_s` and `train_s` (decision 101), so the fixed-versus-variable split is published. The micro level is the tree loop itself: fixed setup (mapper fit, bin, plane build) plus a per-tree cost per grower, which is what decision 98's campaign measured by hand and what this doc makes standing.

## The design

**One new registry axis, `components`.** Per refresh, one profiled run per (grower, anchor cell) captures the node costs doc 16 names: `mapper_fit`, `bin`, `gh_upload`, `hist_build`, `find`, `partition`, `finalize`, `eval`, plus the residual. The instrument is the existing counter families (`BONSAI_GROW_PROFILE`, `BONSAI_CUDA_PROFILE`, `BONSAI_INGEST_PROFILE`, `BONSAI_FIT_PROFILE`), priced at 0 to 2 percent of fit (decision 98), which is why the same run cannot serve as the timing anchor.

**The clean anchor rides beside the profiled run.** Each refresh fits the anchor cell twice per grower: once unprofiled (the published `fit_s`, unchanged meaning) and once with counters on (the components). Both are committed rows in one dated file, `arm` distinguishing them.

**Row schema.** Standard bench provenance (schema v1: `git_sha`, `host`, `cell`, `knobs_hash`) plus `component`, `seconds`, `arm` in {`clean`, `profiled`}, `rep`. The renderer presents the matrix form, components by grower, one anchor cell per division.

**The conservation gate, mechanized.** Doc 16's rule ("an unexplained gap is the next target, not noise") becomes a check in the refresh: the profiled run's component sum must land within a stated band of its own wall clock, and the profiled wall clock within the counter-cost band of the clean anchor. A failure blocks supersession the way the parity gate does, because dark matter in the components means the attribution is fiction.

**Per-tree extraction.** The micro model is `fit = setup + n_iters * per_tree`. Two profiled runs at different iteration counts (100 and 200) give both terms by difference, and the linearity error is itself a published number: if the two-point fit misses the measured 100-iteration wall by more than the band, the model is flagged rather than trusted (early-stopping and cache-warming effects would show here).

## What changes downstream

**Change pricing.** A PR touching one node cites the component refresh: the change clock (#320's routine tier) runs the components axis plus the A/B anchors, minutes instead of hours, and the grids move to the release clock. The decision-98 pattern (hand-run profile ladders per campaign) becomes the standing ritual instead of a per-campaign improvisation.

**Attribution work gets an instrument.** The open train-side questions (the XGBoost depthwise training deficit named by decision 101, the GPU eval overhead of #326) are component questions; a standing components axis means every refresh re-answers them for free.

**Grid diet follows, not leads.** The grids stay published and refreshed on the release clock until the components axis has two release cycles of agreement with its anchors; only then does #322-style thinning of grid cells lean on components as the replacement instrument. End-to-end anchors remain the ground truth the components must sum to, permanently.

## What this deliberately does not do

No cross-pod absolutes: components obey the same-pod rule like every timing. No counter-on numbers presented as fit times: the clean anchor is the only published wall clock. No per-cell component grids: one anchor cell per division is the scope; a second cell is admitted only when a real change shows component shares moving with shape (the width wall of #217 is the likely first customer). No CPU-arm components beyond bonsai's own: reference libraries expose no comparable seams, so the components axis is bonsai-introspective and the cross-library comparison stays end-to-end.

## Implementation sketch

`scripts/standings_refresh_pod.sh` gains a `components` axis running the profiled pairs; the counter parsing lives beside the existing profile-line greps. `scripts/render_results.py` gains the matrix section and the conservation statement. `scripts/check_standings.py` learns the axis. The registry entry carries `hash_set`-style provenance like the quality axes after #321. Measurement lands in the same refresh PR shape as every axis.

## Rejected

A per-iteration timer inside the engine (a per-tree callback would perturb exactly what it measures and duplicate the counters); NVTX/nsys as the instrument (right tool for kernel work, wrong tool for standings: not parseable into rows without tooling the pod image lacks, and the counters already close against event-timed spans per doc 16); publishing profiled wall clocks as fit times (decision 98 priced the counters as nonzero); components for reference libraries via external profilers (unownable measurement, and the fairness rules cannot extend to instruments the libraries do not sanction).
