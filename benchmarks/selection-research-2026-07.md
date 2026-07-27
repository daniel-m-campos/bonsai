# When can you trust feature selection? A research record (2026-07)

Four linked measurements on one question: when does picking the top-k features by an importance ranking actually work, and what breaks it? The prompt was a practical setting: market microstructure data with a markout target, thousands of features, millions of rows that are far from independent, and relationships that decay. This is a reading document, not guide material; the probe ([`scripts/probe_np_crossover.py`](../scripts/probe_np_crossover.py)) carries each study's predictions, written down before the runs, and this page is regenerated from the raw rows by [`scripts/render_np_report.py`](../scripts/render_np_report.py).

The through-line, up front: selection instruments fail exactly where you would want them most. Scarce independent rows and weak signal corrupt the importance ranking faster than they create headroom for removal, changing relationships punish every backward-looking score, and the defenses that work (rank per time period, screen for distribution shift, cluster before ranking, concentrate under drift) are structural, not statistical refinements.

## Study 1: does shrinking the dataset turn selection into a regularizer?

Setup: superconductivity's 81 real features plus 81 shuffled copies (ground-truth junk), so an oracle refit on the real 81 measures exactly what the junk costs. Rows are subsampled from 11,340 down to 498 (5 draws each below full size); an untouched holdout judges every arm; knobs match the chapter-14 survey. QSAR-TID-11 runs unaugmented as the real wide-short contrast. Deltas are holdout rmse against the keep-everything baseline; negative means the arm wins; `nk` counts junk features kept in the top 81.

| dataset | n train | baseline rmse | oracle | gain top-k | shap_val top-k |
|--|--|--|--|--|--|
| superconductivity | 11340 | 11.153 | -1.297 (nk 0) | -0.698 (nk 8) | -0.697 (nk 9) |
| superconductivity | 3968 | 12.594 | -0.946 (nk 0) | -0.314 (nk 17) | -0.248 (nk 18) |
| superconductivity | 1474 | 14.012 | -0.644 (nk 0) | -0.058 (nk 22) | -0.035 (nk 22) |
| superconductivity | 498 | 16.115 | -0.763 (nk 0) | -0.008 (nk 32) | -0.004 (nk 31) |
| QSAR-TID-11 | 3062 | 0.883 | - | +0.007 | +0.006 |
| QSAR-TID-11 | 1531 | 0.977 | - | +0.007 | +0.009 |
| QSAR-TID-11 | 765 | 1.096 | - | +0.021 | +0.012 |

Finding: the hoped-for crossover does not exist. Removing the junk pays at every size (the oracle column), but the rankings recover only half of that at full size and nothing at n=498, where roughly a third of their picks are shuffled copies. The same row scarcity that makes junk expensive inflates its measured importance, so the instrument degrades exactly as fast as the problem grows. Both pre-registered predictions for this table (junk free at full n; rankings tracking the oracle) were refuted. QSAR agrees from the other side: shrinking rows turns its free k=256 tie into a small loss, never a win.

## Study 1b: the memory recipe at production scale

Setup: synthetic 3.2M training rows by 2,048 features, 64 informative, single L40S GPU. The question a memory-bound n x p forces: can the ranking be fitted on a small row slice so the full matrix never has to exist?

| arm | holdout rmse | fit wall (s) | junk kept of 64 |
|--|--|--|--|
| baseline | 1.58899 | 197.46 | - |
| oracle | 1.58758 | 13.62 | - |
| gain_topk_fullrank | 1.58758 | 13.05 | 0 |
| shap_topk_fullrank | 1.5908 | 13.88 | 0 |
| gain_topk_slicerank | 1.58786 | 44.93 | 0 |
| shap_topk_slicerank | 1.58758 | 43.95 | 0 |

Finding: a six-way accuracy tie (1,984 pure-noise columns cost 0.001 rmse at this row count), and every ranking, including the ones fitted on a 250k-row slice (6% of the data, 2 GB instead of 26 GB), recovered the informative set exactly. Rank on the largest slice that fits, refit on all rows times only the kept columns: measured safe, 15x faster and 32x smaller on device than the full-width fit.

## Study 2: the trust map on market-shaped data

Setup: synthetic data with the three difficulties of a markout target. Features are persistent AR(1) series; the target's noise is a forward moving average of width H, the overlap structure of an H-row markout, so 200k rows carry roughly 200k/H independent examples; signal weights optionally rotate over time (the drifting case). 32 of 128 features carry signal with decaying weights; selection keeps 32; each cell reports junk kept and how many of the 10 strongest true signals were found. Splits are temporal with an H-row gap.

### Gain ranking, fixed relationships

| signal | H=1 | H=20 | H=100 |
|--|--|--|--|
| 25% | 0.0 junk, 10.0/10 | 2.0 junk, 10.0/10 | 5.0 junk, 10.0/10 |
| 20% | 0.0 junk, 10.0/10 | 3.5 junk, 10.0/10 | 7.0 junk, 9.5/10 |
| 15% | 0.0 junk, 10.0/10 | 4.5 junk, 10.0/10 | 8.0 junk, 9.5/10 |
| 10% | 0.0 junk, 10.0/10 | 8.0 junk, 10.0/10 | 11.5 junk, 9.0/10 |
| 5% | 1.5 junk, 10.0/10 | 11.0 junk, 9.0/10 | 16.0 junk, 7.5/10 |
| 1% | 8.0 junk, 10.0/10 | 21.5 junk, 6.0/10 | 23.0 junk, 5.0/10 |
| 0.2% | 18.0 junk, 7.5/10 | 25.5 junk, 2.5/10 | 22.0 junk, 5.0/10 |

### Gain ranking, drifting relationships

| signal | H=1 | H=20 | H=100 |
|--|--|--|--|
| 25% | 7.0 junk, 9.5/10 | 9.5 junk, 8.5/10 | 12.0 junk, 9.0/10 |
| 20% | 7.5 junk, 9.5/10 | 10.0 junk, 8.5/10 | 13.0 junk, 8.5/10 |
| 15% | 7.5 junk, 9.0/10 | 11.5 junk, 8.0/10 | 13.0 junk, 8.5/10 |
| 10% | 8.0 junk, 9.0/10 | 16.5 junk, 7.5/10 | 14.0 junk, 7.0/10 |
| 5% | 9.5 junk, 9.0/10 | 19.5 junk, 6.5/10 | 20.0 junk, 5.0/10 |
| 1% | 15.5 junk, 7.0/10 | 23.0 junk, 3.5/10 | 22.0 junk, 4.5/10 |
| 0.2% | 22.5 junk, 3.0/10 | 24.5 junk, 1.5/10 | 22.0 junk, 5.0/10 |

### Era-averaged ranking, drifting relationships

| signal | H=1 | H=20 | H=100 |
|--|--|--|--|
| 25% | 5.0 junk, 10.0/10 | 11.5 junk, 10.0/10 | 13.5 junk, 9.0/10 |
| 20% | 3.0 junk, 10.0/10 | 11.0 junk, 10.0/10 | 16.5 junk, 9.0/10 |
| 15% | 5.0 junk, 10.0/10 | 14.0 junk, 9.5/10 | 14.0 junk, 9.5/10 |
| 10% | 6.5 junk, 10.0/10 | 15.5 junk, 8.5/10 | 16.5 junk, 8.0/10 |
| 5% | 6.5 junk, 10.0/10 | 18.0 junk, 7.5/10 | 16.0 junk, 6.5/10 |
| 1% | 13.0 junk, 9.5/10 | 18.0 junk, 6.0/10 | 17.5 junk, 6.0/10 |
| 0.2% | 14.5 junk, 7.0/10 | 16.5 junk, 5.0/10 | 15.5 junk, 6.5/10 |

Findings. First, trust collapses along both axes, and the first guess at a combined rule (independent examples times R-squared as a single budget) broke in the forgiving direction: settings with the same product differ threefold in junk kept, because signal strength rescues overlap far more than the sample arithmetic predicts. Second, drift imposes a floor that signal strength cannot remove: about 7 junk in 32 even at 25% signal with no overlap. Third, the era-averaged ranking (rank per time period, average the ranks) is the one defense that works under drift, finding two to three times as many strong signals as the plain ranking at the hard settings; scoring importance on a future data block, predicted to help, measurably did not. Fourth, at the hardest settings even the oracle cannot produce reliably positive out-of-sample R-squared: below a threshold the missing resource is independent data, and no selection method substitutes for it.

## Study 3: a staged pipeline raced against naive selectors

Setup: 200 features with full ground truth. 24 independent signals, each with 3 near-copies (about 0.9 correlated), 16 broken features (signal-correlated until late in training, then the relationship dies and their distribution shifts), 88 pure noise. Every selector keeps 24. The staged pipeline: drop features whose recent distribution shifted (a train-recent classifier's importance), group correlated features, rank by importance averaged over 8 time periods, keep one representative per group. Ablations remove one stage each. Grading is against the answer key (signal groups covered, duplicate slots, broken and noise kept) and by refit R-squared on a purged, strictly later holdout.

### Composition, fixed relationships (mean over 12 cells)

| arm | signal groups / 24 | duplicate slots | broken kept | noise kept |
|--|--|--|--|--|
| corr_filter | 5.9 | 14.0 | 3.9 | 0.2 |
| naive_gain | 15.2 | 6.2 | 0.9 | 1.8 |
| naive_shap | 17.3 | 4.2 | 1.8 | 0.7 |
| era_only | 9.3 | 9.7 | 1.1 | 3.9 |
| pipe_no_era | 19.3 | 1.2 | 0.3 | 3.1 |
| pipe_no_drift | 13.4 | 0.0 | 3.4 | 7.2 |
| pipe_full | 14.0 | 1.1 | 0.7 | 8.2 |
| oracle | 24.0 | 0.0 | 0.0 | 0.0 |
| baseline | 24.0 | - | - | - |

### Holdout R-squared x100 by condition, fixed relationships (mean of 2 draws)

| condition | corr_filter | naive_gain | naive_shap | era_only | pipe_no_era | pipe_no_drift | pipe_full | oracle | baseline |
|--|--|--|--|--|--|--|--|--|--|
| 25% H=1 | 10.22 | 23.08 | 23.43 | 20.35 | 23.04 | 22.38 | 22.60 | 23.61 | 22.91 |
| 25% H=20 | 8.87 | 18.94 | 18.80 | 15.80 | 20.42 | 16.48 | 16.87 | 21.83 | 19.42 |
| 10% H=1 | 3.92 | 8.56 | 8.76 | 7.10 | 8.73 | 8.01 | 8.27 | 8.95 | 8.63 |
| 10% H=20 | 2.32 | 5.24 | 5.18 | 3.94 | 5.69 | 3.37 | 3.17 | 7.30 | 5.73 |
| 5% H=1 | 1.85 | 3.87 | 3.82 | 3.11 | 4.06 | 3.35 | 3.49 | 4.22 | 3.91 |
| 5% H=20 | 0.59 | 1.36 | 1.32 | 0.97 | 1.67 | 1.02 | 1.04 | 2.64 | 1.51 |

### Composition, drifting relationships (mean over 12 cells)

| arm | signal groups / 24 | duplicate slots | broken kept | noise kept |
|--|--|--|--|--|
| corr_filter | 5.9 | 13.8 | 3.9 | 0.3 |
| naive_gain | 10.8 | 7.3 | 3.8 | 2.2 |
| naive_shap | 11.1 | 6.2 | 5.4 | 1.3 |
| era_only | 9.8 | 9.3 | 0.6 | 4.2 |
| pipe_no_era | 14.0 | 0.8 | 2.8 | 6.3 |
| pipe_no_drift | 13.0 | 0.0 | 2.6 | 8.4 |
| pipe_full | 12.9 | 1.0 | 0.9 | 9.2 |
| oracle | 24.0 | 0.0 | 0.0 | 0.0 |
| baseline | 24.0 | - | - | - |

### Holdout R-squared x100 by condition, drifting relationships (mean of 2 draws)

| condition | corr_filter | naive_gain | naive_shap | era_only | pipe_no_era | pipe_no_drift | pipe_full | oracle | baseline |
|--|--|--|--|--|--|--|--|--|--|
| 25% H=1 | 3.60 | -3.69 | -2.68 | 2.36 | -3.22 | -4.88 | 2.29 | 2.61 | -2.62 |
| 25% H=20 | 3.47 | -1.26 | -1.17 | -1.65 | -3.35 | -1.19 | 2.75 | 2.70 | -1.92 |
| 10% H=1 | 1.36 | -0.29 | -0.79 | 0.98 | -1.07 | -0.52 | 1.03 | 1.09 | -1.07 |
| 10% H=20 | 1.19 | -0.26 | 0.28 | 1.20 | 0.65 | 1.39 | 1.14 | 1.15 | -0.26 |
| 5% H=1 | 0.62 | 0.05 | -0.32 | 0.45 | -0.67 | 0.47 | 0.38 | 0.54 | -0.29 |
| 5% H=20 | 0.45 | -0.05 | -0.21 | 0.25 | 0.28 | 0.19 | 0.16 | 0.41 | 0.07 |

Findings. With fixed relationships, the best selector is the pipeline with the era stage removed (drift screen, clusters, plain full-data ranking per group): 19.3 of 24 signal groups against 15 to 17 for the naive rankings, almost no duplicates or broken features, and the first selector in this whole line to beat keep-everything on accuracy, mildly but in 5 of 6 conditions. The era stage drags here because each period's fit has too few rows to rank 200 features. With drifting relationships everything reverses: keep-everything and naive top-k go negative (the model anti-predicts, having learned dead relationships and the broken features), the era stage becomes the thing that saves the pipeline, and the crude correlation filter, owner of the objectively worst feature list, produces the best model of all, beating even the oracle refit at every drift condition. Its duplicate-heavy list amounts to a concentrated bet on the few strongest, most persistent signal groups, and under rotating weights concentration transfers better than coverage.

Scoring the written predictions: correlation-filter flooding confirmed; the drift screen's value confirmed (naive keeps 4 to 5 broken features under drift, the screen cuts that to about zero, the no-screen ablation lands between); the pipeline's predicted coverage of 22 or more was missed (best was 19.3); the ablation ordering held under drift and inverted under static, which study 2 had foreshadowed and the design under-weighted. The first run of this race carried a clustering index bug, caught because the pipeline showed duplicate slots that are impossible by construction; it was fixed and the race rerun in full.

## What a practitioner should take away

1. Before believing any importance ranking, estimate your effective sample (rows divided by label-overlap width, discounted for cross-sectional correlation) and your realized R-squared, and place yourself on the study-2 tables. Strong signal forgives a lot; weak signal is forgiven nothing.

2. Always: purged temporal splits, a distribution-shift screen, and correlation clustering with one representative per group. These are cheap, never hurt in any measured cell, and remove the two failure modes (redundancy, broken features) that importance scores cannot see.

3. Do not commit to one ranking. Rank once on all data and once averaged per time period; which one is right depends on whether relationships are currently stable, which is observable on a rolling out-of-sample basis and not assumable in advance.

4. Under suspected drift, prefer concentration over coverage, and judge any selection policy by its bad draw, not its average: the same drifting configuration produced +5% and -11% R-squared across two runs for the naive approaches, while the concentrated and era-averaged arms never went meaningfully negative.

5. Selection earns accuracy only at the edges (removing planted or broken features under drift); its reliable payoff everywhere else is size: memory, latency, and pipeline surface, bought at a measured accuracy tie.

## Caveats

Two draws per cell throughout; drift cells have large draw-to-draw spread, so means there are indicative and the risk framing above is the honest one. All generators are synthetic and milder than real microstructure. One shape per study (feature counts, persistence, era count are fixed); the thresholds are properties of these shapes, not universal constants. A ratio metric (share of oracle R-squared) was retired mid-analysis after it misled on cells with tiny denominators; absolute numbers are reported instead.

