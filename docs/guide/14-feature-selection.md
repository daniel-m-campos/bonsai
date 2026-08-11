# 14. Feature selection

## The idea

Chapter 8 ended with a warning: importance ranks features, but the ranking has no opinion about where to cut. This chapter is about the cut. Selection earns a whole chapter not because trees need help ignoring bad features (mostly they don't, and we measured it), but because two situations force the question whether you like it or not.

**Wide but short.** When features rival or outnumber rows (a 1,024-feature assay with 3,000 samples, a feature store dumped wholesale into training), every node's split search runs a lottery over thousands of candidates, and the math below says the lottery always pays *something*. Wide-short data is where junk features stop being ignored and start being split on, and where an inflated importance ranking is at its most convincing and least trustworthy.

**Inference budgets.** A deployed model pays per feature (computed, fetched, monitored, versioned) and per tree (walked at predict time). When serving cost matters, "should I drop features?" was never the question; the question is "the best model at $k$ features", a budget, not a hypothesis test. Under a budget the real decision becomes *which method* picks your $k$ features, and that is an empirical question. The heart of this chapter is a survey: ten selection methods, one meaningful regression dataset, one shared judge.

## The math: a useless feature never measures zero

The reason selection needs care is an order-statistics effect, not a modeling subtlety.

A pure-noise feature's gain at one node is not zero; it is the **maximum** over every candidate threshold the histogram offers:

```math
\text{gain}_{\text{null}}(f) = \max_{b \,\in\, \text{bins}(f)} \; \text{gain}(b)
```

Each candidate's null gain is a small random variable, but the max of $B$ tries grows with $B$ (for light-tailed gains, roughly like $\log B$). A 255-bin feature buys 254 tries per node, per tree; a forest of depth-6 trees buys thousands. Add the max over *features* that the node then takes, and the winning gain is inflated twice: once per threshold within a feature, once across the $p$ features competing at the node. This is chapter 8's cardinality bias with its mechanism exposed, and it has two practical corollaries:

- **"Importance greater than zero" keeps everything.** Every feature accumulates positive gain by lottery alone, so no absolute threshold separates signal from noise.
- **The inflation grows with $p$ and shrinks with $n$.** More features means more lottery tickets per node; more rows means each candidate's null gain concentrates toward zero. That is why wide-short is the danger regime, and why the same junk features that wreck a 550-row fit are ignored at 5,000 rows.

Because no absolute threshold works, selection methods differ in how they work around the inflation: measure on rows the fit never saw, re-measure after every drop, or sidestep ranking entirely and evaluate feature *sets*. That is the axis the survey below is organized on.

## The trap: grade selection on data it never touched

The oldest and most expensive mistake in this topic (Ambroise and McLachlan, 2002): select features using all the data, then cross-validate the model on the selected set. The gene-expression literature of that era reported near-zero error rates on datasets whose true signal was almost nothing, because the selection step had already seen the labels the "validation" was later scored on. Selection is part of the model. It goes *inside* the validation loop, refit and reselected per fold, or the error estimate is fiction.

Every number in this chapter follows that rule: selection touches only train and validation rows, and the judge is a holdout no method ever saw. The error is easy to commit accidentally: any importance computed on the full matrix, any $k$ tuned against the final test set, any "quick check" that peeks. When a selected-and-validated pipeline looks dramatically better than the all-features baseline, the first hypothesis is leakage through the selection step, not triumph.

## Correlation: what no ranking can do

Chapter 8 warned that correlated features split their importance credit arbitrarily. For selection this is a structural limit, and it divides the topic in two:

- **Noise removal**: drop features carrying no signal. Rankings can do this, because junk sits at the bottom on average.
- **Redundancy removal** (what inference budgets actually want): from a cluster of features sharing one signal, keep one representative. A single importance snapshot is structurally weak here: the cluster's credit is split among its members, each member looks individually mediocre, and dropping one silently shifts its credit onto the survivors, reshuffling the ranking you just used.

Hold that thought through the survey: it predicts which methods should fail at tight budgets (anything that scores features one at a time) and which should win (anything that evaluates *sets*). The measurement below tests exactly that prediction.

## The survey: ten ways to pick k features

Every method here produces a feature ranking, best first. The families, in increasing order of what they get to see:

**Filters** score each feature against the target before any model exists. `corr` (absolute Pearson) and `mutual_info` are the classic pair: free and near-free, but they see each feature *alone*, so they cannot see interactions, and a strong redundant cluster floods their top ranks with copies of the same signal.

**Embedded rankings** read the trained model's own bookkeeping: `gain` and `split` from chapter 8, and mean absolute TreeSHAP attribution (`pred_contribs`, [chapter 15](15-explaining-predictions.md)) computed on the training rows (`shap_train`) or on held-out validation rows (`shap_val`). The validation variant exists because of the inflation math: attribution measured on rows the fit did not memorize discounts lottery winners.

**Validation scoring**: `perm_val`, permutation importance, shuffles one feature at a time on the validation set and charges the feature the resulting error increase. No refits, only predicts, but one pass per feature per repeat.

**Wrappers** put refits in the loop. `rfe_gain` is backward recursive elimination: fit, drop the lowest-gain block, refit, re-rank, repeat down the ladder, so credit that shifts when a feature leaves is re-observed. `forward` is greedy forward selection: start empty, try every remaining feature as the next addition, keep the one that helps validation most, repeat. `rfe_val` is its mirror: start full, and at each step drop the feature whose removal, judged by a cheap refit on the remaining set, costs the least validation error. These two are the only methods that evaluate feature *sets* rather than individual features, and the only ones whose cost scales with $p^2$.

**Calibration methods** (the shadow/Boruta family) attack the no-absolute-threshold problem by appending permuted copies of the features and keeping whatever beats its own copy. They choose a *cutoff*, not a better ranking; we measured the approach against plain truncation in the [selection probe](../../benchmarks/feature-selection-probe-2026-07.md) and it adds nothing on top of the rankings above, so it stays prose here.

## The experiment

**The dataset.** One meaningful regression problem carries the survey: **superconductivity**. 11,340 training rows, 81 real features, target = a material's critical temperature. The features are statistical variants (min, max, mean, weighted mean, range, entropy) of a handful of element properties, so they come in built-in redundancy clusters: exactly the structure the correlation section warns about. A second dataset, QSAR-TID-11 (1,024 fingerprint features, 3,062 rows), covers the wide-short regime in a closing contrast.

**The data split, and who may touch what.** Three parts with one job each. Training rows fit every model. Validation rows are the only data a selection method may read, and they drive early stopping in every fit, so every method works from the identical information budget. The holdout is touched by nothing until judgment: no method, no early stopping, no tuning ever sees it.

**The procedure.** Every method, whatever its machinery, must output one thing: a ranking of the 81 features, best first. The judging loop is then identical for all ten methods. For each budget $k$ on a ladder (64, 48, 32, 24, 16, 12, 8, 4): keep the method's top-$k$ features, refit at the same matched knobs, and read rmse on the holdout. A method's curve is holdout rmse versus $k$, and curve differences are ranking-quality differences, because everything else is held fixed. Each method also carries its selection cost: the wall-clock spent producing the ranking.

**The tie rule.** Refitting the same configuration with a different seed moves rmse by about 2% at these sizes (0.197 here, the benchmark protocol's measured chance band). Two numbers closer than that are ties; only larger gaps count as real.

**Two diagnostics**, measured alongside the curves, that the results below lean on:

- **Set overlap.** At a fixed budget ($k{=}16$), how similar are two methods' chosen feature sets? Measured as shared features divided by total distinct features (the Jaccard similarity): 1.0 means the identical 16 features, 0 means none in common. This separates "similar accuracy because they chose the same features" from "different features arriving at similar accuracy." The results show it as a full method-by-method matrix.
- **Ranking stability.** Refit the gain ranking on 10 bootstrap resamples of the training rows and count how often each feature stays in the top-16. If the ranking itself is noisy, method differences could be luck; if it is stable, they are real.

**One absence in the table below.** `shap_train` is left out of the superconductivity results because its top-$k$ sets coincide with `shap_val`'s there at every budget except one; it stays in the raw rows and in the overlap matrix, where their 1.00 cell records the coincidence.

The protocol above is the whole of it. The survey script and its raw rows are closed evidence and live in git history; [the archive](../method/results/archive.md) names the commit that still reads them.

## Reading the curves

| family | method | wall (s) | k=64 | k=32 | k=16 | k=12 | k=8 | k=4 |
|--|--|--:|--:|--:|--:|--:|--:|--:|
| filter | corr | 0.0 | 9.934 | 10.021 | 10.451 | 10.698 | 11.449 | 13.297 |
| | mutual_info | 2.0 | 9.874 | 10.134 | 11.004 | 11.727 | 11.794 | 13.921 |
| embedded | gain | 4.9 | 9.880 | 9.965 | 10.157 | 10.404 | 10.749 | 12.773 |
| | split | 4.9 | 9.970 | 10.427 | 10.864 | 11.073 | 11.386 | 12.553 |
| | shap_val | 6.9 | **9.856** | 9.996 | 10.183 | 10.420 | 10.697 | 13.709 |
| validation | perm_val | 21.8 | 9.944 | 9.988 | 10.267 | 10.409 | 10.994 | 12.878 |
| wrapper | rfe_gain | 33.0 | 9.880 | **9.954** | 10.255 | 10.500 | 10.743 | 12.671 |
| | forward | 238.8 | - | - | **10.131** | **10.168** | **10.428** | **11.416** |
| | rfe_val | 1090.1 | 9.943 | 10.099 | 10.217 | 10.283 | 10.480 | 11.625 |

(Baseline with all 81 features: 9.856; differences under the 0.197 noise floor are ties. Bold marks the best method at each budget.)

**Start with the one fact that frames everything: no cell beats the baseline.** Every method, at every budget, sits above 9.856. On real, uncontaminated data, selection is a size lever, not an error lever. The reason to walk down this table is a budget, not accuracy hope. So read the curves as answers to one question: *as the budget tightens, whose ranking degrades slowest, and at what cost?*

**Then scan the bold cells.** At $k{=}64$ and $k{=}32$ the best cell wanders (shap_val, then rfe_gain) and almost every method sits within one noise floor of the baseline, so at loose budgets the methods are indistinguishable and the free one wins. From $k{=}16$ down, the bold locks onto forward and never moves again. The family column tells you who follows it down: the set-scoring wrappers stay closest, the individual-scoring rankings trail, and the filters fall away. The rest of this section takes the families in table order.

**Filters: first to fall, and hardest.** At $k{=}16$, `mutual_info` reads 11.004 while the embedded rankings hold near 10.2, four noise floors behind, and neither filter is ever bold. The mechanism is the correlation section's first prediction: a filter scores each feature alone, so one strong cluster floods its top ranks with copies of the same signal. `corr`'s top five contains three variants of the same ThermalConductivity family, and sixteen slots spent on near-duplicates leave no room for complementary signal.

**Embedded rankings: interchangeable, except split.** `gain` and `shap_val` stay within a noise floor of each other at every budget, despite one reading training bookkeeping and the other held-out attribution. The stability diagnostic explains the sameness: gain's top-16 is already stable across bootstrap resamples (a core of 8 features chosen 10 times out of 10), so there is no ranking noise for a fancier signal to repair. `split` is the family's outlier and the worst embedded row at every tight budget, because chapter 8's cardinality bias makes split-count a bad ranking to truncate.

**Validation scoring: error-awareness alone buys nothing.** `perm_val` measures each feature's validation-error impact directly, and its row still tracks `gain` within the noise floor at every budget, at 4.5x the wall clock. Knowing the error is not the missing ingredient; `perm_val` still scores features one at a time.

**Wrappers: refits in the loop, and only the set-scoring pair profits.** `rfe_gain` refits down the ladder but still ranks features individually at each step, and its row tracks `gain` everywhere: with a stable ranking, recursion has nothing to repair. `forward` and `rfe_val` are different: they evaluate each candidate *together with the features already kept*, so they pick complements, which is the correlation section's second prediction. Forward holds every bold cell from $k{=}16$ down, and at $k{=}4$ its 11.416 beats the best ranking's 12.671 by nearly six noise floors. `rfe_val` lands between forward and the rankings at every tight budget, pays 4.6x forward's wall clock, and adds a small loose-budget penalty (its 3,320 validation queries let greedy drops chase noise). The win is bought with compute: forward's 239s is roughly 50x the gain ranking, worth it only where the table shows bold.

### Where the wrapper advantage comes from: the overlap matrix

Accuracy alone cannot say whether two methods succeed for the same reason. The set-overlap diagnostic can: here is the Jaccard similarity between every pair of top-16 sets, rows and columns in the results table's order with `shap_train` restored beside `shap_val`.

| top-16 overlap | corr | mutual_info | gain | split | shap_train | shap_val | perm_val | rfe_gain | forward | rfe_val |
|--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| corr | 1.00 | 0.19 | 0.19 | 0.23 | 0.19 | 0.19 | 0.19 | 0.19 | 0.14 | 0.14 |
| mutual_info | 0.19 | 1.00 | 0.14 | 0.00 | 0.14 | 0.14 | 0.14 | 0.14 | 0.19 | 0.19 |
| gain | 0.19 | 0.14 | 1.00 | 0.19 | 0.88 | 0.88 | 0.78 | 0.78 | 0.28 | 0.45 |
| split | 0.23 | 0.00 | 0.19 | 1.00 | 0.19 | 0.19 | 0.19 | 0.19 | 0.10 | 0.14 |
| shap_train | 0.19 | 0.14 | 0.88 | 0.19 | 1.00 | 1.00 | 0.88 | 0.68 | 0.28 | 0.45 |
| shap_val | 0.19 | 0.14 | 0.88 | 0.19 | 1.00 | 1.00 | 0.88 | 0.68 | 0.28 | 0.45 |
| perm_val | 0.19 | 0.14 | 0.78 | 0.19 | 0.88 | 0.88 | 1.00 | 0.68 | 0.28 | 0.45 |
| rfe_gain | 0.19 | 0.14 | 0.78 | 0.19 | 0.68 | 0.68 | 0.68 | 1.00 | 0.33 | 0.39 |
| forward | 0.14 | 0.19 | 0.28 | 0.10 | 0.28 | 0.28 | 0.28 | 0.33 | 1.00 | 0.33 |
| rfe_val | 0.14 | 0.19 | 0.45 | 0.14 | 0.45 | 0.45 | 0.45 | 0.39 | 0.33 | 1.00 |

Three structures carry the story.

**The dense square in the middle explains the interchangeable tier.** `gain`, `shap_train`, `shap_val`, `perm_val`, and `rfe_gain` overlap at 0.68 to 1.00: five different machineries converging on nearly the same 16 features, which is exactly why their accuracy rows are indistinguishable. Note that this square ignores the family taxonomy: `perm_val` scores validation error and `rfe_gain` refits down the ladder, yet both land on gain's features. Machinery differs; the chosen set does not.

**The winners sit outside the square, and disagree even with each other.** `forward` overlaps the square at just 0.28 and `rfe_val` at 0.45, yet these are the two best tight-budget rows in the results table. They win *by* choosing different features: complements picked in the context of the current set, not the highest individual scorers. Their 0.33 overlap with each other adds a lesson: a good complementary set is not unique, and two wrappers can find two different ones. Forward's first five picks come from five different property families where `corr`'s come from one.

**Low overlap alone is not the win.** The filters and `split` also sit far from the square, near 0.19 (`split` and `mutual_info` share literally nothing), and they are the worst rows in the table. Disagreeing with gain is worthless when the different features are redundant copies of one cluster; it pays only when they are complements. The matrix locates who disagrees, and the results table says which disagreements were worth having.

### The wide-short contrast: where the rankings finally separate

On QSAR-TID-11 (1,024 features, 3,062 rows) the picture changes in the direction the inflation math predicts.

| family | method | wall (s) | k=512 | k=256 | k=128 | k=64 | k=32 |
|--|--|--:|--:|--:|--:|--:|--:|
| filter | corr | 0.0 | 0.892 | 0.927 | 0.965 | 0.996 | 1.122 |
| | mutual_info | 7.5 | 0.889 | 0.896 | 0.941 | 1.011 | 1.149 |
| embedded | gain | 3.4 | 0.895 | 0.890 | 0.918 | 0.973 | 1.081 |
| | split | 3.4 | 0.890 | 0.912 | 0.968 | 1.023 | 1.197 |
| | shap_train | 4.7 | **0.886** | 0.892 | 0.907 | 0.947 | 1.056 |
| | shap_val | 3.7 | 0.887 | **0.889** | **0.904** | **0.944** | 1.063 |
| validation | perm_val | 164.4 | 0.889 | 0.890 | 0.918 | 0.966 | **1.043** |
| wrapper | rfe_gain | 14.6 | 0.895 | 0.895 | 0.922 | 0.954 | 1.049 |

(Baseline with all 1,024 features: 0.883; differences under the 0.0177 noise floor are ties. Bold marks the best method at each budget. The set-scoring wrappers are absent by honest arithmetic: at $p{=}1024$, forward's candidate count is roughly 24,000 fits.)

The embedded rankings stop being interchangeable here, and the bold cells have a new owner. The two SHAP variants take four of the five budgets, with `shap_val` the best ranking through the middle of the ladder: it ties the baseline at $k{=}256$ (a free 4x reduction) and its lead over raw `gain` grows to 1.7 noise floors at $k{=}64$. The mechanism: with many features and few rows, the gain lottery inflates weak features, and attribution measured on rows the fit never memorized discounts them. At the tightest budget the bold moves to `perm_val`, but its lead over the SHAP variants is inside the noise floor, and it pays 164s against shap_val's 3.7s: read down the wall column and `shap_val` is the best accuracy per second on the table by a wide margin.

### What to use

- Budget loose: anything reasonable; use what is free.
- Budget tight, moderate width: forward selection if you can afford its wall clock, else gain top-$k$, knowingly accepting the measured gap.
- Budget tight, wide-short: `shap_val` top-$k$, the best accuracy per second in the survey.
- Never split-count or mutual information for tight budgets on redundant data.
- No binding budget: keep everything; that row of the table is unbeaten.

## In bonsai

Everything the survey used ships already: `importance("gain"/"split")` (chapter 8), exact TreeSHAP via `pred_contribs` (on any rows you choose, which is all `shap_val` is), fast refits for the ladder and the wrappers (chapter 11), and `feature_fraction` (chapter 5) when the goal is regularization rather than removal. The gain-top-$k$ recipe is three lines:

```python
gain = np.asarray(model.importance("gain"))
order = np.argsort(gain)[::-1]
errors = {k: holdout_error(refit(order[:k])) for k in budgets}
```

and forward selection is the same loop with a set instead of a sort. There is deliberately no `bonsai.select` API: a wrapper that cannot beat a sort of numbers you already have is complexity without benefit (measured in [decision 86](../decisions.md#86-honest-shadow-feature-selection-declined-as-an-accuracy-lever-on-all-three-growers-the-gain-ranking-already-has-it-adopted)).

## Try it

Both motivations in one miniature: junk features at small $n$, then the budget curve.

```{.python .run}
import numpy as np
import bonsai

rng = np.random.default_rng(7)
n, p = 900, 8
X = rng.normal(size=(n, p)).astype(np.float32)
w = np.array([3.0, -2.0, 1.5, 1.2, 1.0, -0.9, 0.8, 0.7], dtype=np.float32)
y = (X @ w + np.sign(X[:, 1]) * 2 + X[:, 2] * X[:, 3]
     + 0.3 * rng.normal(size=n)).astype(np.float32)
Xn = np.column_stack([X, rng.permuted(X, axis=0)])  # + 8 shuffled copies
tr, te = slice(0, 600), slice(600, None)

def rmse_at(cols):
    m = bonsai.BonsaiRegressor(n_iters=300, learning_rate=0.05).fit(Xn[tr][:, cols], y[tr])
    return float(np.sqrt(np.mean((np.asarray(m.predict(Xn[te][:, cols])) - y[te]) ** 2)))

m_all = bonsai.BonsaiRegressor(n_iters=300, learning_rate=0.05).fit(Xn[tr], y[tr])
order = np.argsort(np.asarray(m_all.importance("gain")))[::-1]
print("noise features in top-8:", int((order[:p] >= p).sum()))
for k in (16, 8, 4, 2):
    print(f"k={k:>2}: rmse {rmse_at(np.sort(order[:k])):.3f}")
```

The gain ranking puts all eight shuffled copies in the bottom half. The junk costs real accuracy at this size: $k{=}16$ reads 1.357 against 1.295 at $k{=}8$. And the curve has the survey's shape in miniature, flat then a cliff (2.27 at $k{=}4$) as the cut crosses from junk into signal. Double `n` and re-run: the noise gap dissolves, because with enough rows the trees stop splitting on junk by themselves, while the cliff stays, because that part was never about noise.

## Gotchas & war stories

- **The cut is where the losses live.** Every method's curve eventually falls behind the baseline by more than the noise floor on the way down; under a budget that price may be worth paying, but read it off the curve knowingly. Without a binding budget the measured default is to keep everything.
- **Selection bias is quiet and flattering.** The Ambroise-McLachlan trap produces beautiful validation numbers, which is exactly why it survives review. If selection saw data that later graded the model, the grade is void.
- **Importance is also a leakage detector.** An ID-like or timestamp feature at the top of the gain ranking is not a feature to keep; it is a bug report about your data. The wide-short lottery makes this *more* likely on exactly the datasets where the ranking looks most convincing.
- **A survey needs an identity check.** The first run of our RFE arm silently mis-ordered its dropped batches; it was caught because RFE's first elimination step must reproduce gain's top-512 *exactly* (the survivors are the same set), and the cells differed. Build one structural identity into any selection harness; rankings fail quietly otherwise.
- **Wall clocks are part of the answer.** Forward selection's tight-budget win costs 50x the gain ranking; permutation matches SHAP on wide data at 44x the cost. A method table without a cost column is an advertisement, not a survey.
- **Selection is still worth it when accuracy is not the point.** A third of the features means a third of the pipeline to compute, monitor, and explain, and every dataset here hands you a 2.5-4x reduction at no measurable cost. The budget case is the reason this chapter exists.
