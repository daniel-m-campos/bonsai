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

One meaningful regression dataset carries the survey: **superconductivity**, 11,340 training rows and 81 real, physically meaningful features (min/max/mean/weighted-variant families of element properties: exactly the redundancy-cluster structure the correlation section worries about), predicting a material's critical temperature. The wide-short regime gets a footnote below on QSAR-TID-11 (1,024 fingerprint features, 3,062 rows).

The judge is the same for every method: take the method's ranking, refit at matched knobs on its top-$k$ for each rung of a budget ladder ($k$ = 64 down to 4), and read rmse on an untouched holdout. Selection cost is wall-clock for producing the ranking. Ties are declared by a noise floor: refitting the same configuration with a different seed moves rmse by about 2% at these sizes (0.197 here, the benchmark protocol's measured chance band), so two cells closer than that are treated as equal, and only larger gaps count as real differences. Full protocol, tables and raw rows: [`benchmarks/selection-survey-2026-07.md`](../../benchmarks/selection-survey-2026-07.md) and [`scripts/probe_selection_survey.py`](../../scripts/probe_selection_survey.py).

## Reading the curves

![Selection-method survey: holdout rmse vs features kept](../method/assets/selection-survey.svg)

| method | wall (s) | k=64 | k=32 | k=16 | k=12 | k=8 | k=4 |
|--|--:|--:|--:|--:|--:|--:|--:|
| corr | 0.0 | 9.934 | 10.021 | 10.451 | 10.698 | 11.449 | 13.297 |
| mutual_info | 2.0 | 9.874 | 10.134 | 11.004 | 11.727 | 11.794 | 13.921 |
| gain | 4.9 | 9.880 | 9.965 | 10.157 | 10.404 | 10.749 | 12.773 |
| split | 4.9 | 9.970 | 10.427 | 10.864 | 11.073 | 11.386 | 12.553 |
| shap_val | 6.9 | 9.856 | 9.996 | 10.183 | 10.420 | 10.697 | 13.709 |
| perm_val | 21.8 | 9.944 | 9.988 | 10.267 | 10.409 | 10.994 | 12.878 |
| rfe_gain | 33.0 | 9.880 | 9.954 | 10.255 | 10.500 | 10.743 | 12.671 |
| forward | 238.8 | - | - | 10.131 | 10.168 | 10.428 | 11.416 |
| rfe_val | 1090.1 | 9.943 | 10.099 | 10.217 | 10.283 | 10.480 | 11.625 |

(Baseline all 81 features: 9.856. `shap_train` is omitted from the table because it is the same ranking as `shap_val` here to within one rung; both are in the report.)

Five readings, and each one teaches a section of this chapter back:

1. **Nothing beats the baseline.** Every cell sits above 9.856. On real, uncontaminated data, selection is a size lever, not an error lever; the reason to walk down this table is a budget, not accuracy hope.
2. **Loose budgets are free and method-indifferent.** Down to $k{=}32$ (a 2.5x reduction) every model-based ranking ties the baseline (the gap is smaller than the noise floor), and even zero-cost `corr` does. Paying 33s for recursive elimination to reach a place a free filter reaches buys nothing.
3. **Tight budgets separate the families.** At $k{=}16$ the embedded rankings hold ~10.2 while `mutual_info` is at 11.0, worse by four times the noise floor: filters flood their top ranks with copies of one strong cluster (their top-5 here contains three variants of the same ThermalConductivity family). `split` is the worst embedded ranking at every tight rung, chapter 8's cardinality bias made visible.
4. **The one-shot model rankings are interchangeable, and so are their refinements.** `gain`, `shap_val`, `perm_val`, `rfe_gain` stay within a noise floor of each other all the way down; recursion and validation-scoring have nothing to repair here because (per the bootstrap diagnostic) the gain ranking is already stable, with a core of 8 features chosen in 10 of 10 resamples.
5. **The set-evaluating wrappers win every rung at $k \le 12$, by more than the noise floor.** Forward leads and `rfe_val`, its backward mirror, lands between forward and the rankings at 4.6x forward's cost: evaluating candidates *in the context of the current set* is the active ingredient, and validation-error scoring alone (which `perm_val` already had) buys none of it. 10.168 vs 10.404 at $k{=}12$ (a 0.24 gap against the 0.197 floor); 11.416 vs 12.553-13.921 at $k{=}4$, where its lead over the best ranking is 1.14 rmse, nearly six times the floor. This confirms the correlation section's prediction: forward evaluates *sets*, so it picks complements where every ranking picks redundant twins. The overlap diagnostic makes it concrete: forward's top-16 shares only Jaccard 0.28 with gain's, and its first five picks span five *different* property families where `corr`'s span one. It paid 239s, about 50x the gain ranking, and at tight budgets that cost bought real accuracy.

**The wide-short footnote (QSAR-TID-11, 1,024 features).** At 81 features the rankings were interchangeable; at 1,024 they finally separate, in the direction the inflation math predicts. `shap_val` becomes the best ranking: it ties the baseline at $k{=}256$ (a free 4x reduction) and leads raw `gain` by 1.4 to 2.9 times the noise floor at $k \le 128$. The mechanism: attribution on rows the fit never memorized discounts the lottery winners that wide-short data mass-produces. `perm_val` matches it only at the tightest rung and pays 164s against `shap_val`'s 3.7s. Forward is absent by honest arithmetic: at $p{=}1024$ its candidate count is ~24,000 fits.

**What to use, then.** Budget loose: anything, so use what's free. Budget tight at moderate width: forward selection if you can afford its wall, else gain top-$k$ knowingly. Budget tight and wide-short: `shap_val` top-$k$, the best accuracy-per-second in the survey. Never split-count or mutual information for tight budgets on redundant data. And if no budget binds: keep everything, because that row of the table is unbeaten.

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
- **A survey needs an identity check.** The first run of our RFE arm silently mis-ordered its dropped batches; it was caught because RFE's first rung must reproduce gain's top-512 *exactly* (the survivors are the same set), and the cells differed. Build one structural identity into any selection harness; rankings fail quietly otherwise.
- **Wall clocks are part of the answer.** Forward selection's tight-budget win costs 50x the gain ranking; permutation matches SHAP on wide data at 44x the cost. A method table without a cost column is an advertisement, not a survey.
- **Selection is still worth it when accuracy is not the point.** A third of the features means a third of the pipeline to compute, monitor, and explain, and every dataset here hands you a 2.5-4x reduction at no measurable cost. The budget case is the reason this chapter exists.
