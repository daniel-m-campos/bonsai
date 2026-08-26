# 13. Categorical features

## The idea

A tree splits on `x <= threshold`, so a feature's values need an order, and category IDs (`MGR_ID = 71705`) have none.

The industry offers three answers: one-hot expansion (explodes width), native set splits in the split scan (LightGBM: the tree learns *subsets* of categories), and target statistics (CatBoost: replace each category with a number summarizing its relationship to the label, which *gives* it an order).

bonsai's answer is the third, deliberately outside the core: an encoder in the Python package, not a node type in the engine (decision 58). We measured the alternatives before choosing: the section at the end has the table, and [architecture doc 17](../architecture/17-categorical-splits.md) preserves the full native-split design we priced and declined.

## The math

The **target statistic** for category $c$ with prior $p = \bar{y}$ and smoothing weight $a$:

```math
\text{TS}(c) = \frac{\sum_{i:\, c_i = c} y_i + a\,p}{\left|\{i : c_i = c\}\right| + a}
```

Small categories shrink toward the prior; big ones speak for themselves. Trees now split on "categories whose response rate exceeds t": one cut does the work of an arbitrary category subset.

Computed naively over the whole training set, this **leaks**: row $i$'s own label sits inside its own feature. The pathology is sharpest for a category appearing once: its TS is $\tfrac{y_i + ap}{1 + a}$, a near-copy of the label. Training loss looks wonderful; the test set, whose labels obviously don't appear in its features, sees an uninformative column. K-fold encoding reduces the leak but not the distribution shift: every training row is still encoded with statistics from a world where its fold is missing.

The fix is CatBoost's insight, **ordered** target statistics: draw a permutation $\sigma$ and let each row see only its predecessors,

```math
\text{TS}(i) = \frac{\sum_{j:\, \sigma(j) < \sigma(i),\, c_j = c_i} y_j + a\,p}{\left|\{j : \sigma(j) < \sigma(i),\, c_j = c_i\}\right| + a}
```

Row $i$ never sees $y_i$: the encoding is *causal*, like a time series. The first-visited row of every category gets exactly the prior (zero evidence), later rows get progressively sharper estimates, and the train-time feature distribution now matches what a deployed model sees: statistics accumulated from data that came before.

## In bonsai

[`OrderedTargetEncoder`](../../python/bonsai/encoding.py): categories arrive as numeric codes in float cells (`.cat.codes` for pandas users), NaN is one shared missing category:

```python
enc = bonsai.OrderedTargetEncoder(columns=[0, 1, 8], prior_weight=10.0)
Xtr_enc = enc.fit_transform(Xtr, ytr)   # causal: permuted running means
Xte_enc = enc.transform(Xte)            # full-train statistics
```

The `fit_transform`/`transform` asymmetry *is* the leakage story: training rows get causal encodings, everything after training gets the full statistics, the same convention sklearn's `TargetEncoder` adopted, for the same reason.

`keep_codes=True` (default) appends the raw code columns after the features, so trees get both views: the TS column orders categories by response rate, the code column lets an exact-match split recover identity where that matters. Unseen categories at `transform` have zero evidence and resolve to exactly the prior.

The implementation is one segmented cumulative sum over a `(category, visit-order)` sort (vectorized, seeded, deterministic); the causality property is pinned by a test that flips one row's label and asserts that row's own encoding cannot move ([`test_encoding.py`](../../python/tests/test_encoding.py)).

## Try it

```python
import numpy as np, bonsai

def load(p):
    d = np.loadtxt(p, delimiter=",", skiprows=1, dtype=np.float32)
    return d[:, 1:], d[:, 0]

Xtr, ytr = load("tests/data/amazon_train.csv")   # 9 columns, all category IDs
Xte, yte = load("tests/data/amazon_test.csv")

enc = bonsai.OrderedTargetEncoder(columns=range(Xtr.shape[1]))
m = bonsai.BonsaiRegressor(n_iters=200, objective="logloss", grower="depthwise")
m.fit(enc.fit_transform(Xtr, ytr), ytr)
```

On the committed `tests/data/amazon_train.csv` / `amazon_test.csv` split the encoder is worth **+0.049 AUC** over ordinal codes (0.811 → 0.860), past LightGBM's *native* categorical machinery on the same data. That split is the one the code above runs and the one `python/tests/test_encoding.py` pins, so it is the number to expect if you copy the snippet.

## The evidence: why an encoder and not an engine feature

Every library's categorical machinery, toggled on/off at matched knobs, on three real categorical datasets (test AUC; protocol recorded in [the tradeoff writeup](../../benchmarks/categorical-tradeoff-2026-07.md), probe script in git history). This is a different protocol from the snippet above: the full OpenML dataset, split 80/20 at random with seed 42, so the amazon row reads +0.028 rather than +0.049 for the same encoder.

| variant | amazon | adult | kick |
|---|--:|--:|--:|
| bonsai, ordinal codes | 0.8307 | 0.9300 | **0.7830** |
| bonsai + `OrderedTargetEncoder` | **0.8590** | **0.9306** | 0.7797 |
| lightgbm, ordinal → native set splits | 0.8278 → 0.8572 | 0.9302 → 0.9303 | 0.7846 → 0.7666 |
| catboost, ordinal → native ordered TS | 0.7791 → 0.8894 | 0.9269 → 0.9283 | 0.7740 → 0.7777 |
| xgboost, ordinal → native | 0.8052 → 0.7878 | 0.9276 → 0.9283 | 0.7827 → 0.7791 |

Native set splits (the feature that would have grown the then 1,400-line split/tree core by a third) measure **+0.029 / +0.000 / −0.018** by LightGBM's own toggle: a coin flip that costs split-scan complexity even when it loses. The encoder beats it where it wins and is a per-dataset *choice* where it doesn't. CatBoost's remaining amazon edge comes from per-tree permutations and crossed-category statistics, which set splits wouldn't have bought either.

## Crossing categories: the last of the gap

CatBoost's remaining amazon edge is not its split machinery; it is *crossed* categorical statistics: the response rate of the (user, resource) **pair** carries signal neither column holds alone, and a depth-6 tree burns levels rediscovering it.

Crosses are also just preprocessing: a pair of codes is one joint category (packed into an int64 key), and the same ordered TS applies:

```python
enc = bonsai.OrderedTargetEncoder(columns=range(9), cross=2)  # + C(9,2) pair columns
```

Measured on the repo's amazon split, with CatBoost's own cross-toggle as the control:

| variant | AUC |
|---|--:|
| bonsai, singles TS + codes | 0.8604 |
| **bonsai, `cross=2`** | **0.8877** |
| bonsai, + triples | 0.8859 (overfits) |
| catboost native, full CTR crosses | 0.8897 |
| catboost native, crosses disabled (`max_ctr_complexity=1`) | 0.8587 |

Two readings: the crosses *are* CatBoost's whole remaining edge (its no-cross line falls below our singles), and pair-TS closes the gap to 0.002, chance-band at this test size. Pairs grow as $\binom{k}{2}$: fine at nine columns, a decision at ninety.

## The Fisher sort: what a native split engine would do

The declined engine feature (decision 58) would split a node on an arbitrary *subset* of categories. Scanning all $2^{K-1}$ subsets looks exponential, but it is not, and the reduction is worth knowing.

A split assigns each category $k$ with sums $(G_k, H_k)$ to the left or right child. The second-order objective for child weights $w_L, w_R$ is separable per category:

```math
\mathcal{L} = \sum_{k \in L} \left( G_k w_L + \tfrac{1}{2} H_k w_L^2 \right) + \sum_{k \in R} \left( G_k w_R + \tfrac{1}{2} H_k w_R^2 \right) + \tfrac{\lambda}{2}(w_L^2 + w_R^2)
```

For any fixed pair $w_L < w_R$, category $k$ prefers the left child exactly when

```math
\frac{G_k}{H_k} \le -\tfrac{1}{2}(w_L + w_R)
```

which is a threshold on $G_k/H_k$. So the optimal partition is *contiguous* in $G_k/H_k$ order (Fisher 1958's grouping result), and the $2^{K-1}$ subsets collapse to $K-1$ prefixes of one sorted order.

With leaf-level L2 the argument is exact only at $\lambda = 0$; LightGBM sorts by $G_k / (H_k + \text{cat\_smooth})$, where the smoothing also guards against low-count categories dominating the order. Near-optimal in practice, exact where the proof applies.

## Inside CatBoost, for comparison: the engine we didn't need

Per tree, CatBoost draws from a pool of random permutations and recomputes its categorical statistics ("CTRs") **online** (every split candidate's statistic reflects the current permutation, quantized against CTR-specific borders), and as a tree grows it **greedily builds combinations**: the categoricals along the current path get crossed with each remaining categorical, up to `max_ctr_complexity` (default effectively pairs-and-beyond), each cross getting its own online TS.

That machinery is why categoricals are stitched through CatBoost's training loop, model format, and predictor: the equivalent of what doc 17 priced for bonsai and decision 58 declined. Every piece of it has a preprocessing counterpart in the two tables above, and each counterpart measures level with it: fresh per-tree permutations against one seeded permutation, online recomputation against one causal pass, greedy path combinations against all pairs encoded once.

The pattern behind the pattern: **a feature earns engine residency only if it must see training state that preprocessing cannot.** Ordered target statistics need labels and a permutation, both available before the first tree, so the engine version buys freshness that measures as noise. Contrast a ranking objective, which needs the loss itself and genuinely cannot be preprocessed; that is what the [feature-admission gate](../../.claude/skills/feature-admission/SKILL.md) asks first.

What CatBoost pays for the engine version and bonsai does not: encoding work on every tree of every fit, CTR tables riding in the model file, and a categorical dimension through every component. What bonsai pays instead: one $O(n \log n)$ encode before training, and the user must remember `fit_transform` vs `transform`: the leakage line the API is shaped around.

## Gotchas & war stories

- **Never encode with plain (greedy) target statistics.** The single-appearance pathology above is not theoretical: it is why CatBoost exists. The stage-1 study ([feature_gap §18](https://github.com/daniel-m-campos/bonsai/blob/main/docs/feature_gap.md)) measured plain K-fold encoding at 0.8462 on amazon vs 0.8590 ordered: causality is the load-bearing part.
- **TS is not free on every dataset.** On kick (18 noisy mixed-cardinality columns) it costs 0.003 AUC, and every library's native machinery loses there too, LightGBM by 0.018. Measure per dataset; the encoder being outside the core is what makes skipping it a one-line decision.
- **Regression targets work unchanged** (the math never assumes $y \in \{0,1\}$); multiclass needs one TS column per class (one-vs-rest) and is not built in yet.
- **The codes themselves stay meaningful.** With `keep_codes` the original columns ride along NaN-intact: bonsai routes missing values natively (chapter 2), so don't impute them away before encoding.
