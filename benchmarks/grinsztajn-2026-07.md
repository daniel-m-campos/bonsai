# External standings: the Grinsztajn tabular benchmark (2026-07)

> **Supersedes** the 2026-07-14 run in place (decision 69): the bonsai rows were re-run on 2026-07-15 after decision 74's missing-bin closer (reference rows unchanged, same protocol, same seeds; runner identity was proven by replay during the decision-69 refactor). Value-level, 43 of 55 datasets moved less than 0.001; 8 improved and 4 regressed beyond it. Rank-level, the near-ties broke systematically in bonsai's favor. The prior standings (mean rank 1.73, 27 wins) remain in git history.

The internal quality campaign's ten datasets were chosen by this project, which caps the credibility of "best on 9 of 10" no matter how honestly it was run. This run adopts the benchmark of Grinsztajn, Oyallon, and Varoquaux (2022), "Why do tree-based models still outperform deep learning on tabular data?": four OpenML suites (297/298 numerical regression/classification, 299/304 categorical regression/classification), 55 tasks, selected by someone else.

Protocol (`scripts/run_tabular_suite.py`): the paper's medium setting (train capped at 10k rows, test at 50k), campaign knobs for every library (200 iters, lr 0.05, depth 6, 255 bins, matched regularization), categorical features as ordinal codes for every library, 3 seeds per task, metrics averaged. Deviations from the paper, stated: fixed random splits instead of its resampling protocol, and no per-model tuning (matched knobs is the point). Raw rows: `results/grinsztajn-2026-07.jsonl` (990 fits, zero failures).

Library standings (each library at its best grower/configuration per dataset; xgboost at the campaign mapping `min_child_weight = min_data_in_leaf = 20`):

| library | mean rank | outright wins /55 | vs bonsai head-to-head |
|---|--:|--:|--|
| bonsai | **1.44** | 36 | |
| lightgbm | 2.51 | 5 | bonsai >= on 46/55 |
| xgboost | 2.84 | 6 | bonsai >= on 48/55 |
| catboost | 3.22 | 8 | bonsai >= on 47/55 |

bonsai is second or better on 50 of 55 datasets and last on none. The decision-55 residual narrowed with the closer: `year` recovered +0.0036 of its +0.0066 gap to xgboost and `yprop_4_1` +0.0006 of +0.0053. The honest reading of the 1.73 to 1.44 jump: the per-dataset value moves are mostly small (43 of 55 inside 0.001), but this suite had many photo-finish second places, and decision 74's closer nudged enough of them across. Rank tables amplify near-ties; the durable claims are the head-to-head counts and never-last, which moved the same direction.

**The min_child_weight bracket (correction, same day).** The first published run hardcoded xgboost's `min_child_weight = 1`, deviating from the campaign mapping and allowing xgboost ~20x smaller leaves than the other libraries; under that setting xgboost ranked 2.11 with 26 wins and bonsai 2.04 with 10. The corrected run above uses the campaign mapping. Neither convention is perfectly fair to xgboost: `min_child_weight` is hessian-weighted, so 20 on classification (hessian <= 0.25) implies ~80+ rows per leaf, harsher than the 20 rows the others get, while 1 is far looser. The two runs bracket xgboost's honest position, and the robust claim is the one that holds at both ends: **bonsai has the best mean rank under either convention** (1.44 campaign-mapped post-closer, 2.04 loose pre-closer). Sensitivity rows: `results/grinsztajn-2026-07-xgb-mcw1.jsonl`.

Caveats, because standings without them are advertising:

1. The 10k-row cap is the regime where xgboost's cut-quality edge (decision 55) matters most; the internal campaign at full dataset sizes and the 16M-row scale ladder tell complementary stories.
2. Ordinal codes strip catboost's native categorical machinery. The convention treats every library identically (and matches the campaign), but catboost's 3.33 undersells what it does with its own encoders on categorical-heavy data (see `categorical-tradeoff-2026-07.md`).
3. depthwise and leafwise nearly coincide at depth 6 with 63 leaves (2^6 = 64), so per-grower win counts are not meaningful at these knobs; the library-level table uses each library's best variant per dataset.

Verdict: adopted as the standings suite. The internal ten-dataset campaign remains the fast smoke tier; this is the citable table.
