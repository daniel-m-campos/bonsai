# External standings: the Grinsztajn tabular benchmark (2026-07-14)

The internal quality campaign's ten datasets were chosen by this project, which caps the credibility of "best on 9 of 10" no matter how honestly it was run. This run adopts the benchmark of Grinsztajn, Oyallon, and Varoquaux (2022), "Why do tree-based models still outperform deep learning on tabular data?": four OpenML suites (297/298 numerical regression/classification, 299/304 categorical regression/classification), 55 tasks, selected by someone else.

Protocol (`scripts/run_tabular_suite.py`): the paper's medium setting (train capped at 10k rows, test at 50k), campaign knobs for every library (200 iters, lr 0.05, depth 6, 255 bins, matched regularization), categorical features as ordinal codes for every library, 3 seeds per task, metrics averaged. Deviations from the paper, stated: fixed random splits instead of its resampling protocol, and no per-model tuning (matched knobs is the point). Raw rows: `results/grinsztajn-2026-07.jsonl` (990 fits, zero failures).

Library standings (each library at its best grower/configuration per dataset; xgboost at the campaign mapping `min_child_weight = min_data_in_leaf = 20`):

| library | mean rank | outright wins /55 | vs bonsai head-to-head |
|---|--:|--:|--|
| bonsai | **1.73** | 27 | |
| lightgbm | 2.35 | 10 | bonsai >= on 37/55 |
| xgboost | 2.73 | 9 | bonsai >= on 42/55 |
| catboost | 3.20 | 9 | bonsai >= on 46/55 |

bonsai's rank distribution across the 55 datasets: first 27, second 17, third 10, fourth 1 (second or better on 44 of 55, last exactly once). The largest remaining per-dataset losses to any library are small and named: `year` (xgboost +0.0066 r²) and `yprop_4_1` (+0.0053), the decision-55 cut-quality regime with real datasets attached.

**The min_child_weight bracket (correction, same day).** The first published run hardcoded xgboost's `min_child_weight = 1`, deviating from the campaign mapping and allowing xgboost ~20x smaller leaves than the other libraries; under that setting xgboost ranked 2.11 with 26 wins and bonsai 2.04 with 10. The corrected run above uses the campaign mapping. Neither convention is perfectly fair to xgboost: `min_child_weight` is hessian-weighted, so 20 on classification (hessian <= 0.25) implies ~80+ rows per leaf, harsher than the 20 rows the others get, while 1 is far looser. The two runs bracket xgboost's honest position, and the robust claim is the one that holds at both ends: **bonsai has the best mean rank under either convention** (1.73 campaign-mapped, 2.04 loose). Sensitivity rows: `results/grinsztajn-2026-07-xgb-mcw1.jsonl`.

Caveats, because standings without them are advertising:

1. The 10k-row cap is the regime where xgboost's cut-quality edge (decision 55) matters most; the internal campaign at full dataset sizes and the 16M-row scale ladder tell complementary stories.
2. Ordinal codes strip catboost's native categorical machinery. The convention treats every library identically (and matches the campaign), but catboost's 3.33 undersells what it does with its own encoders on categorical-heavy data (see `categorical-tradeoff-2026-07.md`).
3. depthwise and leafwise nearly coincide at depth 6 with 63 leaves (2^6 = 64), so per-grower win counts are not meaningful at these knobs; the library-level table uses each library's best variant per dataset.

Verdict: adopted as the standings suite. The internal ten-dataset campaign remains the fast smoke tier; this is the citable table.
