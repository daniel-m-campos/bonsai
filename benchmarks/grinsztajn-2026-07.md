# External standings: the Grinsztajn tabular benchmark (2026-07-14)

The internal quality campaign's ten datasets were chosen by this project, which caps the credibility of "best on 9 of 10" no matter how honestly it was run. This run adopts the benchmark of Grinsztajn, Oyallon, and Varoquaux (2022), "Why do tree-based models still outperform deep learning on tabular data?": four OpenML suites (297/298 numerical regression/classification, 299/304 categorical regression/classification), 55 tasks, selected by someone else.

Protocol (`scripts/run_tabular_suite.py`): the paper's medium setting (train capped at 10k rows, test at 50k), campaign knobs for every library (200 iters, lr 0.05, depth 6, 255 bins, matched regularization), categorical features as ordinal codes for every library, 3 seeds per task, metrics averaged. Deviations from the paper, stated: fixed random splits instead of its resampling protocol, and no per-model tuning (matched knobs is the point). Raw rows: `results/grinsztajn-2026-07.jsonl` (990 fits, zero failures).

Library standings (each library at its best grower/configuration per dataset):

| library | mean rank | outright wins /55 | vs bonsai head-to-head |
|---|--:|--:|--|
| bonsai | **2.04** | 10 | |
| xgboost | 2.11 | 26 | bonsai >= on 25/55 |
| lightgbm | 2.53 | 10 | bonsai >= on 37/55 |
| catboost | 3.33 | 9 | bonsai >= on 46/55 |

bonsai's rank distribution across the 55 datasets: first 10, second 34, third 10, fourth 1. xgboost is the peak library (most outright wins); bonsai is the consistency library (second or better on 44 of 55, last exactly once). Per track: bonsai leads categorical regression (mean rank 1.77), and is second behind xgboost on both classification tracks (xgboost 1.86/1.93 vs bonsai 2.00/2.27) and numerical regression (2.00 vs 2.05).

Caveats, because standings without them are advertising:

1. The 10k-row cap is the regime where xgboost's cut-quality edge (decision 55) matters most; the internal campaign at full dataset sizes and the 16M-row scale ladder tell complementary stories.
2. Ordinal codes strip catboost's native categorical machinery. The convention treats every library identically (and matches the campaign), but catboost's 3.33 undersells what it does with its own encoders on categorical-heavy data (see `categorical-tradeoff-2026-07.md`).
3. depthwise and leafwise nearly coincide at depth 6 with 63 leaves (2^6 = 64), so per-grower win counts are not meaningful at these knobs; the library-level table uses each library's best variant per dataset.

Verdict: adopted as the standings suite. The internal ten-dataset campaign remains the fast smoke tier; this is the citable table.
