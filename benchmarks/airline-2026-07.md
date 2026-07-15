# The airline speed rung: benchm-ml at 0.1M / 1M / 10M (2026-07)

The real-data speed benchmark (Szilard Pafka's benchm-ml airline on-time ladder, issue #154): mixed categorical/numeric columns, real class balance, the industry-standard GBM comparison shape. Complements the synthetic scaling suite (smooth features, no encoder story) and the Grinsztajn standings (10k-row cap).

Protocol: `python -m bonsai.bench.airline`, one RunPod L40S (SECURE US-NC-1, driver 570.124.06), all 78 rows same-pod, `fit()` timed end to end including each library's ingest, AUC on the upstream 100k test split. Two knob shapes per decision on issue #154: the campaign shape (depth 8, 100 iters, lr 0.1, 255 bins) as the headline and Pafka's depth 10 for cross-table comparability with the GBM-perf ecosystem. Categoricals are ordinal codes for every library (decision 68 convention); the `bonsai_ts_*` rows are the labeled exception (OrderedTargetEncoder pipeline, decision 58; `fit_s` includes the encode, recorded separately as `encode_s`). Raw rows: `results/airline-2026-07.jsonl`.

## Campaign knobs (depth 8)

| variant | 0.1m fit / AUC | 1m fit / AUC | 10m fit / AUC |
|---|--:|--:|--:|
| bonsai_depthwise | 2.6s / .7260 | 7.0s / .7447 | 35.3s / .7468 |
| bonsai_oblivious | 1.6s / .7218 | 5.2s / .7256 | 40.0s / .7264 |
| bonsai_cuda_depthwise | 0.3s / .7268 | **0.8s / .7447** | 5.5s / .7467 |
| bonsai_cuda_oblivious | 0.9s / .7218 | 1.4s / .7253 | 6.1s / .7264 |
| bonsai_ts_depthwise | 2.8s / .7238 | 9.5s / **.7462** | 72.6s / **.7498** |
| bonsai_ts_cuda_depthwise | 0.5s / .7239 | 3.1s / .7460 | 37.8s / .7498 |
| xgb_hist | 0.9s / .7264 | 3.6s / .7429 | 29.9s / .7469 |
| xgb_cuda | 0.2s / .7257 | 0.6s / .7438 | **3.2s** / .7476 |
| lgbm_cpu | 1.7s / .7264 | 4.2s / .7448 | 19.7s / .7459 |
| catboost_cpu | 1.0s / .7167 | 5.7s / .7234 | 37.5s / .7245 |
| catboost_gpu | 0.5s / .7167 | 0.9s / .7229 | 5.4s / .7254 |

(ts_oblivious rows omitted from the table for brevity; in the jsonl. TS lifts oblivious +0.007 at 1M and +0.0075 at 10M but oblivious still trails depthwise on this data.)

## Pafka protocol (depth 10)

| variant | 0.1m fit / AUC | 1m fit / AUC | 10m fit / AUC |
|---|--:|--:|--:|
| bonsai_depthwise | 3.1s / .7242 | 12.2s / **.7547** | 47.8s / .7596 |
| bonsai_cuda_depthwise | 0.3s / .7244 | 1.0s / .7535 | 6.2s / .7594 |
| bonsai_ts_depthwise | 6.2s / .7206 | 15.2s / .7529 | 83.8s / **.7615** |
| bonsai_ts_cuda_depthwise | 0.6s / .7206 | 3.2s / .7537 | 38.2s / .7610 |
| xgb_hist | 1.0s / .7260 | 2.6s / .7502 | 36.9s / .7580 |
| xgb_cuda | 0.3s / .7253 | 0.8s / .7489 | **3.8s** / .7579 |
| lgbm_cpu | 2.4s / .7248 | 8.0s / .7516 | 29.2s / .7600 |
| catboost_cpu | 2.0s / .7177 | 6.3s / .7284 | 49.4s / .7295 |
| catboost_gpu | 0.7s / .7163 | 1.1s / .7287 | 6.3s / .7304 |

## Reading it

- **A bonsai variant has the best AUC in every cell from 1M up, under both protocols.** Campaign knobs: `bonsai_ts_depthwise` (.7462 / .7498). Pafka depth 10: plain `bonsai_depthwise` at 1M (.7547, ahead of lightgbm .7516 and xgboost .7502 with no encoder at all) and `bonsai_ts_depthwise` at 10M (.7615, vs lightgbm .7600, bonsai plain .7596, xgboost .7580).
- **The encoder's value grows with data and with what the trees cannot do**: TS lifts depthwise +0.0015/+0.0030 at 1M/10M under depth 8, but at depth 10 plain trees can carve the code spaces themselves at 1M and TS only wins again at 10M. For oblivious it is structural rescue at every size (+0.007-0.008), exactly the diagnosis from the 0.1m smoke: symmetric trees on meaningless ordinal codes need monotone features. Depthwise is the right bonsai grower on categorical data.
- **xgboost-GPU owns raw speed on this narrow shape** (3.2-3.8s at 10M; 8 columns is its best case), with `bonsai_cuda_depthwise` and catboost-GPU tied for second (5.4-6.3s). On CPU, lightgbm leads at 10M (19.7s / 29.2s), with bonsai trailing on this column count as the scaling suite predicts (narrow data is the documented weak spot; bonsai wins the wide end there).
- **catboost is last on AUC in every cell** (-0.02 to -0.03): the uniform ordinal convention strips its native categorical machinery, and its oblivious structure pays the same code-space tax bonsai's oblivious does. Its own encoders would tell a different story (`categorical-tradeoff-2026-07.md` measures that); this table measures engines under one convention.
- **The TS cost is the encode, not the trees**: at 10M, `encode_s` is ~32s of the 72.6s/83.8s CPU pipeline (and dominates the 38s GPU pipeline). Anyone reproducing the quality crown at speed should encode once and reuse, which `bonsai.Dataset` supports.

## Verdict

On the standard real-data ladder, bonsai is the quality leader at scale under both protocols (with the shipped encoder at campaign knobs; outright at Pafka's depth 10 at 1M), GPU-competitive with catboost, and behind xgboost-GPU on raw wall clock for this narrow column count. The rows were produced on post-decision-74 code; the closer is part of this story (the airline columns include exactly the capped/heavy shapes it fixed).
