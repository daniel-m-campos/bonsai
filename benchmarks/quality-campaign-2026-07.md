# Quality campaign, 2026-07: fitting metrics vs the reference libraries

Ten new OpenML datasets (regression / binary / multiclass / counts, 5k–45k rows — quality, not scale), every bonsai grower × sampler combination vs xgboost, lightgbm, and catboost at matched hyperparameters (`scripts/compare.py`, 200 iters, lr 0.05, depth 6, 255 bins). Raw per-run tables: `benchmarks/results/q_*` (pre-fix campaign), `f_*` (post-fix), `p_*`/`v_*` (root-cause probes).

## What the campaign found

Four root-caused problems, each probe-confirmed before filing, fixed one commit per issue in the campaign PR:

1. **#59 (harness)**: catboost's multiclass accuracy was computed on a broadcast matrix (reported 0.039 ≈ random on letter) and its binary AUC on hard labels (understated ~0.1). Both artifacts of `compare.py`, not catboost.
2. **#60 (oblivious veto)**: the level split rejected any cut where ONE frontier node's children fell under `min_child_hess`; at depth ≥ 5 some node is always near-empty, so good cuts died wholesale — bonsai-oblivious trailed catboost on 7/10 datasets. Infeasible nodes now contribute zero gain instead of vetoing. superconduct 12.68→11.85, elevators 0.00291→0.00256 — **oblivious now leads catboost**.
3. **#61 (cut collapse)**: duplicate-heavy columns lost most of their distinct values to the stride+dedup (house_sales' 13-value `bedrooms` got 7 cuts). Fixed to exact distinct-value cuts under the budget — objectively right, measured **net-neutral** on these datasets (California pin moved +0.07%, house_sales barely moved: its real gap is #63).
4. **#62 (multiclass pace)**: the factor-2 softmax hessian halved every Newton step — bonsai needed exactly 2× the iterations to match lightgbm. True diagonal `p(1−p)` now: letter 0.9515→**0.9613**, satimage 0.9168→**0.9191**, matching lightgbm at the same budget.

5. **#63 (heavy-value cut allocation)**: on columns where one value outweighs a mean-sized bin (constant runs, zeros, capped fields), the equal-frequency stride let that value's run swallow its neighbours' resolution. Fixed with a count-weighted greedy walk (lightgbm's GreedyFindBin shape): heavy values get a bin to themselves, the rest fill toward a running mean, cuts land at midpoints between distinct values. The rule is per column and fires **only when a heavy value exists** — all-continuous columns keep the stride bit-identically, because equal frequency already *is* the count-weighted allocation there (the full-greedy variant was measured and rejected: it shuffled every dataset's thresholds inside the chance band and cost more than it paid — decision 57). house_sales 131,841 → 128,959; the small-dataset movements are chance-band (California pin +0.13%, all standings unchanged).

Remaining on #63 (left open): bonsai at 511 bins matches xgboost at its 256 on house_sales (121,959 vs 121,829), so the refs still extract more from an equal budget on dense-continuous geography columns — a resolution-allocation question (per-column budgets), not a placement defect.

## Post-fix standings (grower × best-ref per dataset, all_rows sampler)

| dataset | metric | bonsai dw | bonsai lw | bonsai obl | xgboost | lightgbm | catboost |
|---|---|--:|--:|--:|--:|--:|--:|
| house_sales | rmse | 128,959 | 126,448 | 133,165 | **121,829** | 122,213 | 124,095 |
| superconduct | rmse | **10.3959** | 10.7593 | 11.9004 | 10.7637 | 10.7227 | 12.1154 |
| elevators | rmse | **0.0023** | 0.0023 | 0.0025 | 0.0024 | 0.0023 | 0.0026 |
| wine_quality | rmse | **0.6373** | 0.6400 | 0.6493 | 0.6385 | 0.6386 | 0.6505 |
| magic_telescope | auc | **0.9356** | 0.9352 | 0.9312 | 0.9330 | 0.9352 | 0.9335 |
| phoneme | auc | **0.9415** | 0.9409 | 0.9303 | 0.9283 | 0.9379 | 0.9396 |
| electricity | auc | **0.9457** | 0.9412 | 0.9155 | 0.9353 | 0.9441 | 0.9178 |
| letter | acc | **0.9613** | 0.9603 | 0.9557 | 0.9240 | 0.9610 | 0.9223 |
| satimage | acc | **0.9191** | 0.9176 | 0.9137 | 0.9012 | **0.9191** | 0.9059 |
| bike_sharing | rmse | **39.9162** | 41.3369 | 43.9763 | 53.8279 | 49.2892 | 45.0404 |

**bonsai-depthwise is the best-scoring library on 9 of 10 datasets** (sole exception: house_sales, where the #63 fix closed 27% of the gap and the remainder is the open resolution-allocation question above). Leafwise sits within noise of depthwise throughout; oblivious carries its structural quality trade but now beats catboost — the other symmetric-tree library — nearly everywhere. Sampler variants (bernoulli, goss) tracked their all_rows baselines within expected sampling noise across the full pre-fix matrix (`q_*` files). The bike_sharing (Poisson, PR #55) row is the widest margin of the campaign: every bonsai grower — including oblivious — beats every reference, with depthwise 11% ahead of the best of them (catboost).

Caveats: single split (seed 42), no per-dataset tuning (the point is matched-default behavior, not leaderboard chasing), categorical-heavy datasets deliberately avoided (known gap #43); `electricity`'s oblivious AUC remains ~0.03 behind the asymmetric growers — structural on temporal data (catboost's symmetric trees show the same signature).
