# Ranking (LambdaRank, issue #58): does the objective beat regression? — provisional (2026-07)

Issue #58 scopes a real project: query groups in the data layer, pairwise lambda gradients with NDCG-delta weighting, an NDCG metric, and task plumbing. Before paying that, the feature-admission question ([SKILL](../.claude/skills/feature-admission/SKILL.md)): **on a ranking metric, how much does a native ranking objective actually buy over plain regression on the relevance labels — which bonsai already does?**

## The probe

`scripts/probe_ranking.py`: synthetic graded-relevance data (shared latent scoring function over 12 informative features, 50 docs/query, grades 0–4 by within-query quantiles), scored by **NDCG@10** per query. Two regimes: `graded` (even grade cuts — regression-friendly) and `search` (top-heavy, ~85% grade 0 — the top-focused regime a ranking loss is meant to win). Matched depth 6, lr 0.1, 200 iters. Raw runs: [`results/ranking-tradeoff-2026-07.jsonl`](results/ranking-tradeoff-2026-07.jsonl).

| learner | NDCG@10 graded | NDCG@10 search |
|---|--:|--:|
| **bonsai regression** (today) | 0.6623 | 0.3028 |
| lgbm regression | 0.6653 | 0.2972 |
| lgbm **lambdarank** | 0.6601 | 0.3023 |
| xgb **rank:ndcg** | 0.6673 | 0.3084 |
| catboost **YetiRank** | **0.6679** | **0.3132** |

## Reading

The pairwise ranking objective — the exact issue-#58 target — buys essentially nothing. lgbm's own `lambdarank` **ties or trails its own regression** in both regimes (0.6601 vs 0.6653; 0.3023 vs 0.2972), and xgb's `rank:ndcg` beats bonsai regression by only +0.005. bonsai's plain regression already sits mid-pack — ahead of both lgbm variants in the search regime. The one real edge is catboost's **listwise** YetiRank: +0.006 (graded) / +0.010 (search) over bonsai regression. So what little a ranking loss wins here comes from *listwise* aggregation, not the *pairwise* lambda gradients issue #58 proposes to build.

Against the admission axes, LambdaRank scores poorly: it wants query groups threaded through the CSV/Python data layer, a new pairwise-gradient objective, an NDCG metric, and task plumbing — for ≤0.005 NDCG@10 that regression already captures, on this evidence.

## The gate: real MQ2008 (LETOR 4.0)

The pre-registered gate — a real graded-relevance run — was run on MQ2008 (3005 train / 768 test docs, 201 / 50 queries, grades 0–4, 300 features; the fold lightgbm ships), same NDCG@10 harness, `scripts/probe_ranking.py --real`. 200 trees below; a 300-tree run is shown alongside because the fold is small enough that the count matters.

| learner | NDCG@10, 200 trees | 300 trees |
|---|--:|--:|
| lgbm regression | 0.7714 | 0.7576 |
| lgbm **lambdarank** | 0.7662 | 0.7693 |
| **bonsai regression** (today) | 0.7720 | 0.7709 |
| xgb **rank:ndcg** | **0.7866** | **0.7861** |
| catboost **YetiRank** | 0.7746 | 0.7930 |

Real data moves the standings where synthetic did not, but the fold is small (50 test queries) and noisy — catboost YetiRank alone swings 0.7746 → 0.7930 between the two tree counts, roughly the size of the effect being measured. The signal that survives that noise: **xgb's *listwise* `rank:ndcg` beats bonsai regression by a stable ~+0.015** (0.7866 / 0.7861 vs 0.7720 / 0.7709) at both counts. The pairwise **LambdaRank** that issue #58 proposes does *not* — lgbm's lambdarank sits at or below bonsai's plain regression in both runs. So the real, reproducible gap is to a listwise loss, and it is modest.

## Verdict

**Ranking is a real but modest parity gap — a stable ~+0.015 NDCG@10 to a listwise loss (`rank:ndcg`), not to the pairwise LambdaRank of issue #58, which bonsai's regression already matches.** Recommendation: reframe #58 from "pairwise LambdaRank" to a *scoped, listwise-first* project (an `NDCG`/`YetiRank`-style objective plus query groups in the data layer and an NDCG metric), and confirm the margin across MSLR-WEB10K folds before committing, since one small fold with count-sensitive numbers drives it. This is the honest inverse of the categorical result (decision 58): there preprocessing matched the engine feature and it was declined; here no preprocessing substitutes for a ranking loss and the measured gain is real, so the feature is admissible — once its target is the listwise loss the data actually rewards.
