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

## Provisional verdict and the gate

**Provisionally defer LambdaRank.** The pairwise objective does not clear the bar; if ranking is ever built, catboost's result points at *listwise* (YetiRank-style) as the better target, not pairwise LambdaRank.

This rests on synthetic data, and the admission rule requires standings to move *across multiple datasets*. So the decision is not final: the **pre-registered gate** before a firm decline is one real LETOR/MSLR run (MQ2008 or MSLR-WEB10K Fold1) through the same NDCG@10 harness. If real graded-relevance data shows lambdarank clearing regression by a margin that matters, this reopens; if it echoes the synthetic wash, decision-log it as declined-by-measurement (the [categorical precedent](../docs/decisions.md), decision 58). A LETOR mirror was not reachable when this probe ran; the run is queued, not skipped.
