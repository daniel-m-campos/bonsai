# PROVENANCE NOTE (decision 69): this probe is a completed experiment whose
# committed evidence was produced by this exact code; it stays as-run rather
# than being refactored onto bonsai.bench. NEW probes must import their knobs
# and reference-library mappings from bonsai.bench.params and their metrics
# from bonsai.bench.metrics.
#!/usr/bin/env python3
"""Feature-admission probe for issue #58 (LambdaRank).

Before building query groups + pairwise lambda gradients + an NDCG metric into
the core, measure the BENEFIT: on a graded-relevance ranking task, how much
NDCG@10 does a native ranking objective (lgbm lambdarank, xgb rank:ndcg,
catboost YetiRank) buy over plain regression on the relevance labels — which
bonsai can already do today? If the gap is small, the objective is core
complexity for a metric that regression already captures.

Synthetic and reproducible (no LETOR download): a shared latent scoring
function over informative features, graded 0-4 by within-query quantiles, so
the model must order docs within groups. Two regimes:
  --regime graded : even grade cuts (a smooth, regression-friendly task)
  --regime search : top-heavy cuts (~85% grade 0), the top-focused regime a
                    ranking loss is supposed to win.

    python scripts/probe_ranking.py --regime search --out results/rank.jsonl

For the real gate, point --real at a dir holding lightgbm's MQ2008 fold
(rank.train, rank.train.query, rank.test, rank.test.query from
LightGBM/examples/lambdarank):

    python scripts/probe_ranking.py --real /path/to/mq2008 --out results/rank_real.jsonl

Verdict and numbers in benchmarks/ranking-tradeoff-2026-07.md.
"""
import argparse
import json
import sys

import numpy as np
from sklearn.metrics import ndcg_score

sys.path.insert(0, "scripts")
sys.path.insert(0, "build/python")

N_FEAT, N_INFORMATIVE, DOCS_PER_Q = 30, 12, 50
ITERS, DEPTH, LR, NOISE = 200, 6, 0.1, 2.0
CUTS = {"graded": [0.2, 0.4, 0.6, 0.8], "search": [0.85, 0.93, 0.97, 0.99]}


def make(rng, w, n_queries, edges_q):
    X, y, groups = [], [], []
    for _ in range(n_queries):
        f = rng.random((DOCS_PER_Q, N_FEAT)).astype(np.float32)
        score = f[:, :N_INFORMATIVE] @ w + rng.normal(0, NOISE, DOCS_PER_Q)
        grades = np.digitize(score, np.quantile(score, edges_q)).astype(np.float32)
        X.append(f)
        y.append(grades)
        groups.append(DOCS_PER_Q)
    return np.vstack(X), np.concatenate(y), np.array(groups)


def ndcg_at_10(y_true, y_score, groups):
    out, off = [], 0
    for g in groups:
        yt, ys = y_true[off:off + g], y_score[off:off + g]
        off += g
        if g >= 2 and len(set(yt)) > 1:  # NDCG is undefined for a constant group
            out.append(ndcg_score(yt[None, :], ys[None, :], k=10))
    return float(np.mean(out))


def learners(Xtr, ytr, gtr, Xte):
    def bonsai_reg():
        import bonsai
        pairs = [("dispatch.grower_name", "depthwise"), ("dispatch.objective_name", "mse"),
                 ("booster.n_iters", str(ITERS)), ("booster.learning_rate", str(LR)),
                 ("booster.random_seed", "42"), ("tree.max_depth", str(DEPTH)),
                 ("tree.max_leaves", str(1 << DEPTH)), ("tree.min_data_in_leaf", "20"),
                 ("tree.lambda_l2", "1.0"), ("bin_mapper.max_bin", "255")]
        return np.asarray(bonsai.train(pairs, Xtr, ytr).predict(Xte))

    def lgbm_reg():
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=ITERS, max_depth=DEPTH, num_leaves=1 << DEPTH,
                              learning_rate=LR, min_child_samples=20, reg_lambda=1.0, verbose=-1)
        return m.fit(Xtr, ytr).predict(Xte)

    def lgbm_rank():
        import lightgbm as lgb
        m = lgb.LGBMRanker(objective="lambdarank", n_estimators=ITERS, max_depth=DEPTH,
                           num_leaves=1 << DEPTH, learning_rate=LR, min_child_samples=20,
                           reg_lambda=1.0, label_gain=list(range(32)), verbose=-1)
        return m.fit(Xtr, ytr, group=gtr).predict(Xte)

    def xgb_rank():
        import xgboost as xgb
        dtr = xgb.DMatrix(Xtr, label=ytr)
        dtr.set_group(gtr)
        params = {"objective": "rank:ndcg", "eta": LR, "max_depth": DEPTH,
                  "min_child_weight": 20, "lambda": 1.0}
        return xgb.train(params, dtr, num_boost_round=ITERS).inplace_predict(Xte)

    def catboost_rank():
        from catboost import CatBoostRanker, Pool
        qid = np.repeat(np.arange(len(gtr)), gtr)
        m = CatBoostRanker(loss_function="YetiRank", iterations=ITERS, depth=DEPTH,
                           learning_rate=LR, random_seed=42, verbose=False)
        m.fit(Pool(Xtr, label=ytr, group_id=qid))
        qte = np.repeat(np.arange(len(gte_global)), gte_global)
        return m.predict(Pool(Xte, group_id=qte))

    return {"bonsai_regression": bonsai_reg, "lgbm_regression": lgbm_reg,
            "lgbm_lambdarank": lgbm_rank, "xgb_rank_ndcg": xgb_rank,
            "catboost_yetirank": catboost_rank}


gte_global = None


def load_letor(directory):
    # lightgbm's MQ2008 fold: svmlight rows + a .query file of group sizes.
    from sklearn.datasets import load_svmlight_file

    def one(split):
        x, y = load_svmlight_file(f"{directory}/rank.{split}")
        g = np.loadtxt(f"{directory}/rank.{split}.query", dtype=int)
        return x.toarray().astype(np.float32), y.astype(np.float32), g

    return one("train"), one("test")


def main() -> int:
    global gte_global
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=list(CUTS), default="search")
    ap.add_argument("--real", default=None,
                    help="dir with lightgbm-format rank.{train,test}{,.query} "
                         "(MQ2008); overrides --regime with real graded relevance")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.real:
        (Xtr, ytr, gtr), (Xte, yte, gte) = load_letor(args.real)
        tag = "mq2008"
    else:
        rng = np.random.default_rng(0)
        w = rng.normal(size=N_INFORMATIVE)
        edges = CUTS[args.regime]
        Xtr, ytr, gtr = make(rng, w, 3000, edges)
        Xte, yte, gte = make(rng, w, 1000, edges)
        tag = args.regime
    gte_global = gte
    print(f"data={tag} train {Xtr.shape} test {Xte.shape}", file=sys.stderr, flush=True)

    rows = []
    for name, fn in learners(Xtr, ytr, gtr, Xte).items():
        try:
            nd = ndcg_at_10(yte, fn(), gte)
            rows.append({"data": tag, "learner": name, "ndcg_at_10": round(nd, 4)})
        except Exception as exc:
            rows.append({"data": tag, "learner": name, "error": str(exc)})
        print("ROW " + json.dumps(rows[-1]), flush=True)
    if args.out:
        with open(args.out, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
