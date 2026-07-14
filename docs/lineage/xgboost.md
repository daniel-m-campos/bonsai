# XGBoost

XGBoost turned gradient boosting from a technique into an industrial tool, and two of its contributions are load-bearing in every library that followed.

The first is the regularized second-order objective: score a split by $G^2/(H+\lambda)$ over the gradient and hessian sums, derive leaf values from the same quantities, and boosting becomes Newton optimization with regularization built into the gain itself.

The second is making approximate, histogram-based split finding respectable at scale: the insight that a few hundred quantile bins per feature lose almost nothing and change the complexity class of training.

bonsai is built on both. The parts of xgboost it declined are conventions, not ideas.

## Adopted: the foundation

The gain formula is bonsai's split scoring, in every grower, on every backend. The CPU and CUDA implementations share the same gain math ([guide chapter 3](../guide/3-finding-splits.md) derives it from the loss expansion down to the implementing lines).

The `depthwise` grower is xgboost's level-wise tree shape, and histogram binning ([guide chapter 2](../guide/2-binning-and-histograms.md)) is the front door of the whole training path.

## Declined: the factor-2 multiclass hessian

xgboost's multiclass implementation uses $2 p (1-p)$ as the softmax hessian; the true diagonal hessian is $p(1-p)$.

The factor of two halves every Newton step, which costs iterations rather than correctness. Measured on the letter dataset, the factor-2 variant needed roughly twice the budget to reach the same accuracy, and at matched budgets it scored worse (0.9515 vs 0.9613; the campaign fix is recorded in the [decisions log](https://github.com/daniel-m-campos/bonsai/blob/main/docs/decisions.md)).

bonsai uses the true hessian. Compatibility with a convention was not worth a 2× iteration budget.

## The score today

On CPU at 16M rows, bonsai and xgboost-hist are a dead tie: 75.8s vs 75.7s, same pod, same session. The tie was earned one measured change at a time; the final step was a software prefetch in the histogram fill loop, priced from an instruction-level cost ledger before the code was written ([decision 61](https://github.com/daniel-m-campos/bonsai/blob/main/docs/decisions.md), [guide chapter 11](../guide/11-performance-engineering.md)).

On GPU at 16M rows bonsai is ahead: `cuda_oblivious` 18.4s and `cuda_depthwise` 20.5s against xgboost-GPU's 19.9s, at ~3× less host memory (7.0 vs 22.2 GB) and ~3× faster predict.

xgboost keeps two things: the best test accuracy at 16M (0.880 r², against bonsai depthwise's 0.879), and a persistent +0.001 r² edge in cut quality that survived a full sweep of bonsai's binning levers ([decision 55](https://github.com/daniel-m-campos/bonsai/blob/main/docs/decisions.md), a study kept open rather than argued away).
