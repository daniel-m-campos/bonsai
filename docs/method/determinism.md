# Determinism as a contract

The rule: model bytes are identical across runs, across thread counts (outside one documented relaxation), and across CPU architectures, and CI enforces it per commit by training a reference model on an arm64 Mac and an x86-64 Linux box and comparing file hashes.

This is not an aesthetic. A bit-exact contract turns "did this change alter behavior?" from a judgment call into a file comparison, and that one property pays for itself everywhere.

## What the contract caught

**Compiler contraction across architectures.** The cross-arch gate's first real catch: the same model trained to different bytes on arm64 and x86-64 (test pins 0.71719 vs 0.71725). The mappers were bit-identical; the divergence was clang fusing multiply-add into FMA on one architecture only. Fix: `-ffp-contract=off` on the host build, measured at zero cost, and the contract upgraded to a claim no reference library makes ([decision 59](../decisions.md)).

**A silent build-system fallback.** Models differed between two supposedly identical builds; the cause was a missing OpenMP quietly downgrading to serial, which changes reduction order. A missing OpenMP is now a hard configure error ([decision 60](../decisions.md)).

## What the contract enables

**Refactoring with proof.** A recent cleanup deduplicated the numerically sensitive softmax code into one helper. Acceptance was not review confidence; it was `np.array_equal` against outputs captured before the change. The refactor shipped with a bit-exactness proof instead of an argument.

**Features that provably cost nothing when off.** New per-row weighting multiplies gradients by 1.0 when unweighted, which is exact in IEEE arithmetic, so every existing model stays byte-identical and the hash gate verifies it. The pattern generalizes: design the off-state of a feature to be arithmetically inert, then let the gate prove it.

**AI-assisted development without trust.** Large parts of bonsai are written by Claude ([how bonsai is built](../about.md)). The reason that works is precisely this contract: an implementation campaign cannot silently change numerical behavior, because the gate would catch the changed bytes. Verification substitutes for trust at exactly the point where trust is weakest.

## Transferring it

Any numerical system can adopt the ladder: fix the reduction orders (deterministic parallelism is a design choice, not a performance sacrifice; bonsai ties xgboost with it), pin the floating-point environment (contraction, denormals), then hash a reference output in CI on more than one architecture. The gate is a few lines of workflow; the property it enforces is the difference between refactoring a numerical codebase and gambling with it.
