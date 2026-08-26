---
name: perf-round
description: >
  bonsai's optimization-round discipline (the compute-DAG method).
  Use when starting any performance work: it
  sequences decompose -> price -> implement -> validate -> record, and
  exists because every shortcut in it has already failed once.
---

Why this exists: profilers lie in exactly the places that matter. Case E1 is the round instrumentation cancelled, where the largest line in the GPU grow profile turned out to be the previous level's histograms draining at the profiler's own sync, and a multi-hour kernel rewrite was called off before any kernel code was written (decision 62). Run the other way, the same ledger predicted the one-line prefetch that closed the largest CPU loss on any chart (decision 61). Isolated measurements mislead in the same shape: a microbenchmark inherits the bottleneck structure of the isolation, not of the system, which is why an accumulator split that looked like a win on cache-resident arrays was flat in the real streaming loop (decision 49). Price a change in the real loop or not at all.

Rationale and case studies as a story: `docs/guide/11-performance-engineering.md`. `scripts/dag_model.py` is the living evaluator. The sequence:

## 1. Decompose before designing

A profiler line you haven't split into wait/work/transfer is not evidence (decisions 52 and PR #35 were both designed against aggregates and refuted). Run the profiled ledger cell (/pod-validate) and check conservation: parts must sum to the whole; an unattributed gap IS the finding, so add lap counters for it FIRST, as their own commit (the `setup=7.13s` memset discovery took one pod run once the buckets existed).

## 2. Price before betting

State the expected saving from measured constants before writing code: deleted host node cost + deleted edge bytes/bandwidth, minus added edge cost (`scripts/dag_model.py`; update its constants from your run). Under ~1s of a ~30s fit: don't spend a pod on it. Constants drift ~25% across hosts, so only play moves that win across the plausible range.

## 3. Implement with gates

/quality-gates per commit. Multi-commit change sets on one branch/PR (the PR #34 pattern); instrumentation ships in the same PR as the optimization it prices.

## 4. Validate same-pod

Before/after on ONE pod (/pod-validate), r² exactly equal for behavior-preserving changes, ledger delta against the priced expectation. If measurement refutes the move: close the PR unmerged and write the refutation into docs/decisions.md with the same care as a win, because refutations are deliverables (decisions 52, 55; PR #35).

## 5. Record

Decision entry (numbered, with the measured delta and the rejected alternatives), PR body carries the ledger table, README performance section only changes with re-baseline data behind it.
