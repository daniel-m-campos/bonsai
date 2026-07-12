---
name: perf-round
description: >
  bonsai's optimization-round discipline (the compute-DAG method,
  architecture doc 16). Use when starting any performance work: it
  sequences decompose -> price -> implement -> validate -> record, and
  exists because every shortcut in it has already failed once.
---

The method's rationale and case studies: `docs/architecture/16-compute-dag.md` and `docs/guide/11-performance-engineering.md`. The sequence:

## 1. Decompose before designing

A profiler line you haven't split into wait/work/transfer is not evidence (decisions 52 and PR #35 were both designed against aggregates and refuted). Run the profiled ledger cell (/pod-validate) and check conservation: parts must sum to the whole; an unattributed gap IS the finding — add lap counters for it FIRST, as their own commit (the `setup=7.13s` memset discovery took one pod run once the buckets existed).

## 2. Price before betting

State the expected saving from measured constants before writing code: deleted host node cost + deleted edge bytes/bandwidth − added edge cost (`scripts/dag_model.py`; update its constants from your run). Under ~1s of a ~30s fit: don't spend a pod on it. Constants drift ~25% across hosts — only play moves that win across the plausible range.

## 3. Implement with gates

/quality-gates per commit. Multi-commit change sets on one branch/PR (the PR #34 pattern); instrumentation ships in the same PR as the optimization it prices.

## 4. Validate same-pod

Before/after on ONE pod (/pod-validate), r² exactly equal for behavior-preserving changes, ledger delta vs the priced expectation. If measurement refutes the move: close the PR unmerged and write the refutation into docs/decisions.md with the same care as a win — refutations are deliverables (decisions 52, 55; PR #35).

## 5. Record

Decision entry (numbered, with the measured delta and the rejected alternatives), PR body carries the ledger table, README performance section only changes with re-baseline data behind it.
