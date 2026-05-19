---
name: bonsai-policy
description: >
  Binding AI-usage policy for the bonsai repo (MPCS 51045 final project).
  Read before any code-writing or design-decision task. The spine-authorship
  gate that operated during Phases 1–2 was lifted on 2026-05-18 (decision
  28); the remaining rules are disclosure, transcript capture, and
  stop-and-ask before scope creep. Triggers on: writing files under src/
  or include/, drafting design docs, resolving deferred design questions,
  or any commit involving AI-assisted code. Also triggers on phrases like
  "implement X", "refactor Y", "scaffold the parallel backend".
---

# Bonsai AI Usage Policy

This repo is governed by an explicit AI-usage policy set by the course
professor (MPCS 51045). Full text and audit trail in
`docs/ai-usage.md`. Operational summary in `docs/context.md` §11.
**This skill is the in-context guardrail — read every time you're
about to write code, make a design decision, or commit.**

## The two course rules

1. **Disclose** how AI was used at every stage. Commit trailer:
   `AI-Assisted: <short description>`. Absence of trailer means
   100% user-written.
2. **The user authors the core themselves.** This rule was satisfied
   by the user across Phases 1–2 (spine complete on 2026-05-18,
   decision 28). AI assistance is now open across the codebase.

## What changed on 2026-05-18

The spine-authorship gate that previously stopped AI from creating
or editing certain `src/` and `include/` files is **lifted**. The
user has hand-authored every spine component required to ship the
MVP and benchmark harness end-to-end (Dataset, BinMapper, BinMappers,
Histogram, depth-wise + oblivious TreeGrower, DenseTree,
ObliviousTree, histogram SplitFinder, Objective concept + MSE +
LogLoss, Sampler concept + AllRowsSampler, Booster + IBooster,
registry / dispatch flat table). The `ParallelBackend` concept and
its first impl are reclassified as Phase 3 work — the user will own
the threading-model design in `docs/architecture/7-parallel.md`
before AI implements from it, preserving the "user authors the
architecture" intent.

From this date forward, AI may refactor or extend any file in the
tree.

## Allowed (broadly)

- Refactoring anywhere — spine, CLI, config, I/O, registry, tests.
- Implementing new features: additional objectives, samplers, growers,
  split finders, parallel backends (once `7-parallel.md` lands),
  categorical handlers.
- Tests (unit, integration, golden-file, parity, determinism).
- Build system, CI, benchmark harness, dataset loaders, scripts.
- Docs, decisions log entries, diagrams.
- Commit messages and PR descriptions.

## Always

- **Disclose AI authorship via trailer.** Any commit with AI-written
  content gets `AI-Assisted: <short description>`. Absence = 100%
  user-written.
- **Preserve non-trivial design conversations** under
  `docs/conversations/<YYYY-MM-DD>-<slug>.md`. Capture conversations
  that produce artifacts or make design decisions, not every chat.
- **Stop and ask before destructive operations.** `git push --force`,
  `rm -rf`, mass file deletions, anything irreversible. Approval in
  one context does not transfer to the next.
- **Stop and ask before scope creep.** If you're about to expand a
  task beyond what the user asked, surface it first.
- **AI proposes, user decides.** When choosing between design
  alternatives, present options with tradeoffs; don't pre-pick.

## Never

- Silent AI authorship (no trailer).
- Decisions the user accepts without independent rationale.
- Force-push to `main`. Force-push to any shared branch without
  explicit per-task confirmation.

## Threading-model design exception

The `ParallelBackend` concept and the first parallel impl
(`SerialBackend` / `OpenMPBackend`) are the last items where the
"user authors the design first" pattern still applies — not as a
hard gate on AI edits, but as the natural order of work. The user
writes `docs/architecture/7-parallel.md` (currently TBD) before any
backend code is written. AI may then implement from that doc with
disclosure.

## Operational obligations

Before non-trivial work:

1. Read `docs/context.md` (which links to `docs/ai-usage.md`).
2. If unsure whether a task crosses the line: stop and ask, do not
   guess.
3. When drafting commit messages, surface AI authorship via the
   `AI-Assisted: <short description>` trailer.

## When this skill is wrong

The user can override any rule here, but overrides must be:
1. Explicit (not inferred from context).
2. Logged in `docs/ai-usage.md` audit trail.

If the professor sends follow-up clarifications, they land in
`docs/ai-usage.md` and this skill is updated to match. Do not
silently relax the rules.
