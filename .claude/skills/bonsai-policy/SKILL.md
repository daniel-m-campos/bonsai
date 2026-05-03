---
name: bonsai-policy
description: >
  Binding AI-usage policy for the bonsai repo (MPCS 51045 final project).
  Read before any code-writing, design-decision, or core-architecture task.
  Triggers on: writing files under src/ or include/, implementing core
  components (Dataset, Booster, registry, TreeGrower, SplitFinder, Objective,
  ParallelBackend), resolving deferred design questions, or any commit
  involving AI-assisted code. Also triggers on phrases like "implement X",
  "write the booster", "draft the dispatch", "scaffold src/".
---

# Bonsai AI Usage Policy

This repo is governed by an explicit AI-usage policy set by the course
professor (MPCS 51045). Full text in `docs/ai-usage.md`. Operational
summary in `docs/context.md` §11. **This skill is the in-context
guardrail — read every time you're about to write code, make a design
decision, or commit.**

## The two rules

1. **Disclose** how AI was used at every stage. Commit trailer:
   `AI-Assisted: <short description>`.
2. **The user authors the core themselves.** AI does not ghost-write
   the core implementation or architecture. AI assists *afterward* with
   tests, build, CLI, benchmarks, extensions, docs, diagrams.

## Hard gate (do not write code without confirmation)

The user has stated: **no source files exist in this repo until the
professor signs off on the policy interpretation**. As of this skill's
last update, sign-off has not happened.

If you (the agent) are about to:
- Create any file under `src/` or `include/`
- Edit any file under `src/` or `include/`
- Run a code generator, scaffolder, or template that produces source
  files in those paths

**Stop and ask the user**:
> "The bonsai-policy skill blocks writing under src/ or include/ until
> the professor has signed off and you've hand-authored the relevant
> core component. Has the sign-off happened, and have you written the
> core path for what I'm about to touch?"

Proceed only on explicit user confirmation, not on inference from
context.

## The "spine" — user must hand-author first

These pieces *must* exist in user-written form before any AI assistance
in those areas:

- `Dataset`, `BinMapper`, `Histogram` (core data structures)
- `Booster<...>` (training loop, gradient/score management,
  `update_one_iter` semantics)
- Registry / dispatch mechanism (the runtime → static boundary)
- Depth-wise `TreeGrower` and histogram `SplitFinder`
- `Objective` concept + MVP impls (MSE, logloss)
- `ParallelBackend` concept + first impl (`SerialBackend`, then
  `OpenMPBackend`)
- Resolution of the deferred dispatch-shape design question (flagged
  in `docs/proposal.md` §3.4) — written by the user, in their own
  voice, into `docs/architecture/6-dispatch.md`

If the user asks you to write any of the above before the spine is
in place, **decline and remind them of the policy**. They may
override; if they do, the override gets logged in
`docs/ai-usage.md` audit trail.

## Always allowed (no spine prerequisite)

- Drafting and editing docs: `proposal.md`, `architecture/*.md`,
  `decisions.md`, READMEs.
- Reference-library surveys, tradeoff summaries.
- Diagrams (architecture, flow, data layout).
- Editorial cleanup of user-written prose or code.
- Searches, lookups, reading existing code.
- The `Makefile` and project setup tooling.

## Allowed once the spine works end-to-end

- Tests (unit, integration, golden-file, parity, determinism).
- Build system (CMake, FetchContent, CI).
- CLI plumbing (argparse, subcommand wiring, config loading,
  progress bars).
- Benchmark harness, dataset loaders, reproduction scripts.
- Phase 4 extensions (leaf-wise/oblivious growers, GOSS, exact
  splitter, categorical handlers, additional objectives like quantile
  or huber).
- Parallel backends *after* the user has written `SerialBackend` and
  `OpenMPBackend` themselves (e.g., `StdExecBackend`).
- Boilerplate (header guards, includes, forward decls, CSV glue).

## Never

- AI ghost-writing the core data structures or training loop.
- AI making design decisions the user accepts without independent
  rationale.
- Any AI assistance without disclosure (commit trailer or audit-trail
  entry).

## Operational obligations for agents

Before non-trivial work:

1. Read `docs/context.md` (which links to `docs/ai-usage.md`).
2. If creating files under `src/` or `include/`: confirm sign-off and
   user authorship as above.
3. If unsure whether a task crosses the line: stop and ask, do not
   guess.
4. When drafting commit messages, surface AI authorship via the
   `AI-Assisted: <short description>` trailer.

## When this skill is wrong

The user can override any rule here, but overrides must be:
1. Explicit (not inferred from context).
2. Logged in `docs/ai-usage.md` audit trail.

If the professor's sign-off arrives with policy adjustments, those
land in `docs/ai-usage.md` "Open questions for the professor" section,
and this skill gets updated to match. Do not silently relax the rules.