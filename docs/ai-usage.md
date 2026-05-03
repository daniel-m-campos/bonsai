# AI Usage Policy and Audit Trail

> Authoritative for this repo; binding on the user and any AI agent. Operational
> summary in [`docs/context.md` §11](context.md). This doc holds full text,
> rationale, and audit trail.

## Course policy

MPCS 51045 (Advanced C++, Spring 2026). Professor's policy by email:

> As for AI, it is first critical that you share exactly how you use AI.
> Beyond that, I would say that it is a little similar to where this week's
> lecture ended up. Write the core functionality and architecture yourself,
> but once you have done so, judicious use of AI to add features, tests,
> produce architecture diagrams, etc. is ok. But to be clear, I'd like the
> initial implementation to be written by you.

Two requirements: **(1) disclose** how AI is used at every stage; **(2) author
the core yourself** — initial implementation by hand, AI assistance afterward
for features, tests, diagrams, etc.

When ambiguous, err toward writing it yourself and disclosing more.

## Core that the user must hand-author

The "spine" — first working version written by the user, no AI ghost-writing:

- `Dataset`, `BinMapper`, `Histogram`.
- `Booster<...>` — training loop, gradient/score management, `update_one_iter`.
- Registry / dispatch mechanism (runtime → static boundary).
- Depth-wise `TreeGrower`, histogram `SplitFinder`.
- `Objective` concept + MVP impls (MSE, logloss).
- `ParallelBackend` concept + first impl (`SerialBackend`, then `OpenMPBackend`).
- Resolution of the deferred dispatch-shape design question (flagged in
  [`proposal.md` §3.4](proposal.md)) — written into the relevant
  `docs/architecture/N-*.md` doc (likely `6-dispatch.md`) by
  the user, in their own voice, before any code is written against it.

Once the spine works end-to-end on the MVP regression dataset, AI assistance
opens up for the rest.

## What AI may do, and when

**Always allowed** (does not touch core implementation):

- Documentation drafting and editorial cleanup (`proposal.md`,
  `docs/architecture/*.md`, ADRs, READMEs). User owns design decisions;
  AI drafts prose.
- Reference-library surveys, tradeoff summaries.
- Diagrams (architecture, flow, data layout).
- Searches and lookups.

**Allowed after the Phase 1 core is working end-to-end:**

- Tests (unit, integration, golden-file, parity, determinism).
- Build system (CMake, FetchContent, CI).
- CLI plumbing (argument parsing, subcommand wiring, config loading,
  progress-bar integration).
- Benchmark harness, dataset loaders, reproduction scripts.
- Phase 4 extensions (leaf-wise/oblivious growers, GOSS, exact splitter,
  categorical handlers, additional objectives). User reviews each.
- Parallel backends *after* `SerialBackend` and `OpenMPBackend` exist —
  e.g., `StdExecBackend`.
- Boilerplate (header guards, includes, forward decls, CSV glue).

**Never:**

- AI ghost-writing core data structures or training loop.
- AI making design decisions the user then accepts without independent rationale.
- Any AI assistance without disclosure.

## Operational rules

**Commit trailers.** Any commit with AI-written or AI-assisted content
includes:

```
AI-Assisted: <short description>
```

Examples: `AI-Assisted: tests authored with Claude (test_objective.cpp)`,
`AI-Assisted: CMake refactor edited with Claude`,
`AI-Assisted: prose drafting from user-supplied outline`.
100% user-written commits have no trailer (absence = default).

**Transcripts.** Non-trivial AI design conversations preserved at
`docs/conversations/<YYYY-MM-DD>-<slug>.md`. First entry:
[`2026-05-02-initial-design.md`](conversations/2026-05-02-initial-design.md) —
the conversation that produced [`proposal.md`](proposal.md) and
[`docs/context.md`](context.md). Capture conversations that produce
artifacts or make design decisions, not every chat.

**Professor sign-off.** User shares this repo with the professor *before* any
implementation code is written. The intent is to confirm the policy
interpretation before forging ahead.

**Pre-sign-off override.** As of 2026-05-03 the user has elected to proceed
with implementation before the professor has confirmed the interpretation.
If the professor objects on review, AI-assisted work to that point is
identifiable via commit trailers (`AI-Assisted: ...`) and the audit-trail
table below, and may be rolled back. The skill (`.claude/skills/bonsai-policy`)
remains strict; agents will continue to ask before creating files under
`src/` or `include/`, and the user explicitly confirms each time.

**Agent obligations.** Any AI agent in this repo:

1. Reads `docs/context.md` (which links here) before non-trivial work.
2. Does not create `src/` or `include/` files without explicit confirmation
   that the professor has signed off *and* the user has hand-authored the
   relevant core component.
3. Stops and asks when a task arguably crosses the line.
4. Surfaces AI authorship in commit message drafts.

## Audit trail

Append-only. New entries at the bottom.

| Date | Artifact | AI's role | Notes |
|---|---|---|---|
| 2026-05-02 | Project scoping, library survey, design discussion | Conversation partner; surveyed xgboost/LightGBM/CatBoost; offered option spaces; pushed back and was pushed back on. | User drove all decisions. Transcript: `docs/conversations/2026-05-02-initial-design.md`. |
| 2026-05-02 | `docs/context.md` (initial draft) | Drafted from the conversation. | Decisions were already the user's. |
| 2026-05-02 | `proposal.md` (initial draft) | Drafted from `context.md` §8 structure. | User narrowed scope twice, removed weeks timeline, reviewed each section. |
| 2026-05-02 | `docs/context.md` + `docs/ai-usage.md` (AI policy) | Edits applied per user direction. | User authored the policy; AI drafted prose. |
| 2026-05-02 | `Makefile` (`make skills` target) | Drafted by AI. | Project-local `make skills` installs the [caveman](https://github.com/juliusbrussee/caveman) Claude Code skill into `.claude/skills/` (gitignored). One-time fetch from upstream; opt-in. |
| 2026-05-02 | `.claude/skills/bonsai-policy/SKILL.md` | Drafted by AI from this doc + `docs/context.md` §11. | In-context guardrail skill that auto-loads when an agent is about to write code or make a design decision. Restates the policy and adds a hard-stop on writes under `src/` or `include/` until the user confirms professor sign-off. |
| 2026-05-02 | Dataset-design discussion + `docs/decisions.md` entries 1–6 + `docs/architecture/1-dataset.md` + `docs/architecture/8-config.md` (data slice) | AI as sounding board (surveyed lgbm/xgb/catboost binning, dataset, reader strategies; named idiomatic options at each step; pushed back where appropriate). User decided each entry. AI drafted prose. | Spans two sessions. User narrowed and corrected the design repeatedly: rejected lgbm-style reference-Dataset coupling, picked free-function readers over a `Reader` concept, dropped the variant + `visit_column` complexity, moved I/O out of domain types. Transcript: this conversation. |
| 2026-05-03 | Pre-sign-off implementation override | n/a (user decision) | User elected to begin implementation before the professor confirmed AI policy. AI-assisted artifacts are identifiable via commit trailers and this audit table for rollback if needed. |

When implementation starts, this table grows per AI-assisted artifact (or per
phase if grouping is cleaner).

## Open questions for the professor

User plans to confirm before code is written:

1. AI-drafted documentation acceptable when the user owns every design
   decision and assistance is disclosed? (User's reading: yes, per "produce
   architecture diagrams, etc.")
2. Deferred dispatch-shape design (in `docs/architecture/6-dispatch.md`
   before code) — does user-written prose with AI editorial cleanup
   afterward satisfy "architecture written by you"?
3. Is the boundary above (user hand-writes spine; AI assists with tests,
   build, CLI, benchmarks, Phase 4) the right interpretation?
4. Is the commit-trailer convention sufficient, or does the professor want
   a different format (per-file header, per-PR summary)?

Professor's answers, once received, get recorded here and override the user's
interpretation where they conflict.
