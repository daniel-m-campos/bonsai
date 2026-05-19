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

## Spine-authorship rule — satisfied 2026-05-18

The course rule "write the core functionality and architecture
yourself" required the user to hand-author the spine of the system
before AI assisted with that area. The list as originally specified:

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

**Status as of 2026-05-18 (decision 28).** Every item above is in
place except the parallel backends, which are reclassified as Phase 3
work — the user will own the threading-model design (`7-parallel.md`)
before AI implements from it, so the spine-authorship intent is
preserved. The course rule is satisfied; the spine gate as a hard
stop on AI edits is lifted.

## What AI may do, from 2026-05-18 onward

**Broadly allowed** — anywhere in the codebase, including spine
components:

- Refactoring existing code (spine, CLI, config, I/O, registry, tests).
- Implementing new features: additional objectives, samplers, growers,
  split finders, parallel backends, categorical handlers.
- Tests (unit, integration, golden-file, parity, determinism).
- Build system (CMake, FetchContent, CI).
- Benchmark harness, dataset loaders, reproduction scripts.
- Docs, decisions log entries, diagrams.
- Commit messages and PR descriptions.

**Always:**

- Disclose via `AI-Assisted: <short description>` trailer on any
  commit with AI-authored content. Absence = 100% user-written.
- Preserve non-trivial design conversations under
  `docs/conversations/<YYYY-MM-DD>-<slug>.md`.
- Stop and ask before destructive operations or anything outside
  the agreed scope. Approval in one task does not transfer.

**Never:**

- Silent AI authorship (no trailer).
- AI making design decisions the user accepts without independent
  rationale. AI proposes; user decides.

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

**Professor sign-off.** Received 2026-05-04. The professor reviewed this
policy and replied "LGTM," approving the interpretation as written. The
four open questions below resolve to the user's stated readings. The
pre-sign-off override (entered 2026-05-03) is moot — AI-assisted work
to that point stands and does not need rollback.

**Agent obligations.** Any AI agent in this repo:

1. Reads `docs/context.md` (which links here) before non-trivial work.
2. From 2026-05-18 onward: may touch any file in the tree (decision
   28). The spine-authorship gate is satisfied and lifted.
3. Stops and asks before destructive operations or scope creep.
   Approval in one task does not transfer to the next.
4. Surfaces AI authorship in commit message drafts via the
   `AI-Assisted: ...` trailer.

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
| 2026-05-04 | Professor sign-off on AI-usage policy | n/a (professor decision) | Professor replied "LGTM" to this policy doc. The four open questions below resolve to the user's stated readings. Spine-authorship rule remains; the sign-off gate is lifted. |
| 2026-05-04 | `docs/ai-usage.md` + `docs/context.md` §11 + `.claude/skills/bonsai-policy/SKILL.md` | AI applied edits per user direction reflecting professor sign-off. | User invoked `/bonsai-policy updatez: professor said LGTM`. AI rewrote the sign-off block, resolved the open questions, and replaced the "no code under src/include until sign-off" hard gate with a narrower "spine components require user authorship first" gate. User reviews. |
| 2026-05-05 | `docs/architecture/2-histogram.md` + `docs/decisions.md` §7 + cross-doc determinism updates + `tests/unit/test_histogram.cpp` | AI as sounding board (surveyed XGBoost/LightGBM/CatBoost determinism guarantees, identified relaxation opportunity); drafted design doc; drafted test cases. User hand-authored `include/bonsai/histogram.hpp` (spine component); AI flagged three correctness bugs in the first draft (uninitialized cells, `clear()` shrinking instead of zeroing, PMF mis-use in `operator-=`); user fixed. AI tightened algorithm framing, worked example, and AoS-vs-SoA rationale per user pushback. | Determinism contract relaxed to fixed-thread-count bytes + cross-thread-count tolerance per field consensus. Initial test draft used non-IEEE-exact decimals and tripped equality checks; AI caught and switched to dyadic fractions. Histogram is the second spine component to land. Transcript: this conversation. |
| 2026-05-07 | `docs/architecture/3-tree.md` + `docs/decisions.md` §§8–20 + `docs/architecture/README.md` (index update) | AI as sounding board (surveyed xgb's `colmaker` / `hist` / `RegTree`, lgbm's `SerialTreeLearner` / `Tree`, CatBoost's `TFullModel` / `TObliviousTreeCalcer` and confirmed CatBoost's column-major fast path is motivated by predict-time rebinarization that bonsai doesn't do); drafted design doc and decisions log entries. User decided each joint: pulling oblivious into Phase 1 to exercise the second-tree-type machinery early, `Tree` as concept, single `SplitFinder` concept (initial draft incorrectly proposed two concepts whose signatures collapsed to the same shape; user spotted it, AI corrected), strategy A partitioning, subtraction trick from day one for both growers, regularization knob set. AI surfaced the `LevelSplit` per-level-vs-per-node `default_left` choice; user picked per-level (CatBoost-faithful). | Discussion preserved as transcript. Doc and decisions are AI-drafted from user decisions; spine impl (growers, tree types) remains user-authored — not yet written. |
| 2026-05-07 | `include/bonsai/types.hpp` + `include/bonsai/tree.hpp` + `src/tree.cpp` + `tests/unit/test_dense_tree.cpp` + `tests/unit/test_oblivious_tree.cpp` + `tests/unit/test_tree_constants.hpp` | User hand-authored the spine code (`DenseTree`, `ObliviousTree`, `Tree` concept, types aliases). AI advised on type aliasing (rejected `row_t`/`column_t` for changing meaning per call site), spotted const-correctness and concept-signature bugs in early drafts, and authored the unit tests after the spine landed. | First two of four `3-tree.md` spine components shipped (tree types). `DepthwiseGrower`, `ObliviousGrower`, `HistogramSplitFinder`, and the `TreeGrower` / `SplitFinder` concept declarations are still pending — user-authored when written. |

| 2026-05-11 | `docs/architecture/4-objective.md` + `docs/decisions.md` §§21–25 + `docs/architecture/README.md` (index update) | AI surveyed xgb (`ObjFunction`, `RegLossObj` family), lgbm (`ObjectiveFunction`, virtual `GetGradients`), CatBoost (`IDerCalcer`) interface shapes; presented three interface options (concept-static, concept-instance, virtual base) and three scope options (initial_score, eval, transform, weighting). User picked concept-static + bundled eval, kept everything else off the concept (initial_score, transform, sample-weighting all booster-side). AI drafted doc prose and decisions entries from those choices. Spine impl (`Objective` concept declaration, `MSEObjective`, `LogLossObjective`) reserved for hand-authoring. | Doc + decisions only; no spine code written. Tests for the objectives land after user-authored impl, per usual. |
| 2026-05-18 | `docs/context.md` + `docs/decisions.md` §28 + `docs/ai-usage.md` + `.claude/skills/bonsai-policy/SKILL.md` (spine-gate lift + Phase 2.5 insertion + AI policy relaxation) | User declared the spine-authorship rule satisfied: Phase 1 MVP and Phase 2 benchmark harness end-to-end on California Housing, with eval baseline pinned at `rmse=0.7175214`. The `ParallelBackend` concept + first impl reclassified as Phase 3 work rather than spine-gated; the user will own the threading-model design (`7-parallel.md`) before AI implements from it. AI drafted prose for the policy update, decision §28, and the skill rewrite from the user's direction. | AI assistance opens up across the codebase, including refactors of spine components, from this date forward. Disclosure trailer + transcript-capture rules stand. Phase 2.5 (CLI design + cleanup) inserted between Phase 2 and Phase 3. |

When implementation starts, this table grows per AI-assisted artifact (or per
phase if grouping is cleaner).

## Open questions for the professor — resolved 2026-05-04

Professor replied "LGTM" on 2026-05-04, approving this policy as written.
All four questions resolve to the user's reading:

1. **AI-drafted documentation acceptable** when the user owns every design
   decision and assistance is disclosed. ✅
2. **Deferred dispatch-shape design** (in `docs/architecture/6-dispatch.md`
   before code) — user-written prose with AI editorial cleanup afterward
   satisfies "architecture written by you". ✅
3. **Boundary stands**: user hand-writes the spine; AI assists with tests,
   build, CLI, benchmarks, Phase 4. ✅
4. **Commit-trailer convention** (`AI-Assisted: ...`) is sufficient. ✅

If the professor sends follow-up clarifications later, they get appended
below this block and override the resolutions above where they conflict.
