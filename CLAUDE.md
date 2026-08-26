# CLAUDE.md

Working agreement for agents in this repository. Where content lives is governed by the routing table in [docs/STYLE.md](docs/STYLE.md); this file states the rules you need every session and points at everything else.

## Conventions

- Commits follow Conventional Commits v1: lowercase imperative description, title 50 characters target and 72 hard cap, body wrapped at 72. No session URLs or generated trailers.
- Branch and open a PR for every change; the maintainer merges. Never push to main, never merge, never close someone else's PR.
- Sync branches by rebase onto main, not merge commits.
- Prose style (docs, PR bodies, issues): no em-dashes; commas, colons, or parentheses instead. One line per paragraph in Markdown, the site soft-wraps. Plain nouns over coined metaphors.
- **Implementation code carries no comments.** `src/` is implementation whatever the file extension says, since the templated hot path lives in headers there. `scripts/comment_lint.py` enforces this as a hard gate in `make docs-check`; it is a rule, not a judgment call, because three separate judgment-based audits each left the same comments standing. If code needs explaining, the fix is a name: extract an inline function, name a constant, introduce a type. A name is checked by the compiler and carried by a rename, where a comment is only asserted and gets left behind. A prose decoder for magic indices or flags is a missing enum. A constraint belongs where it can be violated, not at the declaration of the thing it constrains.
- Exceptions are tagged, and **a tag is only legitimate if it has a mechanical admission test**. "This one is genuine" is not a test; it is the judgment call that failed three times. Three tags exist, each checked by the lint:

| tag | carries | admission test |
|---|---|---|
| `// perf:` | a measurement or hardware or build fact | the block or the declaration under it contains a number. A hardware claim with no number is not evidence, so measure it or drop it |
| `// sync:` | a concurrency or ordering constraint whose violation is a race, undefined behaviour, or a silently wrong answer | the block names a construct from the lint's vocabulary AND that construct appears in the file's code |
| `// ffi:` | a Python boundary constraint (GIL, capsule ownership, DLPack lifetime) | only `src/python/` may carry it, and it names a boundary construct the file uses |

- Structural beats tagged, always. Before writing `// sync:`, try to make the constraint impossible: hoist the throwing validation above the parallel region, funnel device placement through one call site. A constraint that cannot be violated needs no comment. Adding a tag is the fallback, and adding a fourth tag means designing its admission test first.
- Contract comments live in `include/`, above the public declaration they bind, stating what it guarantees, what breaks it, and the test that pins it. `RecycledOutputs` in [include/bonsai/grower.hpp](include/bonsai/grower.hpp) is the house exemplar. Plain `//`, not doxygen: nothing renders doxygen markup here, so it would be dead weight.
- A correctness constraint that survives none of the above becomes a test carrying an `// INVARIANT: <id>` marker, or it goes. No PR, issue, or decision references anywhere in code: keep the measurement, cite the source in docs and commit messages.
- Python follows the repo Python standards skill (3.9 floor, `from __future__ import annotations`, numpy-style docstrings on the public surface).
- Catch2 test names read `"<Component>: <behavior under condition>"`: PascalCase component, colon, present-tense behavioral phrase. One `TEST_CASE` per behavior, `SECTION` for variations. Tags stack the component with behavioral tags from the fixed set (`[fit]`, `[transform]`, `[ctor]`, `[edge]`, `[nan]`, `[perf]`, `[smoke]`).

## Never

- Never run bare `git stash` or `git stash pop`. The stash stack is shared across all worktrees and other sessions may push or pop concurrently. Prefer a temporary WIP commit; if you must stash, use a tagged `git stash push -u -m` and `apply` by SHA.
- Never `git add -A` or `git add -u`. Stage by explicit path. Long-lived uncommitted probe hunks may be present in the working tree and must never reach a commit.
- Never hand-edit generated files. They are produced by `make docs-check` renderers and the build: `docs/invariants.md`, `docs/use/parameters.md`, `docs/learn/timeline.md`, `docs/method/results/*`, the README standings block, and `python/bonsai/_params.py`. Edit the source or the generator, then re-render.
- Never document a contract in prose when a test can hold it. A contract is worth writing down only if something fails when it breaks, so an invariant is a test carrying an `// INVARIANT: <id>` marker, and `docs/invariants.md` is generated from those markers. Only genuinely untestable contracts (build flags, packaging, nondeterminism claims, decisions) are hand-written, in `docs/invariants.src.md`, where they render as asserted rather than enforced. Full rules: [docs/STYLE.md](docs/STYLE.md), "Writing an invariant".
- Never hand-edit provenance: `benchmarks/standings.json` stamps, measured shas, and results rows are written by `scripts/update_standings.py` and the refresh machinery only. A row whose provenance is wrong is re-measured, not corrected in place.
- Never quote cross-pod absolute timings; only same-pod interleaved comparisons are valid. See the benchmark protocol.
- Never leave a rented pod running unattended past its work, and never end a session without verifying the fleet is empty.
- Run linters after renderers, not before: a render can rewrite what a linter just approved.

## Where things are

- Routing policy (what content goes where): [docs/STYLE.md](docs/STYLE.md), section "Where things live".
- Cross-file contracts and invariants: [docs/invariants.md](docs/invariants.md). Linted by `make docs-check`; cite entries instead of restating them.
- Design rationale, measurements, rejected alternatives: [docs/decisions.md](docs/decisions.md), append-only, cited by number. Corrections are italic status banners, never body edits.
- Procedures: [docs/ops/runpod-runbook.md](docs/ops/runpod-runbook.md) for pod work, `.claude/skills/` for the repo rituals (quality gates, pod validation, perf rounds, feature admission, review rounds).
- Reader-facing docs: `docs/guide/` and `docs/learn/` teach; `docs/use/` and `docs/method/` are reference, generated where possible.

## Layout

- `include/bonsai/` public headers; `src/` implementation; `src/cuda/` the device backend; `src/python/` the nanobind module; `src/cli/` the CLI.
- `python/bonsai/` the Python package; `python/bonsai/bench/` the benchmark harness; `python/tests/` pytest.
- `tests/` Catch2 C++ suites; `scripts/` tooling and generators; `benchmarks/` results ledger and evidence records; `configs/` dataset TOMLs.
- Build: `make python` builds the extension into `build/python` (tests read it via `PYTHONPATH=build/python`). `make ci` runs the gates.

## Verification floor

Run `make ci` before opening a PR (`ARGS=--fast` skips clang-tidy). It runs every gate this host can run and NAMES the ones it cannot, because a gate you forgot and a gate that passed look identical in a checklist. The three it cannot run on a laptop (sanitizers, the race gate, the CUDA compile) run in CI and on a GPU host.

A change touching training or serialization must not move `scripts/model_hash.py` unless that is its stated purpose; wire identity is the gate.
