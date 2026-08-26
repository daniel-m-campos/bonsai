# CLAUDE.md

Working agreement for agents in this repository. Where content lives is governed by the routing table in [docs/STYLE.md](docs/STYLE.md); this file states the rules you need every session and points at everything else.

## Conventions

- Commits follow Conventional Commits v1: lowercase imperative description, title 50 characters target and 72 hard cap, body wrapped at 72. No session URLs or generated trailers.
- Branch and open a PR for every change; the maintainer merges. Never push to main, never merge, never close someone else's PR.
- Sync branches by rebase onto main, not merge commits.
- Prose style (docs, PR bodies, issues): no em-dashes; commas, colons, or parentheses instead. One line per paragraph in Markdown, the site soft-wraps. Plain nouns over coined metaphors.
- Code comments are sparse, single-line constraint statements. The exception is a contract-bearing type, which carries a full contract comment: what it guarantees, what breaks it, and the test that pins it. `RecycledOutputs` in [include/bonsai/grower.hpp](include/bonsai/grower.hpp) is the house exemplar. No PR, issue, or decision references in code comments; cite those in docs and commit messages.
- Python follows the repo Python standards skill (3.9 floor, `from __future__ import annotations`, numpy-style docstrings on the public surface).

## Never

- Never run bare `git stash` or `git stash pop`. The stash stack is shared across all worktrees and other sessions may push or pop concurrently. Prefer a temporary WIP commit; if you must stash, use a tagged `git stash push -u -m` and `apply` by SHA.
- Never `git add -A` or `git add -u`. Stage by explicit path. Long-lived uncommitted probe hunks may be present in the working tree and must never reach a commit.
- Never hand-edit generated files. They are produced by `make docs-check` renderers and the build: `docs/use/parameters.md`, `docs/use/make-map.md`, `docs/learn/timeline.md`, `docs/method/results/*`, the README standings block, and `python/bonsai/_params.py`. Edit the source or the generator, then re-render.
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
- Build: `make python` builds the extension into `build/python` (tests read it via `PYTHONPATH=build/python`). `make docs-check`, `make lint`, `make lint-python`, `make format-check` are the local gates; run them before any PR.

## Verification floor

Before opening a PR: format-check, lint (read all output), lint-python, docs-check, the C++ tests via ctest, and pytest. A change touching training or serialization must not move `scripts/model_hash.py` unless that is its stated purpose; wire identity is the gate.
