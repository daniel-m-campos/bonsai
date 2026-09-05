---
name: design-ratchet
description: >
  bonsai's design-review ritual with numbers: run scripts/design_lint.py,
  read each metric that moved the wrong way as a question about a name, a
  home, or a seam, and decide whether to fix or to re-pin with a reason.
  Also carries the altitude test (hardware vocabulary above the engine seam)
  and the admission test for a new concept. Use on a diff before asking for
  review, or on the tree for a sweep. The gate's explanation: design_lint
  runs in docs-check, and this is how to read what it refuses.
---

A style preference is enforceable once it has a number and a direction. `scripts/comment_lint.py` proved the shape in this repo: a judgment that three sweeps failed to apply became a mechanical rule, and the comment count fell from 2145 to 207 only then. This skill applies the same move to design. The numbers earned their gate after three reviews in which they changed the outcome without a false positive (the ledger below), so `scripts/design_lint.py` runs in `docs-check` and a gated row that rises fails the build until it is fixed or re-pinned with a reason.

## Run it

`python3 scripts/design_lint.py` prints one row per metric against `scripts/design_baseline.json`. On a branch, export `origin/main` (`git archive origin/main | tar -x -C <tmp>`, never a checkout that could touch the working tree) and run `python3 scripts/design_lint.py --against <tmp>`: the scoreboard is the tree against the pinned baseline, and the churn block under it is the review, every file and word whose count moved between main and the branch. `REGRESSED` marks a gated row that rose; `tracked` rows never fail but their deltas are read the same way.

Read the churn, not the totals. A total hides a swap: the hunt admitted six words and retired six, and the row read +0. A fall can be a divergence rather than a consolidation: `fit.cpp` lost two clone windows because its prologue drifted away from the three commands it used to match, and that is a fourth copy, not one fewer. A retired word is usually the ratchet working (a concept gained its second home); an admitted one faces the test below.

## Read a regression as a question

Each metric is one sentence of the house style, so each rise has one question:

- `clone_windows` rose: which two sites now share a shape, and what is the one name that houses both? The hunt's three width messages were this exact case, and the fix was one helper with the caller naming what it passed.
- `contract_prose_lines` or `binding_prose_lines` rose: for each new line, could a name, a type, or a test have carried it? A contract that survives that question stays; the rest becomes an extraction, the same routing the comment policy already applies to `src/`.
- `vocabulary_singletons` rose: the new word faces the admission test below.
- `hardware_leaks` rose: a caller above the engine seam is asking about the machine. The seam answers such questions itself, so the fix is a method on the seam, not a string test at the call site.
- `shape_clone_windows` rose alone: a block was copied and renamed, or a legitimate table gained a row. Read the top sites to tell which.

## Scope: the diff, not the tree

Four ratchet rounds settled what this skill is for. `clone_windows` fell from 431 to 127 across PRs #448 to #459 and then moved by 3 in a round; the last two rounds ran on cyclomatic complexity in `scripts/`, which the lint did not scan at the time, so the scoreboard read flat by construction. The instrument is at its floor as a target and stays useful as a question about a diff. So the skill runs per PR, on the code the diff touches, and a standalone round is a sweep with a reason (a new instrument, a new blind spot), not a cadence.

| trigger | rule | proof |
|---|---|---|
| any PR touching code | the scoreboard above, main against the branch; every rising gated row gets FIX or PIN in its commit message | scoreboard in the PR body |
| a function the diff touches reads over CCN 12 (`uvx lizard -l cpp -l python -C 12 -w <touched files>`) | pin its exact outputs with a test first, then split in a second commit, or write `left whole: <reason>` in the PR body | a `test:` commit, then a `refactor:` commit |
| a block the diff touches shows in `uvx lizard -Eduplicate -l cpp -l python <touched files>` (token-based, so it sees a copy whose names and literals drifted: the five objective evals shared one reduction that the exact window read as 0) | pin its exact outputs with a test first, then house the shape in one name in a second commit | a `test:` commit, then a `refactor:` commit |
| the split touches a gate script (`scripts/*_lint.py`, `update_standings.py`, `render_*.py`) | main's copy against the branch's over the real corpus, byte-identical; run main's copy with `PYTHONPATH=scripts` so its sibling imports resolve | the commit's `Ran:` line |
| the split touches training or serialization | `scripts/model_hash.py` unchanged | the commit's `Ran:` line |
| the split is inside a kernel or fill loop | a SASS diff or a same-pod min before merge, else leave it whole and say so | the perf ledger |

Pin before moving, always. The pin is the durable output: the rounds gave `standings_refresh.py` its first 17 tests and `comment_lint.py` its first 18, and the greedy cut walk's exact-cut pin caught its own first draft copying from a dangling span. Targets come from reading the diff, not from the scoreboard: the exact-window clone measure misses a copy whose lines drifted, and it does not see complexity at all. Reasons a function stays whole are few and named: a hot loop, a constructor whose width is the design, a body with no injection seam (`standings_refresh.measure` rents a pod).

Only a feature or fix PR can score on the ledger below. A ratchet PR farms the metric it is meant to test.

## Altitude

There is no single correct altitude. A seam belongs where the cost model changes shape, and this tree has three regimes: statistical (objective, split, leaf, where a question is settled by a rule about the type, per decision 116), layout (bins, SoA, blocked rows, where a question is settled by a number, per the `perf:` tag's admission test), and device (warp, stream, occupancy, settled by same-pod measurement behind `HistogramEngine`). The Metal engine satisfying that concept in 700 lines where CUDA needs 4000 is the evidence the seam sits right: it hides exactly what varies and exposes exactly what costs.

Two tests hold a seam at its altitude. Containment: the vocabulary of one regime does not appear above its seam, which is what `hardware_leaks` counts. Conservation: below a seam, the parts sum to the whole, which is doc 16's rule for profiles and applies equally to a vocabulary, since a layer whose cost cannot be decomposed in its own words is at the wrong altitude.

## Admit a concept

A new word in a public name earns its place only by passing one of two tests, in the spirit of the comment lint's tags: a rule that distinguishes it from its nearest neighbour concept, or a measurement in which the neighbour loses. Decision 116 is the template for the second (a type versus a value, ceiling measured at zero, the simpler concept kept) and decision 74 for the first. A word with neither is a metaphor, and the plain noun that already exists replaces it. Record the verdict in `docs/decisions.md` with the rejected alternative and a reopener, so the argument is had once.

## Re-pin only on purpose

A lower number re-pins freely: run `--update-baseline` in the commit that improved it. A higher number re-pins only in a commit whose message says which metric rose and why the rise is the design; that message is the whole audit trail, so a rise nobody could explain is a fix, not a pin.

## Output contract

The scoreboard, then one line per rising row: `<metric> +N: <the question above, answered>. FIX: <the name, home, or seam method> | PIN: <why the rise is the design>`. Then the findings the numbers could not see, if any: the same rule written under different words is invisible to both clone measures when each copy is shorter than a window, and the vocabulary churn is where it shows. Append a review in which a number changed the outcome to the ledger below; a false positive among the gated rows is recorded there too, since it is the evidence for narrowing the metric.

## Promotion ledger

Reviews in which a number changed the outcome, with no false positive among the gated rows.

1. PR #447, 2026-09-01. Found: the width rule written three times with three messages, three names for the keys an invocation states, a stale test citation, four clone pairs, and 24 lines of decision-grade rationale in a header. Applied in one commit: `clone_windows` from +6 to -13, `contract_prose_lines` from +121 to +89 with the rest pinned as new contracts. Instrument lessons: include blocks were 38% of the clone count and are now skipped; `--against` exists because the totals hid the swap and the divergence.
2. PR #458, 2026-09-03. Found: after the predict and SHAP walks took a row view, `clone_windows` read +2 and named the staging prologue the two launchers now shared; the eval plane's second map in the next commit took it back to flat, and the `shape_clone_windows` +1 that remained was pinned as the one home the row map was introduced to be. No false positive among the gated rows.
3. PR #462, 2026-09-04. Found: `contract_prose_lines` +4 from a first draft of the `GrowResult` and `finish_round` contracts, cut by rewriting both in place; `vocabulary_singletons` +1 for `LeafBounds`, adjudicated in decision 121 (`bounds` is the word `SplitInput`'s contract already uses for `lo` and `hi`; `Range` and `interval` name other things) and pinned with the reopener there. No false positive; the third review, and the gate went into `docs-check` after it.
