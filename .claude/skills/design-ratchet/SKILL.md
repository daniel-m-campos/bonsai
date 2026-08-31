---
name: design-ratchet
description: >
  bonsai's design-review ritual with numbers: run scripts/design_lint.py,
  read each metric that moved the wrong way as a question about a name, a
  home, or a seam, and decide whether to fix or to re-pin with a reason.
  Also carries the altitude test (hardware vocabulary above the engine seam)
  and the admission test for a new concept. Use on a diff before asking for
  review, or on the tree for a sweep. A skill first; it becomes a CI gate
  once it has changed the outcome of three reviews without a false positive.
---

A style preference is enforceable once it has a number and a direction. `scripts/comment_lint.py` proved the shape in this repo: a judgment that three sweeps failed to apply became a mechanical rule, and the comment count fell from 2145 to 207 only then. This skill applies the same move to design, by hand, until the numbers have earned a gate.

## Run it

`python3 scripts/design_lint.py` prints one row per metric against `scripts/design_baseline.json`. On a branch, run it once on `origin/main` (export with `git archive origin/main | tar -x -C <tmp>` and pass `--root <tmp>`, never a checkout that could touch the working tree) and once on the branch head; the deltas between the two are the review, not the absolute values. `REGRESSED` marks a gated row that rose; `tracked` rows never fail but their deltas are read the same way.

## Read a regression as a question

Each metric is one sentence of the house style, so each rise has one question:

- `clone_windows` rose: which two sites now share a shape, and what is the one name that houses both? The hunt's three width messages were this exact case, and the fix was one helper with the caller naming what it passed.
- `contract_prose_lines` or `binding_prose_lines` rose: for each new line, could a name, a type, or a test have carried it? A contract that survives that question stays; the rest becomes an extraction, the same routing the comment policy already applies to `src/`.
- `vocabulary_singletons` rose: the new word faces the admission test below.
- `hardware_leaks` rose: a caller above the engine seam is asking about the machine. The seam answers such questions itself, so the fix is a method on the seam, not a string test at the call site.
- `shape_clone_windows` rose alone: a block was copied and renamed, or a legitimate table gained a row. Read the top sites to tell which.

## Altitude

There is no single correct altitude. A seam belongs where the cost model changes shape, and this tree has three regimes: statistical (objective, split, leaf, where a question is settled by a rule about the type, per decision 116), layout (bins, SoA, blocked rows, where a question is settled by a number, per the `perf:` tag's admission test), and device (warp, stream, occupancy, settled by same-pod measurement behind `HistogramEngine`). The Metal engine satisfying that concept in 700 lines where CUDA needs 4000 is the evidence the seam sits right: it hides exactly what varies and exposes exactly what costs.

Two tests hold a seam at its altitude. Containment: the vocabulary of one regime does not appear above its seam, which is what `hardware_leaks` counts. Conservation: below a seam, the parts sum to the whole, which is doc 16's rule for profiles and applies equally to a vocabulary, since a layer whose cost cannot be decomposed in its own words is at the wrong altitude.

## Admit a concept

A new word in a public name earns its place only by passing one of two tests, in the spirit of the comment lint's tags: a rule that distinguishes it from its nearest neighbour concept, or a measurement in which the neighbour loses. Decision 116 is the template for the second (a type versus a value, ceiling measured at zero, the simpler concept kept) and decision 74 for the first. A word with neither is a metaphor, and the plain noun that already exists replaces it. Record the verdict in `docs/decisions.md` with the rejected alternative and a reopener, so the argument is had once.

## Re-pin only on purpose

A lower number re-pins freely: run `--update-baseline` in the commit that improved it. A higher number re-pins only in a commit whose message says which metric rose and why the rise is the design; that message is the whole audit trail, so a rise nobody could explain is a fix, not a pin.

## Output contract

The scoreboard, then one line per rising row: `<metric> +N: <the question above, answered>. FIX: <the name, home, or seam method> | PIN: <why the rise is the design>`. End with `PROMOTE: <n>/3` counting the reviews so far in which a number changed the outcome without a false positive; at 3, add `@python3 scripts/design_lint.py` back to `docs-check` and this skill becomes the gate's explanation.
