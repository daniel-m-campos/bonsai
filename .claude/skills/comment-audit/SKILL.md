---
name: comment-audit
description: >
  Audit code comments against the CLAUDE.md comment policy: sparse constraint
  statements, contract essays only on contract-bearing types, no PR/issue/
  decision references, cross-file claims registered in docs/invariants.md.
  Use on a diff during review, or on a directory for a periodic sweep. Invoke
  via /comment-audit.
---

The policy is CLAUDE.md's comment paragraph and docs/STYLE.md's routing test; read both before judging anything. The exemplar of a correct contract comment is `RecycledOutputs` in `include/bonsai/grower.hpp`: it states what the type guarantees, what breaks it, and the test that pins it. A comment that does that is correct however long it is.

## Scope first

On a diff, audit the changed hunks and the declarations they touch. On a sweep, split the tree by directory and fan out: one agent cannot hold 3,000 comment lines and still judge them, and a single overloaded pass returns confident nonsense. The 2026-08 sweep used two agents for `include/` and `src/`; each spawned its own batches.

Exclude any file carrying uncommitted work from the fix pass. Audit it, report it, but a file with unstaged hunks cannot take a committed edit without dragging them in.

## The six checks

Judge every comment against exactly these. Anything else is a style opinion, not a finding.

**V1 restates-code.** Says what the next line plainly does. Delete. Expect zero in a healthy tree; the 2026-08 sweep found none in `include/`, which is the check's real value: it is the guard, not the yield.

**V2 cross-file-claim.** Asserts something you would open other files to verify. Read `docs/invariants.src.md` first and say whether the claim is ALREADY registered (shrink the comment to its local consequence plus the entry name) or NOT (candidate entry, or drop the outward claim). Never leave a bare pointer: a comment that only says "see the invariants" makes the reader leave the file, which is the cost colocation exists to avoid.

**V3 ref.** Contains a PR, issue, or decision reference. CLAUDE.md bans these outright. Cut the citation, keep the constraint and any measurement it carries. This reliably improves the sentence: "useful work (issue #2: 60 vCPU ran 10x slower than 16)" becomes "useful work: 60 vCPU measured 10x slower than 16", same evidence, one less hop, immune to a tracker moving.

**V4 essay-without-contract.** Multi-paragraph prose explaining history or design musing rather than what breaks. Be conservative: an essay ON a contract-bearing type is CORRECT. A check that fires on the best comments in the codebase is worse than no check.

**V5 stale.** Names a symbol, file, flag, or number that no longer matches. GREP BEFORE REPORTING and cite what you checked; an unverified staleness claim is a guess. Highest severity, because a stale comment costs more than a missing one: a reader may act on it and find out late.

**V6 missing.** A contract-bearing declaration with no comment, where misuse breaks something silently. Name the contract. Adding a comment fights "fewer comments", so the bar is silent breakage, not mere complexity.

## Output contract

One line per finding, worst first, V5 and V3 ahead of the rest:

```
<file>:<line> [V<n>] <=25 words on what is wrong. FIX: <the concrete edit>
```

Then, always:

```
TOTALS: files=N comment_lines=N V1=N V2=N V3=N V4=N V5=N V6=N
KEEP-AS-IS: <count>
UNCOVERED: <paths, or "none">
```

`KEEP-AS-IS` is not optional. A review that lists only problems cannot say whether the surface is healthy, and "42 findings" means something different against 300 comments than against 3,000. `UNCOVERED` names what a batch failed to reach, so a gap is visible rather than silent.

## Applying the findings

Never regex. The audit finds sites mechanically; every fix is a judgment about one comment, and a pattern cannot see whether a claim happens to be registered. A find-and-replace over this tree once turned 37 dead doc paths into 34 pointers at a file that did not discuss them, and once mangled a dozen comments into fragments.

Write literal old/new pairs, one per site, applied by a script that requires exactly one match and reports a skip otherwise. Then read every changed line before building: the mangling was obvious on sight and invisible to the tests.

Verify with /quality-gates. Comment-only changes must leave `scripts/model_hash.py` unmoved; drift means the edit was not comment-only.
