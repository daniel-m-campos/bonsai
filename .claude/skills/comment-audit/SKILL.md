---
name: comment-audit
description: >
  Audit code comments against the CLAUDE.md comment policy: a comment only
  where no name, type, or test could carry the information, so most findings
  are extractions rather than deletions. Also catches PR/issue/decision
  references and cross-file claims that belong in docs/invariants.md. Use on a
  diff during review, or on a directory for a sweep. Invoke via /comment-audit.
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

**V4 refactorable.** The comment explains how the code below it works. That information belongs in the code, so the finding is an extraction, not a deletion: name the function to pull out and what to call it, and the comment goes with it. Run this one hardest. It is where reduction actually lives, and a comment that survives it is one whose content no name, type, or test could have carried. The 2026-08 sweep found almost nothing because this check was written to be conservative, so it never fired, and the audit spent itself on categories that were already clean.

Two guards on V4, both narrow. A contract on a contract-bearing type is not refactorable prose, it is the contract; leave it. And an extraction inside a fill or kernel inner loop is a performance change until the ledger says otherwise, so propose it and let /perf-round price it, never apply it blind.

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

Write literal old/new pairs, one per site, applied by a script that requires exactly one match and reports a skip otherwise.

Then read the resulting SENTENCE, not the changed line. This is not the same rule and the difference matters: a diff shows the added line beside the line it replaced, never beside the unchanged line it now has to join, so damage that lands across a line boundary is exactly what the diff hides. Removing a citation removes whatever punctuation it carried, and two of the 2026-08 sweep's 29 strips left run-ons that a careful diff read passed over ("Device placement first cudaSetDevice is thread-local"). Print the edited comment whole, or grep it back with context, and read it as prose. The tests cannot see any of this.

Verify with /quality-gates. Comment-only changes must leave `scripts/model_hash.py` unmoved; drift means the edit was not comment-only.
