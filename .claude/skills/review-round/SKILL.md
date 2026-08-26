---
name: review-round
description: >
  bonsai's adversarial review discipline for a branch or PR: fan out
  fresh-context reviewers with the repo's conventions in the brief, verify
  every finding before applying it, and measure any change that touches a
  hot path. Use before asking for human review, or when a diff has grown
  past what one reading can hold. Invoke via /review-round.
---

The author is the worst reviewer of their own diff and the only one with the context. This method spends fresh context to buy independence, then spends the author's context verifying what comes back. Worked example: PR #416 (31 commits, +4900/-240), three reviewers, ~15 findings, **two of which were wrong in ways that would have shipped a bug**.

## Step 1: scope to the real base

`git diff origin/main...HEAD`, never `main...HEAD`. A stale local main reported 161 commits and 18k insertions for a 31-commit, 4.9k-insertion PR, which would have aimed every reviewer at the wrong diff.

Then split by language. Idioms differ, and one reviewer per idiom reads better than one reviewer per repo.

## Step 2: brief against the conventions, or lose findings to noise

A fresh reviewer suggests exactly what this repo has already rejected: added doc comments, defensive validation, `-> None`, decision numbers in comments. Put the conventions in the brief (comments-vs-docs, C++23 references over pointers, the 3.9 floor, tests are not bloat) so what comes back is about the code.

Two more brief items that pay for themselves:

- **Name the uncommitted hunks.** Probe code in the working tree is not in the diff; a reviewer reading raw files flags code that is not shipping.
- **Hand over your own suspicions, phrased "verify or refute, do not just agree".** In #416 this refuted two: `_STRATEGIES` and `RowView`'s three forms both survived, and only the prose around them was cut.

For a design review add a **design brief** listing what is deliberate (global indexing, spent views, stale out-of-view scores). Without it the reviewer rediscovers the design instead of critiquing it.

## Step 3: ponytail before design

No point grading the SOLID of code that should be deleted. Ponytail is diff-scoped and mechanical. Design review is codebase-scoped by default and **must be narrowed** to the surface the diff introduces, or it relitigates settled architecture and buries the new material.

## Step 4: verify every finding before applying it

This is the step that earns the method. Findings arrive confident and wrong at a real rate.

- **Check scope first.** `git show origin/main:<file> | grep -c <symbol>` says whether a "pre-existing" smell is new in this diff.
- **Check the premise.** For a `native:` finding claiming the library ships something, go read the library's header. In #416 nanobind really did ship `nb::value_error`, and a hand-rolled helper across 15 sites went away.
- **Run the suite before believing a correctness fix.** The top design-review recommendation, iterating the view's rows in the score update to make an invariant structural, was a **regression**: a view may repeat a row id (a with-replacement draw is why duplicates are supported), `scores_` holds one slot per row, and iterating the view advances a repeated row's prediction twice. An existing test caught it in under a minute.
- **Refute in writing, in the commit message.** Two findings were declined on reasoning (merging two footers that share two lines; a free `gather()` on doctrinal grounds). That reasoning belongs where the next reader looks.

When a finding is right but its *reason* is wrong, fix the reason too. #416's recycled-output contract claimed every element is overwritten before read, which the PR had made false. The code was fine; the comment was the defect.

## Step 5: measure anything that touches a hot path

/perf-round applies, with three rules this round proved:

- **Make the instrument resolve before trusting it.** The grow profile prints two decimals, so at `populate = 0.56s` quantization is +/-1.8% and the whole apparent effect was the last digit. Size the cell until the bucket reads in seconds.
- **Swap the code, hold the machine.** Commit, `git checkout <parent> -- <files>`, rebuild, measure, restore, rebuild, measure. Never `git stash` (the stack is shared across worktrees). The same-pod rule, applied locally.
- **Quote the min, not the mean.** Noise on a fixed workload is one-sided: it can only add time. In #416 the mean was the only statistic showing a regression, entirely from one outlier.

Test-first still applies to a refactor: pin the behavior the move must preserve *before* moving it. The mirror's multi-block layout had no coverage at all, and it is the only case where the addressing rule is non-trivial.

## Step 5b: two ways this round nearly shipped a defect through its own gates

- **Never read a gate through `tail`.** `make lint 2>&1 | tail -2` shows the last finding and hides everything above it. A whole session ran on "clean apart from the known one" without knowing whether there was anything else. Read the full output, or grep for `warning:|error:` and count.
- **`git add -u` is `git add -A` wearing a hat.** It stages every tracked modification, which in #416 swept probe hunks that thirty prior commits had kept out, and CI's tidy caught them. Stage by explicit path, always, including in a throwaway checkpoint commit.
- **Lint the tree CI lints.** Uncommitted working-tree state makes local and CI disagree in both directions. `git checkout HEAD -- <files>` (after backing the hunks up outside the repo) reproduces what the runner sees.
- **Every edit reopens every gate.** Three CI reds in one day were all the same slip: a fix applied AFTER a gate had run, committed on the strength of the stale pass. The gate battery is not a checklist to complete once; it is a postcondition of the final tree. If you touch a file after format ran, format ran on a tree that no longer exists.

The tell in both cases was a contradiction sitting in plain view: a finding described as working-tree-only, in a file the PR diff had been listing as modified all along. When two facts about the same file disagree, stop and reconcile them.

## Step 6: record

Findings applied, findings refuted and why, and any measurement go in the commit messages and the PR body. A refutation is a deliverable: it stops the next reviewer re-proposing it, and the two in #416 were the most useful output the round produced.
