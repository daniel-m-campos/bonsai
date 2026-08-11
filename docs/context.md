# Context briefing (retired)

The context briefing was the early handoff note for anyone picking the project up: goals, deferred work, and the state of the plan around v0.4.0. Its counts and status lines were overtaken long ago (it reports 36 decisions against the 100 plus on record today), so it is no longer a usable briefing.

The full text lives in git history. To read it:

```sh
git log --diff-filter=D --oneline -- docs/context.md   # find the removing commit
git show <commit>^:docs/context.md
```

This file stays so that the section citation in [docs/architecture/1-dataset.md](architecture/1-dataset.md) keeps resolving. For the current orientation, start at [the documentation home](index.md).
