# Proposal (retired)

The original proposal set the scope for the first version of bonsai: a from-scratch histogram gradient boosted trees library in C++23, a CLI, and parity with the reference libraries on one regression dataset. Most of what it listed as out of scope (Python bindings, GPU training, DART, competitive speed) shipped later, so the document no longer describes the project.

The full text lives in git history. To read it:

```sh
git log --diff-filter=D --oneline -- docs/proposal.md   # find the removing commit
git show <commit>^:docs/proposal.md
```

This file stays so that the section citations in [docs/architecture/](architecture/) keep resolving. For what the project is today, start at [the documentation home](index.md); for why it changed, read [the decisions log](decisions.md).
