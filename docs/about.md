# How bonsai is built

bonsai is developed by a human maintainer working with Claude, and the collaboration is part of the project's story rather than a footnote: what began as a first-principles learning exercise reached CPU parity with the reference libraries, then GPU parity, then GPU leads, over a compressed calendar, with the codebase held small and readable throughout, because readability was the original point and remained a hard constraint.

What makes AI-assisted implementation trustworthy here is not trust; it is the verification discipline described in the Method section. Models are bit-identical across CPU architectures and thread counts, enforced per-commit in CI, so a refactor cannot silently change results. Performance claims come from same-pod benchmark runs committed to the repository. Every feature passes an admission gate with pre-registered kill criteria, and refuted hypotheses are recorded in the [decisions log](decisions.md) alongside adopted ones, including correctness bugs in AI-written code that were found by adversarial post-hoc review and fixed the same day.

The transparency is auditable, not asserted: every commit in the repository carries a session trailer linking the session that produced it.
