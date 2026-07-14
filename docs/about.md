# How bonsai is built

bonsai is developed by a human maintainer working with Anthropic's Claude models — the recent phases with Claude Fable 5, earlier ones with Claude Opus — and the collaboration is part of the project's story rather than a footnote.

The division of labor: the maintainer sets direction, scope, and taste (what to build, what to decline, when a result justifies ambition), reviews and merges every change, and owns the releases. Claude drives implementation campaigns end to end — writing the C++ and CUDA, designing and running benchmark sweeps on rented GPU hardware, filing the decisions log, and reviewing its own merged work adversarially (several correctness bugs on this site's [decisions log](decisions.md) were found by post-hoc multi-agent review of AI-written code, then fixed the same day).

What makes AI-driven implementation trustworthy here is not trust — it is the verification discipline described in the Method section. Models are bit-identical across CPU architectures and thread counts, enforced per-commit in CI, so a refactor cannot silently change results. Performance claims come from same-pod benchmark runs committed to the repository. Every feature passes an admission gate with pre-registered kill criteria, and refuted hypotheses are recorded alongside adopted ones.

The transparency is auditable, not asserted: every commit in the repository carries a `Claude-Session` trailer linking the session that produced it.

The scope escalation this enabled is the project's headline: what began as a first-principles learning exercise reached CPU parity with the reference libraries, then GPU parity, then GPU leads, over a compressed calendar — with the codebase held small and readable throughout, because readability was the original point and remained a hard constraint.
