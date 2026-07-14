# The method

bonsai's results come from a small set of working rules, applied without exception, and the rules transfer to any performance-critical numerical system. This section states each one, shows the bonsai episode that earned it, and notes what it looks like outside a GBT library.

The common thread: measurement replaces argument. A hypothesis is priced before it is implemented, a change ships with the measurement that justifies it, and a refutation is written down with the same care as a win, because the refutation is what stops the next person from paying for the same idea twice.

The four disciplines:

1. **[Instrument first](instrument-first.md)**: decompose and price before optimizing anything; profilers lie at sync boundaries.
2. **[The feature-admission gate](feature-admission.md)**: prototype at zero core cost, define the kill criteria before the experiment, record the declines.
3. **[Benchmarks you can trust](benchmarking.md)**: same-hardware comparisons only, probe the hardware before believing it, commit the raw runs. The normative rules (divisions, suites, metrics, timing modes, the row schema) are the [benchmark protocol](benchmark-protocol.md).
4. **[Determinism as a contract](determinism.md)**: bit-identical outputs as a CI-enforced invariant, which turns refactoring and AI-assisted development into verifiable operations.

The raw feed behind all four is the [decisions log](../decisions.md): numbered, dated, with the rejected alternatives recorded next to the adopted ones. [How bonsai is built](../about.md) explains why the discipline matters twice over in a human-plus-AI workflow.
