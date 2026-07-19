# The engine track

This track is how a ~1,800-line engine became the fastest route to every measured accuracy at 16M rows on one GPU. It teaches that through the episodes that did it.

## What this is

The algorithm track teaches how gradient boosting works. This track teaches how bonsai got fast. It is HPC pedagogy told case-method, like a college class. Each chapter is one real engineering episode from the project record, built up slowly, with the discipline as the through-line. The losses and the refutations are the curriculum, not the footnotes.

## Who it is for

A developer who knows some C++ and has never priced a GPU kernel. You do not need CUDA experience. You need to be willing to trust a measurement over an intuition, which is the entire method.

## The method in brief

Four rules recur across every case.

- **Instrument first.** No optimization begins until instrumentation has decomposed and priced the cost it attacks.
- **Price before building.** An edge move states its model price from measured constants before anyone writes a kernel.
- **Same-pod discipline.** Two identical GPUs measured 25% apart across the fleet, so every delta is a same-pod delta.
- **Refutation is a deliverable.** A measured no, with the conditions that would reopen it, is worth as much as a win.

## The chapters

- **E1. [The marginal round](1-the-marginal-round.md).** Instrumentation cancels a kernel rewrite, then a price list cuts the 16M round from 155 to 104 ms.
- **E2. [The missing bin](2-the-missing-bin.md).** An acceptance test fails, and the fix closes a train/predict skew hiding in every fitted model.
- **E3. [The parity verdict](3-the-parity-verdict.md).** A data-parallel multi-GPU engine, built and measured to parity, then parked as an experiment.
- **E4. [The resident objective](4-the-resident-objective.md).** Deleting the per-tree host round-trip, and the round falls again from 104 to 64 ms.
- **E5. [The ceiling](5-the-ceiling.md).** 500M rows by 100 features trained end to end on one 80GB card.
