# Benchmarks you can trust

The rules: comparative numbers come only from runs on the same machine in the same session; rented hardware is probed before it is believed; the raw results are committed to the repository; and every public claim links the run and the decision that recorded it, losses included.

The rules exist because cloud hardware is not a controlled instrument, and because a benchmark without its raw data is an anecdote.

## Same pod or it does not count

Identical-model GPUs across the rental fleet measure up to ~25% apart on the same workload. Any cross-machine comparison is noise at exactly the margins that decide a "faster than" claim, so bonsai's performance tables only ever compare variants run on one machine in one session, and the [README table](https://github.com/daniel-m-campos/bonsai/blob/main/README.md#performance) says so above the numbers.

## Probe the hardware before believing it

One rented host produced a mysterious 11 to 14 second overhead on every GPU fit. Diagnosis: its GPU sync round-trip was ~300µs against a healthy host's 4µs, a host-level misconfiguration invisible from inside the container, and bonsai's tens of thousands of synchronizing operations per fit reproduced the excess exactly.

The protocol consequence: every rented machine must pass a 30-second sync-latency probe before any of its numbers are trusted (round-trips over 50µs reject the pod), and the affected rows in the committed data are annotated as defective-host measurements rather than silently dropped ([decision 48](../decisions.md)).

## Controlled comparisons find your own bugs

The payoff of controlled benchmarking is not marketing, it is diagnosis. Two apparent catboost advantages dissolved under it: an accuracy gap at scale that localized to a bonsai GPU kernel bug precisely because everything else was held equal ([decision 63](../decisions.md)), and a binning-cost gap that turned out to be a per-feature sampling pass catboost simply does not pay, fixed for a 24× mapper speedup ([decision 64](../decisions.md)).

A comparison loose enough to flatter you is also too loose to debug you.

## Transferring it

For anyone benchmarking on cloud or rented hardware: same-machine-same-session as a hard rule, a cheap hardware sanity probe as a gate, raw results in version control, and claims that cite runs. The discipline costs one script and a habit; its absence costs arguments that nobody can settle.
