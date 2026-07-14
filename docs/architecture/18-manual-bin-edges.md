# 18. Manual bin edges: the design (specified, admission-gated)

> Status: **priced, not built**. Decision 67 declined automatic per-feature bin budgets on accuracy evidence; this doc specifies the remaining capability, explicit user-supplied edges, and the gate that admits it: a concrete workload that needs domain-mandated bins inside the model artifact.

## The problem the emulation cannot solve

Pre-discretizing features to bin ids reproduces any binning scheme bit-exactly at train time (the issue-#61 one-cut-per-distinct rule; verified in `scripts/probe_binning.py`). But the binning then lives outside the model: every serving path must re-apply the transform, and `model.predict(raw_X)` silently mis-predicts. The point of a native API is that edges become part of the model artifact, exactly like fitted quantile cuts are today.

## API

```
ds = bonsai.Dataset(X, y, bin_edges={0: edges_a, 7: edges_b})   # others: fitted
model = bonsai.train(params, ds)
model.predict(raw_X)          # edges travel inside the model's mappers
model.save("m.msgpack")       # and serialize with it, no format change
```

`bin_edges` maps column index to a strictly increasing 1-D float array of interior cut points (k edges = k + 1 bins). Columns not listed keep quantile fitting. Validation: monotone, finite, non-empty; a column with more than 256 bins flips the dataset to u16 storage (existing rule, documented cost).

## Why it is small

The engine already represents a fitted feature as an explicit cut vector inside `BinMapper`, and `BinMappers` serialize per-feature cuts into the model envelope (models round-trip today). The change is a constructor, not a mechanism:

1. `BinMapper::from_cuts(std::vector<float>)` beside the fitting path (validation + the same missing/+inf sentinel handling).
2. `BinMappers::fit(...)` gains an optional per-column override map; overridden columns skip sampling and fitting entirely.
3. `module.cpp Dataset` gains the `bin_edges` argument (dict of int to float array) and threads it through; the sealed-config guard extends to it.
4. No model-format bump (cuts already serialize), no hot-path branches (binning is ingest-time), no CUDA change (the device plane consumes mappers as-is).

Estimate: 100 to 150 lines core-adjacent plus tests; the predict/save/load round-trip on raw values is the acceptance test that the emulation structurally cannot pass.

## The admission gate

Build when a workload arrives that needs edges in the artifact: regulatory score bands, clinical thresholds, reproducing an incumbent system's scheme for A/B parity. Accuracy is not the bar (decision 67 measured that at saturation it is a wash); the bar is the artifact story plus one user who needs it.
