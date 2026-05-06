# Decisions

Append-only log. Order = decision order. Caveman style. New entries at bottom.

---

## 1. Binning: quantile, with low-cardinality fallback

`BinMapper::fit` per feature.

- Equal-frequency cuts at `k/max_bin`-th quantiles. `max_bin = 255` default
  (`uint8` indices).
- If `n_distinct < max_bin`: one cut between each pair of consecutive
  distinct values. Bucket count = `n_distinct`.
- Dedupe cut collisions (sentinel values like `0.0`). Actual count
  `<= max_bin`, never exact.
- Sampling from the start. Default sample 200K rows uniform random,
  fixed seed. Configurable. If column has `<= sample_size` rows, use full
  column.
- Bin 0 reserved for missing. NaN + user-configured sentinel short-circuit
  to bin 0. Real values bins `1..n_buckets-1`. Quantile skips NaNs.
- `BinMapper` serializable. Round-trip through model file. Predict on new
  data reuses train boundaries exact.

Rejected: equal-width (skew kills it). Quantile sketch (overkill, swap
in later). xgb per-node default direction (complicates split scoring).

Knock-on: bucket count varies per feature, histogram reads `n_buckets[fid]`.
Bin 0 special, split scoring skips it for real-valued cuts. `BinMapper`
ownership vs `Dataset` is next decision.

Defer: `min_data_in_bin` knob.

---

## 2. `BinMapper` independent of `Dataset`. Two-stage API.

```
auto mappers = BinMappers::fit(train_source, cfg);
auto train   = Dataset::bin(train_source, mappers, cfg);
auto val     = Dataset::bin(val_source,   mappers, cfg);
auto test    = Dataset::bin(test_source,  mappers, cfg);
```

- `BinMappers` is `std::vector<BinMapper>` plus minimal wrapper (count,
  serialize). Built once on train, immutable thereafter.
- `Dataset::bin` is pure transform: takes source + mappers, returns
  binned column-major storage. No "training Dataset" vs "val Dataset"
  distinction.
- Model file serializes `BinMappers`. Predict-time `Dataset` builds
  fresh from them.

Rejected: lgbm-style `Dataset::from_csv(..., reference=train_ds)`. Couples
mapper lifetime to a Dataset, awkward serialization, "training Dataset"
becomes special.

Knock-on: train path is two calls instead of one. Trivial. `bin` is
single-pass; `fit` does its own sampling + sort internally.

---

## 3. Trees store raw float thresholds, not bin indices

Tree node split = `(feature_id, threshold: float)`. Predict reads raw
`float` from input row, compares directly. No binning at predict time.

`TreeGrower` finds the best split as `(fid, bin_idx)` during training, then
converts to `threshold = cuts[bin_idx]` when writing the node. Conversion
is one lookup per finalized split, free.

xgb + catboost do this. lgbm stores bin indices in tree nodes (and re-bins
at predict, which is why lgbm forces the reference-Dataset dance).

Knock-on:
- Predict path doesn't need `BinMappers`. Single tree walk over raw floats.
- Model file: trees serialize directly, `BinMappers` optional in model file
  (kept for diagnostics + reproducibility, not load-bearing for predict).
- Training-time histogram code unchanged: still bins, still works on bin
  indices internally.
- Float threshold means tree comparison is `<` on float, not `<=` on int.
  Watch for off-by-one when comparing parity vs lgbm (different convention).

---

## 4. `Dataset` storage layout

Column-major. Per-feature `std::vector<uint16_t>` (uniform width). Labels +
weights owned by `Dataset` (weights empty if uniform). `BinMappers` held by
value (not `shared_ptr`); ~30KB copy is trivial, no shared mutable state.

```cpp
class Dataset {
    std::vector<std::vector<uint16_t>> features_;
    std::vector<float>                  labels_, weights_;
    BinMappers                          mappers_;
    std::vector<bool>                   is_categorical_;  // Phase 4 placeholder
    // n_rows, n_features
};
```

Public API: `num_rows()`, `num_features()`, `labels()`, `weights()`,
`mappers()`, `n_buckets(fid)`, `is_categorical(fid)`,
`column(fid) -> span<uint16_t const>`.

Rejected: `std::variant<vector<uint8_t>, vector<uint16_t>>` per feature
to save ~50% on binned column memory. Saves ~45MB on YearPredictionMSD,
~308MB on Higgs — neither pressure-tests modern hardware. Cost was
variant dispatch complexity at every column scan via a `visit_column`
wrapper. Rejected for MVP; reversible if a future dataset makes memory
the bottleneck.

Group columns (ranking) deferred (non-goal).

---

## 5. *(reserved)*

Originally `visit_column` for variant-aware column access. Dropped when
decision 4 collapsed to uniform `uint16_t` storage. Renumbering decisions
breaks references; left as a placeholder.

---

## 6. Readers: free function per format, returning `Dataset`

```cpp
Dataset read_csv    (const std::string& path, const DataConfig&, const BinMappers&);
Dataset read_parquet(const std::string& path, const DataConfig&, const BinMappers&);  // Phase 4+
Dataset read_libsvm (const std::string& path, const DataConfig&, const BinMappers&);  // later

BinMappers fit_from_csv(const std::string& path, const Config&);
```

CLI dispatches on `cfg.data.format` string: `if "csv" call read_csv else if
"parquet" ...`. Each reader its own translation unit. Adding a new format =
new file + new branch in CLI dispatch.

No `Reader` concept, no abstract base, no template plumbing. File loading
is once-per-program, not hot-path; concepts buy nothing here. Internal
shared helper per reader (`CsvReader::columns(path, cfg) -> ColumnBatch`)
keeps `read_csv` and `fit_from_csv` from duplicating logic.

**Phase 1 ships CSV only**, hand-rolled (~50 LOC). Numeric-only is enough
for YearPredictionMSD. No Arrow, no parquet.

**Phase 4+: Arrow optional**, gated by `BONSAI_PARQUET=ON` CMake flag.
Arrow handles CSV multi-threaded + parquet + feather + IPC. Heavier dep
(transitive thrift/snappy), so kept optional. Arrow is the *reader* layer
only. `Dataset` (binned storage + `BinMappers` + labels/weights) is bonsai's;
Arrow's `Table` is raw column data we'd copy or borrow from.

Rejected: `Reader` concept + `Reader auto&` template params for `fit` and
`bin`. Over-engineered for once-per-program file loading. Free functions are
the simpler shape.

---

## 7. Determinism contract: fixed thread count, not cross-thread

Same seed + same data + **same thread count** → same model bytes.
Different thread counts: predictions within numerical tolerance, but
bytes may differ.

What this rules in:
- Per-thread local histograms (no atomic FP adds — those are bit-unstable
  even at fixed thread count).
- Deterministic chunking (e.g., OpenMP `schedule(static)`).
- `random_seed` carries through samplers / shufflers.

What this rules out (relative to earlier framing):
- Promising cross-thread bit-exactness. The earlier draft demanded
  fixed-order merge (`tid` outer, bin inner) so the per-thread reduction
  shape didn't depend on thread count. Dropped — costs design constraints
  on `ParallelBackend` (must expose ordered reduction primitive) and
  forecloses `OpenMP reduction(+:...)` and `std::execution` reduce
  shapes.

Field check: XGBoost and CatBoost don't promise cross-thread
determinism. LightGBM offers it behind `deterministic=true` +
`force_col_wise|row_wise`, and its own maintainers describe the
guarantee as fragile (RFC #6731). The pragmatic, industry-standard
contract is "thread count is part of the reproducibility input."

Test contract:
- `test_determinism_fixed_threads`: two runs at `n_threads=k` for
  `k ∈ {1, 4, 8}` produce identical model files. Required to pass.
- `test_determinism_cross_threads`: predictions across different
  thread counts agree to numerical tolerance (e.g., max abs diff
  < 1e-5 on YearPredictionMSD). Required to pass.

Knock-on:
- `ParallelBackend` does not need to expose "ordered reduction" as
  a primitive. `parallel_for` + thread-local accumulators is enough.
- The histogram's parallel-build description in
  [`architecture/2-histogram.md`](architecture/2-histogram.md) §"Parallel
  construction" reflects this — no fixed-tid-order requirement.
- Atomic FP adds remain forbidden, but for the bit-stability-at-fixed-N
  reason, not the cross-N reason.
