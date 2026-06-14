# 1. Dataset

> **Status:** Phase 1 design, ratified in [`../decisions.md`](../decisions.md) entries 1–6.

## Why bin

Histogram GBT scans bins, not raw values. `float32` cell → small integer bin index. Split-finding goes from `O(n_rows)` candidates per feature to `O(max_bin)`. `uint8` is 4× smaller than `float32` — memory and cache wins are the side benefit.

Train/val/test must use the **same** bin boundaries; otherwise "feature 7 < bin 3" means different things across splits and the model is broken on eval. `BinMappers` is fit once on training data, reused everywhere.

## Pipeline

```
   train ──► read_csv ──► BinMappers::fit ──► BinMappers
                                                                   │
                                                                   ▼
   train ──► read_csv ──► Dataset::bin ───► Dataset (train)
   val   ──► read_csv ──► Dataset::bin ───► Dataset (val)
   test  ──► read_csv ──► Dataset::bin ───► Dataset (test)
```


Two calls on the train path. File loading is once-per-program; not hot.

## `ColumnBatch`

Internal helper. The shape both readers and the binner consume.

```cpp
namespace bonsai::detail {

struct ColumnBatch {
    std::vector<std::vector<float>> features;       // [n_features][n_rows]
    std::vector<float>              labels;          // [n_rows]
    std::vector<float>              weights;         // empty if uniform
    std::vector<std::string>        feature_names;   // [n_features]
};

}
```

Plain aggregate: raw `float` columns, no binning, no invariants beyond "all features have `n_rows` rows, labels has `n_rows`, weights is empty or has `n_rows`." Lives in an internal header. Not part of the public API.

Why it exists: `BinMappers::fit` and `Dataset::bin` both need raw column data. The reader produces a `ColumnBatch`; the consumers take it as input. Without this shared intermediate, you'd either parse the file twice or collapse `fit` and `bin` into one factory (rejected as decision 2).

CLI default: parse twice on the train path (cheap for California Housing). If that ever matters — once larger datasets like YearPredictionMSD enter the perf benchmark — the CLI can call `csv::parse` directly once and pass the same `ColumnBatch` to both `fit` and `bin`.

## `BinMapper`

One per feature.

```cpp
class BinMapper {
public:
    static BinMapper fit(std::span<float const> column,
                         BinMapperConfig const& cfg);

    uint16_t transform(float x) const;
    size_t   n_bins() const;
    std::span<float const> cuts() const;
    float min() const;
    float max() const;

private:
    std::vector<float> cuts_;
    uint16_t           n_bins_;
    bool               has_missing_bin_;
    float              min_value_, max_value_;
};
```

### Fit (decision 1)

- **Quantile cuts** by stride over the sorted-by-`nth_element` subsample, budgeted at `max_bin - 2` cuts (one slot reserved for the `+inf` sentinel, one for the missing bin). Default `max_bin = 255`.
- **Low-cardinality fallback** falls out naturally: duplicate cuts are dropped during stride extraction, so columns with `< max_bin - 2` distinct values produce fewer real cuts and a smaller `n_bins`.
- **`+inf` sentinel** is appended as the final cut. `n_bins()` = `cuts_.size()`.
- **NaN short-circuits to bin `n_bins - 1`** (the last bin = missing). Quantile sampling skips NaNs.
- **Sampling**: `n_samples` rows reservoir-sampled with fixed seed by default, configurable. Full column if smaller.

Rejected: equal-width (skew kills it); xgb GK sketch (overkill); xgb-style per-node default direction for missing (complicates split scoring without parity benefit).

### `transform`

```cpp
uint16_t BinMapper::transform(float x) const {
    if (std::isnan(x)) return n_bins() - 1;             // missing → last bin
    return std::ranges::lower_bound(cuts_, x) - cuts_.begin();
}
```

Right-edge cuts: bin `b` holds values in `[cuts_[b-1], cuts_[b])` (half-open), with bin 0 holding values `< cuts_[0]`. The final cut is `+inf`, so any finite real value lands in a real bin (`0 .. n_bins - 2`); NaN short-circuits to the missing bin (`n_bins - 1`).

Returns `uint16_t` regardless of caller's storage width; storage type is decided per-feature based on `n_bins()`.

## `BinMappers`

```cpp
class BinMappers {
public:
    static BinMappers fit(ColumnBatch const&, Config const&);

    BinMapper const& operator[](size_t fid) const;
    size_t size() const;
    std::span<std::string const> feature_names() const;

private:
    std::vector<BinMapper>   mappers_;
    std::vector<std::string> feature_names_;
};
```

Built once on train, immutable, round-trips with the model file. `Dataset` holds it **by value** (decision 4): ~30KB, no shared state.

Two-stage `fit` then `bin` is decision 2. Rejected lgbm-style `Dataset::from_csv(..., reference=train_ds)`: couples mapper lifetime to a Dataset, awkward serialization.

## `Dataset`

```cpp
class Dataset {
public:
    static Dataset bin(ColumnBatch const&, BinMappers const&,
                       DataConfig const&);

    size_t n_rows() const;
    size_t n_features() const;
    floats_view               labels()  const;
    floats_view               weights() const;       // empty if uniform
    BinMappers const&         mappers() const;
    size_t                    n_bins(size_t fid) const;
    bool                      is_categorical(size_t fid) const;
    std::span<bin_id_t const> feature_bins(size_t fid) const;
private:
    std::vector<std::vector<bin_id_t>> features_;
    std::vector<float>                 labels_, weights_;
    BinMappers                         mappers_;
    std::vector<bool>                  is_categorical_;
    size_t                             n_rows_;
};
```

### Layout (decision 4)

- **Column-major.** Row-major destroys cache on histogram scans.
- **Uniform `bin_id_t` storage** for all binned columns. Doesn't matter if a feature has 8 bins or 250 — same width.
- Group columns (ranking) are non-goals.

Rejected: `std::variant<vector<uint8_t>, vector<uint16_t>>` per feature to halve memory on small-bin features. Forecast savings on planned perf datasets: ~45MB on YearPredictionMSD, ~308MB on Higgs — neither pressure-tests modern hardware. Cost was variant dispatch (a `visit_column` wrapper at every column scan). Rejected for MVP. Reversible if memory becomes the bottleneck on a future dataset.

### Histogram inner loop

```cpp
auto bins = ds.feature_bins(fid);  // span<bin_id_t const>
for (size_t i = 0; i < bins.size(); ++i)
    hist[bins[i]] += grad[i];
```

Direct array indexing. No variant, no dispatch.

## Trees store float thresholds (decision 3)

Tree node split = `(feature_id, threshold: float)`. Predict reads raw `float`, no re-binning. `TreeGrower` finds best split as `(fid, bin_idx)`, converts to `threshold = cuts[bin_idx]` when writing the node — one lookup per finalized split.

Knock-on: predict path doesn't need `BinMappers`. Single tree walk over raw floats. `BinMappers` in the model file is for diagnostics + reproducibility, not load-bearing for predict.

## Readers (decision 6)

```cpp
Dataset    read_csv     (std::string const& path, DataConfig const&,
                         BinMappers const&);
Dataset    read_parquet (...);   // Phase 4+
Dataset    read_libsvm  (...);   // later

BinMappers fit_from_csv(std::string const& path, Config const&);
```

CLI dispatches on `cfg.data.format`. Each reader its own translation unit. `read_X` and `fit_from_X` share parsing through an internal `namespace bonsai::csv { ColumnBatch parse(...); }` helper.

**Phase 1: hand-rolled CSV** (~50 LOC). Numeric only, comma-separated, header row, configurable label column. Enough for California Housing (integration) and the planned YearPredictionMSD perf benchmark.

**Phase 4+: Arrow optional** behind `BONSAI_PARQUET=ON` CMake flag. Arrow handles CSV + parquet + feather. Heavier dep (transitive thrift/snappy); kept optional. Arrow is the reader layer; `Dataset` is bonsai's binned-storage layer.

Rejected: `Reader` concept + template `Reader auto&` params. Over-engineered for once-per-program file loading.

## Serialization

I/O lives outside the domain types. `BinMapper`, `BinMappers`, `Dataset`, `Tree` etc. are pure data + their conceptual operations (fit/transform/bin). Reading and writing model files is a separate concern in `bonsai::io`:

```cpp
namespace bonsai::io {
    void write_binary(std::ostream&, BinMapper const&);
    void write_binary(std::ostream&, BinMappers const&);

    BinMapper  read_bin_mapper_binary(std::istream&);
    BinMappers read_bin_mappers_binary(std::istream&);
    // similar for Tree, Booster, etc.
}
```

Free functions, format suffix in the name. Adding JSON or another binary layout = new TU in `io/`, no domain change. Keeps domain types free of stream dependencies; testable without I/O machinery.

Rejected: visitor pattern (overkill in modern C++; virtual dispatch and interface churn for cold I/O); cereal-style `template<Archive> serialize` on the domain class (still couples persistence to the type, adds a dep).

Note: `BinMapper::fit` and `Dataset::bin` are static factory methods (named-constructor idiom). The verbs read naturally as `Type::verb(...)` and the construction is non-trivial work that benefits from a name. `read_csv`, `fit_from_csv` are free functions because their identity is "produce X from external context," not "construct X."

## What's not here

- `Histogram` shape, subtraction trick, parallel reduction → `2-histogram.md`.
- `Tree` / `Node` representation → `3-tree.md`.
- `Booster<...>` layout, training loop → `5-booster.md`.
- Deferred dispatch mechanism → `6-dispatch.md`.
- Categoricals (Phase 4). The `is_categorical_` flag is a placeholder; storage layout doesn't change, split-finding will dispatch on it.

## Cross-references

- [`../decisions.md`](../decisions.md) entries 1–6 are source of truth.
- [`../proposal.md` §3.1–3.3](../proposal.md) for performance-sensitive surfaces.
- [`../context.md` §4](../context.md) for the original briefing-level summary.
