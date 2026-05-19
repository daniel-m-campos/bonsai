# 8. Config

> **Status:** Mostly done. `DataConfig`, `BinMapperConfig`, `BoosterConfig`,
> `TreeConfig`, `DispatchConfig` are pinned and parsed. `ParallelConfig`,
> `IOConfig` to be filled in as their components are designed (post-spine).

## Shape

Strongly-typed nested structs in C++; TOML on disk; CLI overrides via
dotted keys. Resolution order: struct defaults → TOML file → CLI flags.
Last write wins. Strict parsing — unknown TOML keys are an error.

```cpp
namespace bonsai {

struct Config {
    DataConfig      data;
    BinMapperConfig bin_mapper;
    TreeConfig      tree_config;
    BoosterConfig   booster_config;
    DispatchConfig  dispatch;
    // ParallelConfig, IOConfig — TBD
};

}
```

Component-specific params live with the component (`top_rate` is in
`SamplerConfig`, not global). Defaults at struct level, not in the
parser.

Validation in the consumer's constructor / factory. `BinMapper::fit`
throws `ConfigError` on bad `BinMapperConfig` input; same for any other
component. No central validator.

## `DataConfig`

```cpp
struct DataConfig {
    std::string train;
    std::vector<std::string> valid;          // for multi-validation
    std::string test;

    std::string format = "csv";              // csv | libsvm | parquet
    bool        header = true;
    int         label_column  = 0;           // index; name lookup deferred
    int         weight_column = -1;          // -1 = no weights
    std::vector<int> ignore_columns;

    // Missing-value handling. See semantics below.
    bool                  missing_nan      = true;
    std::optional<float>  missing_sentinel = std::nullopt;
};
```

TOML:

```toml
[data]
train = "train.csv"
valid = ["val.csv"]
test  = "test.csv"
format = "csv"
header = true
label_column = 0
weight_column = -1
ignore_columns = []
missing_nan = true
# missing_sentinel = -999.0   # optional
```

## `BinMapperConfig`

```cpp
struct BinMapperConfig {
    int  max_bin             = 255;          // <= 65535 (uint16 storage)
    int  bin_construct_sample = 200000;      // 0 = full column
    uint64_t seed             = 0;           // sampler RNG seed
    int  min_data_in_bin     = 1;            // (deferred: see decision 1)
};
```

TOML:

```toml
[bin_mapper]
max_bin = 255
bin_construct_sample = 200000
seed = 0
```

`min_data_in_bin` is in the struct (deferred per decision 1) but ignored
by `BinMapper::fit` until that knob is wired up. Default 1 = no merging.

## Missing-value semantics

The proposal sketched `missing = "nan"` and `missing = "value:<float>"`.
The struct above splits that into two fields for clarity:

- `missing_nan` (default `true`): NaN inputs route to bin 0 in
  `BinMapper::transform`, and `BinMapper::fit` skips NaNs in quantile
  computation.
- `missing_sentinel` (default `nullopt`): if set, that exact float value
  is also treated as missing — routes to bin 0, skipped in quantile.

Both can be active simultaneously. If both are off (`missing_nan = false`
and `missing_sentinel = nullopt`), no missing-value handling: NaN inputs
are undefined behavior in `transform`, and any NaN in fit poisons the
quantile sort. `BinMapper::fit` validates and throws `ConfigError` if
this case occurs.

The TOML keys are `missing_nan` and `missing_sentinel`; the proposal's
`missing = "value:-999"` form translates to:

```toml
missing_nan = true
missing_sentinel = -999.0
```

## `TreeConfig`, `BoosterConfig`, `DispatchConfig`

```cpp
struct TreeConfig {
    float   min_child_hess     = 1.0F;
    float   min_gain_to_split  = 0.0F;
    float   lambda_l2          = 1.0F;
    uint8_t max_depth          = 6;
    uint8_t min_data_in_leaf   = 20;
};

struct BoosterConfig {
    uint32_t n_iters       = 100;
    float    learning_rate = 0.05F;
    uint32_t random_seed   = 42;
};

struct DispatchConfig {
    std::string objective_name = "mse";       // mse | logloss
    std::string grower_name    = "depthwise";
    std::string sampler_name   = "all_rows";
};
```

TOML:

```toml
[tree]
max_depth = 6
min_data_in_leaf = 20
lambda_l2 = 1.0

[booster]
n_iters = 200
learning_rate = 0.05

[dispatch]
objective_name = "mse"
grower_name = "depthwise"
sampler_name = "all_rows"
```

## CLI overrides

Dotted keys via `--set section.key=value` (repeatable):

```
bonsai fit --config base.toml \
           --set tree.max_depth=8 \
           --set booster.n_iters=300 \
           --set booster.learning_rate=0.03
```

Keys mirror the underscored TOML names (e.g. `tree.max_depth`, not
`tree.max-depth`). Last write wins across multiple `--set` flags and
between the file and the CLI. Unknown keys throw `ConfigError`.

## Parsing

```cpp
namespace bonsai::config {
    Config parse_toml(std::string const& path);
    void   apply_cli_overrides(Config&, /* CLI11 result */);
}
```

`toml++` with strict mode (reject unknown keys). Each struct gets a
hand-written deserializer in `config/parse.cpp`; no reflection-based
auto-mapping. Errors throw `ConfigError` with a key path:
`"data.format: unknown value 'tsv', expected csv|parquet|libsvm"`.

## What's not here

- `BoosterConfig`, `TreeConfig`, `SamplerConfig`, `SplitConfig`,
  `ParallelConfig`, `IOConfig` — added as their components are designed.
- `MetricConfig` — same.
- TOML→struct deserializer details — pinned down when `parse_toml` is
  implemented.
- Profiles / presets — explicitly rejected (decision *reserved*; was
  ADR-005 in the original sketch, kept as a non-goal).

## Cross-references

- [`1-dataset.md`](1-dataset.md) consumes `DataConfig` and
  `BinMapperConfig`.
- [`../decisions.md`](../decisions.md) entry 1 (binning strategy) for
  the `bin_mapper` knobs.
- [`../proposal.md` §3.7](../proposal.md) for the original config
  schema sketch.
