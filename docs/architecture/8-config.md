# 8. Config

> **Status:** Done for everything shipped. `DataConfig`, `BinMapperConfig`, `BoosterConfig`, `TreeConfig`, `SamplerConfig`, `DispatchConfig`, `MetricsConfig`, `ParallelConfig` (`[parallel] n_threads`, decision 32), and `ObjectiveConfig` (`[objective] huber_delta / quantile_alpha`, decision 35) are pinned and parsed. `IOConfig` remains future.

## Shape

Strongly-typed nested structs in C++; TOML on disk; CLI overrides via dotted keys. Resolution order: struct defaults → TOML file → CLI flags. Last write wins. Strict parsing: unknown TOML keys are an error.

```cpp
namespace bonsai {

struct Config {
    DataConfig      data;
    BinMapperConfig bin_mapper;
    TreeConfig      tree_config;
    BoosterConfig   booster_config;
    DispatchConfig  dispatch;
    MetricsConfig   metrics;
    ParallelConfig  parallel;   // [parallel] n_threads (decision 32)
    ObjectiveConfig objective;  // [objective] huber_delta, quantile_alpha
    // IOConfig: TBD
};

}
```

Component-specific params live with the component (`top_rate` is in `SamplerConfig`, not global). Defaults at struct level, not in the parser.

Validation in the consumer's constructor / factory. `BinMapper::fit` throws `ConfigError` on bad `BinMapperConfig` input; same for any other component. No central validator.

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

`min_data_in_bin` is in the struct (deferred per decision 1) but ignored by `BinMapper::fit` until that knob is wired up. Default 1 = no merging.

## Missing-value semantics

The proposal sketched `missing = "nan"` and `missing = "value:<float>"`. Neither survives as a knob, because neither was ever a choice the engine could make.

NaN is the missing marker on every path: `BinMapper::fit` skips it when cutting, `BinMapper::transform` routes it to the missing bin, and every split learns a default branch for it. Nothing configures that. Turning a placeholder value like -999 into NaN is preprocessing the caller does before bonsai sees the data, one line of numpy or one pass over a file.

The CSV contract follows from the same rule: a missing value is written as the literal `nan`, and an empty field is a parse error naming its row and column. The reader cannot tell an intended gap from a truncated line, and a config knob choosing between those readings only moves the guess.

## `TreeConfig`, `BoosterConfig`, `DispatchConfig`, `MetricsConfig`

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
    uint32_t log_intervals = 0;   // 0 = silent; else
                                  // floor(n_iters/log_intervals)+1
                                  // metric ticks during fit
};

struct DispatchConfig {
    std::string objective_name = "mse";       // mse | logloss
    std::string grower_name    = "depthwise";
    std::string sampler_name   = "all_rows";
};

struct MetricsConfig {
    std::vector<std::string> fit;    // empty → objective defaults
    std::vector<std::string> eval;   // empty → objective defaults
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
# log_intervals = 10

[dispatch]
objective_name = "mse"
grower_name = "depthwise"
sampler_name = "all_rows"

[metrics]
# fit and eval lists; empty means the objective's defaults
fit = ["rmse"]
eval = ["rmse", "mae"]
```

## CLI overrides

Dotted keys via `--set section.key=value` (repeatable):

```
bonsai fit --config base.toml \
           --set tree.max_depth=8 \
           --set booster.n_iters=300 \
           --set booster.learning_rate=0.03
```

Keys mirror the underscored TOML names (e.g. `tree.max_depth`, not `tree.max-depth`). Last write wins across multiple `--set` flags and between the file and the CLI. Unknown keys throw `ConfigError`.

## Parsing

```cpp
namespace bonsai::config {
    Config parse_toml(std::string const& path);
    void   apply_cli_overrides(Config&, /* CLI11 result */);
}
```

`toml++` with strict mode (reject unknown keys). Each struct gets a hand-written deserializer in `config/parse.cpp`; no reflection-based auto-mapping. Errors throw `ConfigError` with a key path:
`"data.format: unknown value 'tsv', expected csv|parquet|libsvm"`.

The on-disk *model file* (`bonsai::io::save_booster`) serializes the same Config to JSON-in-msgpack via `NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE` macros, not through this codec; see decision 29 for the rationale (less code at the cost of duplicating the field list).

## What's not here

- `SplitConfig`, `IOConfig`: added as their components are designed (`SamplerConfig`, `ParallelConfig`, `ObjectiveConfig` since landed).
- TOML→struct deserializer details: pinned down when `parse_toml` is implemented.
- Profiles / presets: explicitly rejected (decision *reserved*; was ADR-005 in the original sketch, kept as a non-goal).

## Cross-references

- [`1-dataset.md`](1-dataset.md) consumes `DataConfig` and `BinMapperConfig`.
- [`../decisions.md`](../decisions.md) entry 1 (binning strategy) for the `bin_mapper` knobs.
- [`../proposal.md` §3.7](https://github.com/daniel-m-campos/bonsai/blob/main/docs/proposal.md) for the original config schema sketch.
