# OBJECT libraries per component family. Avoids static-archive dead-code
# stripping; lets tests link against component pieces without rebuilding.
# See docs/decisions.md for rationale.

# ---- bonsai_data: Dataset, BinMapper, BinMappers ----------------------------

add_library(bonsai_data OBJECT
    bin_mapper.cpp
    bin_mappers.cpp
    bin_store.cpp
    dataset.cpp
)

target_include_directories(bonsai_data PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_include_directories(bonsai_data PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(bonsai_data PUBLIC bonsai_parallel bonsai_warnings)

# ---- bonsai_tree: DenseTree, ObliviousTree, growers (later) -----------------

add_library(bonsai_tree OBJECT
    tree.cpp
    grower.cpp
    shap.cpp
    shap_paths.cpp
    cuda/grower_cuda.cpp
    metal/grower_metal.cpp
)

# The CUDA grower instantiation above is plain C++ against the pimpl'd
# builder header; only the backend behind it varies. BONSAI_CUDA=ON links
# the real kernels via bonsai_gpu, OFF compiles the throwing stub.
if(NOT BONSAI_CUDA)
    target_sources(bonsai_tree PRIVATE
        cuda/histogram_engine_stub.cpp
        cuda/predict_stub.cpp
        cuda/shap_stub.cpp
    )
endif()

# Same pattern for Metal: the grower instantiation compiles everywhere, the
# engine behind it is a plain C++ TU against Apple's metal-cpp bindings on
# BONSAI_METAL builds and the throwing stub otherwise. SYSTEM keeps the
# warnings target out of Apple's headers.
if(BONSAI_METAL)
    target_sources(bonsai_tree PRIVATE metal/histogram_engine.cpp)
    target_include_directories(bonsai_tree SYSTEM PRIVATE ${metal_cpp_SOURCE_DIR})
    target_link_libraries(bonsai_tree PUBLIC
        "-framework Metal" "-framework Foundation")
else()
    target_sources(bonsai_tree PRIVATE metal/histogram_engine_stub.cpp)
endif()

target_include_directories(bonsai_tree PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
# grower.cpp and cuda/grower_cuda.cpp share the private grower_impl.hpp; src/
# on the path lets both include it as "grower_impl.hpp".
target_include_directories(bonsai_tree PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(bonsai_tree PUBLIC bonsai_data bonsai_split bonsai_parallel
                                         bonsai_gpu bonsai_warnings)

# ---- bonsai_split: SplitCandidate, SplitFinder concept, HistogramSplitFinder

add_library(bonsai_split OBJECT
    split.cpp
    monotone.cpp
)

target_include_directories(bonsai_split PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(bonsai_split PUBLIC bonsai_data bonsai_parallel bonsai_warnings)

# ---- bonsai_objective: Objective concept, MSEObjective, LogLossObjective ----

add_library(bonsai_objective OBJECT
    objective.cpp
    objective_traits.cpp
    metric.cpp
)

target_include_directories(bonsai_objective PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(bonsai_objective PUBLIC bonsai_parallel bonsai_warnings)

# ---- bonsai_sampler: Sampler concept, AllRowsSampler ------------------------

add_library(bonsai_sampler OBJECT
    sampler.cpp
)

target_include_directories(bonsai_sampler PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(bonsai_sampler PUBLIC bonsai_warnings)

# ---- bonsai_booster: header-only Booster<O,G,Sa> training loop --------------
#
# Booster is a class template whose method bodies live in booster.hpp, so this
# target only aggregates include paths + transitive deps. Consumers
# (bonsai_registry, bonsai_io, the bonsai exe, bonsai_tests) link against it
# the same way regardless.

add_library(bonsai_booster INTERFACE)

target_include_directories(bonsai_booster INTERFACE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(bonsai_booster INTERFACE
    bonsai_data
    bonsai_tree
    bonsai_split
    bonsai_objective
    bonsai_sampler
    bonsai_parallel
    bonsai_warnings
)

# ---- bonsai_registry: typelist + flat dispatch table -> IBooster ------------

add_library(bonsai_registry OBJECT
    registry/make_booster.cpp
    registry/objective_dispatch.cpp
)

target_include_directories(bonsai_registry PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(bonsai_registry PUBLIC
    bonsai_booster
    bonsai_objective
    bonsai_tree
    bonsai_split
    bonsai_sampler
    bonsai_warnings
)

# ---- bonsai_config: TOML loader + dotted-key overrides ----------------------

add_library(bonsai_config OBJECT
    config/toml.cpp
    config/toml_dump.cpp
)

target_include_directories(bonsai_config PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(bonsai_config PUBLIC
    bonsai_warnings
    tomlplusplus::tomlplusplus
)

# ---- bonsai_io: CSV (and later parquet, libsvm) + model save/load -----------

add_library(bonsai_io OBJECT
    io/csv.cpp
    io/model.cpp
)

target_include_directories(bonsai_io PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(bonsai_io PUBLIC
    bonsai_data
    bonsai_booster
    bonsai_registry
    bonsai_parallel
    bonsai_warnings
    nlohmann_json::nlohmann_json
)

# ---- bonsai_cli: subcommand handlers + shared helpers -----------------------

add_library(bonsai_cli OBJECT
    cli/common.cpp
    cli/pipeline.cpp
    cli/info.cpp
    cli/params.cpp
    cli/fit.cpp
    cli/predict.cpp
    cli/eval.cpp
    cli/bench.cpp
    cli/importance.cpp
    cli/dump.cpp
)

target_include_directories(bonsai_cli PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_include_directories(bonsai_cli PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# The CLI talks only to IBooster, make_booster, the by-name objective
# dispatchers, BinMappers/Dataset, io::*, and Config. Concrete objective /
# grower / sampler / split / booster pieces come in transitively via
# bonsai_registry and bonsai_io.
target_link_libraries(bonsai_cli PUBLIC
    bonsai_data
    bonsai_io
    bonsai_registry
    bonsai_config
    bonsai_parallel
    bonsai_warnings
)

# ---- bonsai (CLI executable) ------------------------------------------------

add_executable(bonsai cli/main.cpp)

target_link_libraries(bonsai PRIVATE
    bonsai_cli
    bonsai_data
    bonsai_io
    bonsai_booster
    bonsai_tree
    bonsai_split
    bonsai_objective
    bonsai_sampler
    bonsai_registry
    bonsai_config
    bonsai_warnings
    CLI11::CLI11
    nlohmann_json::nlohmann_json
)
