# OBJECT libraries per component family. Avoids static-archive dead-code
# stripping; lets tests link against component pieces without rebuilding.
# See docs/architecture/README.md and decisions.md for rationale.

# ---- bonsai_data: Dataset, BinMapper, BinMappers ----------------------------

add_library(bonsai_data OBJECT
    bin_mapper.cpp
    bin_mappers.cpp
    dataset.cpp
)

target_include_directories(bonsai_data PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(bonsai_data PUBLIC bonsai_warnings)

# ---- bonsai_tree: DenseTree, ObliviousTree, growers (later) -----------------

add_library(bonsai_tree OBJECT
    tree.cpp
    grower.cpp
)

target_include_directories(bonsai_tree PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(bonsai_tree PUBLIC bonsai_data bonsai_split bonsai_warnings)

# ---- bonsai_split: SplitCandidate, SplitFinder concept, HistogramSplitFinder

add_library(bonsai_split OBJECT
    split.cpp
)

target_include_directories(bonsai_split PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(bonsai_split PUBLIC bonsai_data bonsai_warnings)

# ---- bonsai_objective: Objective concept, MSEObjective, LogLossObjective ----

add_library(bonsai_objective OBJECT
    objective.cpp
)

target_include_directories(bonsai_objective PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(bonsai_objective PUBLIC bonsai_warnings)

# ---- bonsai_sampler: Sampler concept, AllRowsSampler ------------------------

add_library(bonsai_sampler OBJECT
    sampler.cpp
)

target_include_directories(bonsai_sampler PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(bonsai_sampler PUBLIC bonsai_warnings)

# ---- bonsai_io: CSV (and later parquet, libsvm) -----------------------------

add_library(bonsai_io OBJECT
    io/csv.cpp
)

target_link_libraries(bonsai_io PUBLIC bonsai_data bonsai_warnings)

# ---- bonsai (CLI executable) ------------------------------------------------

add_executable(bonsai cli/main.cpp)

target_link_libraries(bonsai PRIVATE
    bonsai_data
    bonsai_io
    bonsai_warnings
)
