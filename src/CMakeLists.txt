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
