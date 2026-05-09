include(Catch)

add_executable(bonsai_tests
    unit/test_bin_mapper.cpp
    unit/test_bin_mappers.cpp
    unit/test_dataset.cpp
    unit/test_csv.cpp
    unit/test_histogram.cpp
    unit/test_dense_tree.cpp
    unit/test_oblivious_tree.cpp
)

target_link_libraries(bonsai_tests PRIVATE
    bonsai_data
    bonsai_io
    bonsai_tree
    bonsai_warnings
    Catch2::Catch2WithMain
)

target_include_directories(bonsai_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

catch_discover_tests(bonsai_tests)
