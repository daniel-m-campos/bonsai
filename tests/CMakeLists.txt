include(Catch)

add_executable(bonsai_tests
    unit/test_bin_mapper.cpp
    unit/test_bin_mappers.cpp
    unit/test_dataset.cpp
    unit/test_csv.cpp
    unit/test_histogram.cpp
    unit/test_dense_tree.cpp
    unit/test_oblivious_tree.cpp
    unit/test_split_node.cpp
    unit/test_split_level.cpp
    unit/test_grower.cpp
    unit/test_feature_sampling.cpp
    unit/test_leafwise_grower.cpp
    unit/test_oblivious_grower.cpp
    unit/test_sampler.cpp
    unit/test_objective.cpp
    unit/test_objective_traits.cpp
    unit/test_objective_dispatch.cpp
    unit/test_booster.cpp
    unit/test_cli_pipeline.cpp
    unit/test_eval_baseline.cpp
    unit/test_task.cpp
    unit/test_metric.cpp
    unit/test_typelist.cpp
    unit/test_make_booster.cpp
    unit/test_toml.cpp
    unit/test_model_io.cpp
)

target_link_libraries(bonsai_tests PRIVATE
    bonsai_data
    bonsai_io
    bonsai_tree
    bonsai_split
    bonsai_objective
    bonsai_sampler
    bonsai_booster
    bonsai_registry
    bonsai_config
    bonsai_cli
    bonsai_warnings
    Catch2::Catch2WithMain
)

target_include_directories(bonsai_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_definitions(bonsai_tests PRIVATE
    BONSAI_TESTS_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/data"
)

catch_discover_tests(bonsai_tests)
