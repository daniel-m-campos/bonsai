add_executable(bonsai_bench
    bench_split.cpp
    bench_sampler.cpp
    bench_tree.cpp
)

target_link_libraries(bonsai_bench PRIVATE
    bonsai_data
    bonsai_split
    bonsai_sampler
    bonsai_tree
    bonsai_warnings
    Catch2::Catch2WithMain
)

target_include_directories(bonsai_bench PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
