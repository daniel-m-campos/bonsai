add_executable(bonsai_bench bench_split.cpp)

target_link_libraries(bonsai_bench PRIVATE
    bonsai_split
    bonsai_warnings
    Catch2::Catch2WithMain
)

target_include_directories(bonsai_bench PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)
