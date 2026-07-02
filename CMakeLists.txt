cmake_minimum_required(VERSION 3.25)
project(bonsai LANGUAGES CXX)

# ---- Standards / warnings ---------------------------------------------------

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

add_library(bonsai_warnings INTERFACE)
target_compile_options(bonsai_warnings INTERFACE
    -Wall -Wextra -Wpedantic -Werror
)

# When building with Homebrew LLVM clang on macOS, the SDK's libc++ may
# lag the compiler's libc++ headers (e.g. __hash_memory only in newer
# libc++). Point the linker at LLVM's libc++ if present so symbols match.
if(APPLE AND CMAKE_CXX_COMPILER MATCHES "/opt/homebrew/opt/llvm/")
    add_link_options(
        "-L/opt/homebrew/opt/llvm/lib/c++"
        "-Wl,-rpath,/opt/homebrew/opt/llvm/lib/c++"
    )
endif()

# ---- Options ----------------------------------------------------------------

option(BONSAI_TESTS "Build tests" ON)
option(BONSAI_OPENMP "Enable OpenMP parallelism" ON)
option(BONSAI_PYTHON "Build the Python extension module" OFF)

# Shared-module builds (the Python extension) need PIC everywhere; the cost
# on the static targets is negligible.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---- Parallelism ------------------------------------------------------------
# Interface target consumed by every component that includes
# bonsai/parallel.hpp. Without OpenMP the wrappers degrade to serial loops.

option(BONSAI_OPENMP_STATIC
       "Link libomp statically (required for the Python module: a private \
runtime avoids deadlocks with xgboost/lightgbm's bundled libomp)" OFF)

add_library(bonsai_parallel INTERFACE)
if(BONSAI_OPENMP)
    find_package(OpenMP COMPONENTS CXX)
    if(OpenMP_CXX_FOUND)
        target_compile_definitions(bonsai_parallel INTERFACE BONSAI_USE_OPENMP)
        find_library(BONSAI_LIBOMP_A libomp.a
                     HINTS /opt/homebrew/opt/libomp/lib /usr/local/opt/libomp/lib)
        if(BONSAI_OPENMP_STATIC AND BONSAI_LIBOMP_A)
            target_compile_options(bonsai_parallel INTERFACE ${OpenMP_CXX_FLAGS})
            target_link_libraries(bonsai_parallel INTERFACE ${BONSAI_LIBOMP_A})
            message(STATUS "OpenMP: static ${BONSAI_LIBOMP_A}")
        else()
            if(BONSAI_OPENMP_STATIC)
                message(WARNING "BONSAI_OPENMP_STATIC set but libomp.a not found; "
                                "falling back to the dynamic runtime")
            endif()
            target_link_libraries(bonsai_parallel INTERFACE OpenMP::OpenMP_CXX)
        endif()
    else()
        message(STATUS "OpenMP not found; bonsai builds serial")
    endif()
endif()

# ---- Dependencies via FetchContent ------------------------------------------

include(FetchContent)

if(BONSAI_TESTS)
    FetchContent_Declare(Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.5.4
    )
    FetchContent_MakeAvailable(Catch2)
endif()

# CLI parsing, header-only.
FetchContent_Declare(CLI11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG        v2.6.2
)

# TOML config, header-only.
FetchContent_Declare(tomlplusplus
    GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
    GIT_TAG        v3.4.0
)

# nlohmann/json — single-header, MIT, used for MessagePack model format
# via to_msgpack / from_msgpack. No transitive deps.
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)

FetchContent_MakeAvailable(CLI11 tomlplusplus nlohmann_json)

# ---- Subdirectories ---------------------------------------------------------

add_subdirectory(src)

if(BONSAI_TESTS)
    enable_testing()
    add_subdirectory(tests)
    add_subdirectory(benchmarks)
endif()

# ---- Python bindings --------------------------------------------------------

if(BONSAI_PYTHON)
    find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE nanobind_ROOT
        COMMAND_ERROR_IS_FATAL ANY)
    find_package(nanobind CONFIG REQUIRED)

    nanobind_add_module(_bonsai src/python/module.cpp)
    if(APPLE)
        # Export only the module entry point: statically linked libomp
        # symbols must not leak and interpose into other extensions'
        # OpenMP runtimes (xgboost/lightgbm bundle their own).
        target_link_options(_bonsai PRIVATE
                            "-Wl,-exported_symbol,_PyInit__bonsai")
    endif()
    target_link_libraries(_bonsai PRIVATE
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
        bonsai_parallel
        nlohmann_json::nlohmann_json
    )
    target_include_directories(_bonsai PRIVATE ${CMAKE_SOURCE_DIR}/include)

    # Dev layout: build/python/bonsai is an importable package
    # (PYTHONPATH=build/python). Wheels install via the same target.
    set_target_properties(_bonsai PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python/bonsai)
    add_custom_command(TARGET _bonsai POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_SOURCE_DIR}/python/bonsai/__init__.py
            ${CMAKE_BINARY_DIR}/python/bonsai/__init__.py)
    install(TARGETS _bonsai LIBRARY DESTINATION bonsai)
endif()
