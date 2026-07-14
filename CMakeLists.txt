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

# Host-plane floats are contract-free: clang fuses a*b+c into a
# single-rounded fma on targets that have the instruction (arm64) and not
# on ones that don't (baseline x86-64), which made trained models
# platform-dependent (decision 59 — the 0.71719-vs-0.71725 pin split).
# With contraction off, models are bit-identical across architectures.
# The CUDA kernel TU opts back out below: the device plane has its own
# precision scheme and no cross-platform hash contract.
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>)

# When building with Homebrew LLVM clang on macOS, the SDK's libc++ may
# lag the compiler's libc++ headers (e.g. __hash_memory only in newer
# libc++). Point the linker at the KEG'S libc++ so symbols match — any keg,
# versioned or not: the old "llvm/"-only match let llvm@21 builds link
# against the stale SDK libc++ (undefined string-hash symbols on the
# macos-15 CI runner; masked locally by a newer SDK).
if(APPLE AND CMAKE_CXX_COMPILER MATCHES "(/opt/homebrew/opt/llvm[^/]*)/")
    add_link_options(
        "-L${CMAKE_MATCH_1}/lib/c++"
        "-Wl,-rpath,${CMAKE_MATCH_1}/lib/c++"
    )
endif()

# On Linux, clang defaults to the system libstdc++, which on Ubuntu 22.04
# (GCC 11) predates <print> and <mdspan>. Build against LLVM's libc++
# instead, matching the macOS toolchain. Scoped to CXX so the flag never
# reaches nvcc's host-compiler pass on the CUDA kernel TU.
if(NOT APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-stdlib=libc++>)
    add_link_options(-stdlib=libc++)
endif()

# ---- Options ----------------------------------------------------------------

option(BONSAI_TESTS "Build tests" ON)
option(BONSAI_OPENMP "Enable OpenMP parallelism" ON)
option(BONSAI_PYTHON "Build the Python extension module" OFF)
option(BONSAI_SANITIZE "Build with ASan + UBSan (CI test profile)" OFF)

if(BONSAI_SANITIZE)
    if(BONSAI_CUDA)
        message(FATAL_ERROR "BONSAI_SANITIZE does not compose with the CUDA "
                            "kernel TU; sanitize CPU builds only.")
    endif()
    # Global so FetchContent deps (Catch2) get the same instrumentation.
    add_compile_options(-fsanitize=address,undefined -fno-sanitize-recover=all
                        -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()

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
    # Homebrew's libomp is keg-only and the llvm kegs ship no OpenMP of
    # their own, so FindOpenMP fails on a stock Mac without hints. The
    # failure used to be a silent serial fallback — and serial builds train
    # different (valid, but not build-reproducible) model bits than
    # parallel ones, which masqueraded as issue #72's cross-architecture
    # divergence.
    if(APPLE AND EXISTS /opt/homebrew/opt/libomp/include/omp.h
       AND NOT DEFINED OpenMP_CXX_FLAGS)
        set(OpenMP_CXX_FLAGS "-fopenmp;-I/opt/homebrew/opt/libomp/include")
        set(OpenMP_CXX_LIB_NAMES omp)
        set(OpenMP_omp_LIBRARY /opt/homebrew/opt/libomp/lib/libomp.dylib)
    endif()
    find_package(OpenMP COMPONENTS CXX)
    if(NOT OpenMP_CXX_FOUND)
        message(FATAL_ERROR
            "OpenMP not found. bonsai refuses the silent serial fallback: a "
            "serial build trains different model bits than a parallel one "
            "(issue #72). Install libomp (brew install libomp / apt install "
            "libomp-21-dev) or pass -DBONSAI_OPENMP=OFF to build serial "
            "deliberately.")
    endif()
    if(OpenMP_CXX_FOUND)
        target_compile_definitions(bonsai_parallel INTERFACE BONSAI_USE_OPENMP)
        find_library(BONSAI_LIBOMP_A libomp.a
                     HINTS /opt/homebrew/opt/libomp/lib /usr/local/opt/libomp/lib
                           /usr/lib/llvm-21/lib)
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
    endif()
endif()

# ---- CUDA (optional) ---------------------------------------------------------
# Interface target consumed by bonsai_tree. cuda_depthwise is registered in
# every build (the registry headers are CUDA-free), so no config macro
# exists: when BONSAI_CUDA is ON this target carries the real
# histogram-kernel library, when OFF a stub TU that throws on training
# (src/cuda/histogram_engine_stub.cpp, added in src/CMakeLists.txt).
# Availability is a runtime question — bonsai::cuda_available().
#
# The kernel TU is CUDA C++ compiled by the project's own clang (-x cuda),
# not nvcc — same C++23, same libc++, so it uses bonsai types directly and
# no second toolchain or C ABI is involved.

option(BONSAI_CUDA "Enable the CUDA histogram backend (cuda_depthwise grower)" OFF)

add_library(bonsai_gpu INTERFACE)
if(BONSAI_CUDA)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR "BONSAI_CUDA compiles the kernel TU as clang CUDA "
                            "C++; build with clang (the project default).")
    endif()
    find_package(CUDAToolkit REQUIRED)
    set(BONSAI_CUDA_ARCH "native" CACHE STRING
        "GPU architecture for clang --offload-arch (e.g. sm_87, or native)")
    add_library(bonsai_cuda_kernels STATIC src/cuda/histogram_engine.cu)
    set_source_files_properties(src/cuda/histogram_engine.cu
                                PROPERTIES LANGUAGE CXX)
    target_compile_options(bonsai_cuda_kernels PRIVATE
        -x cuda
        --offload-arch=${BONSAI_CUDA_ARCH}
        --cuda-path=${CUDAToolkit_LIBRARY_ROOT}
        -Wno-unknown-cuda-version
        # Target flags come last, overriding the directory-level
        # -ffp-contract=off: fma is the native device op and the kernels'
        # f32-chunk/f64-merge scheme owns its own precision story.
        -ffp-contract=fast)
    target_include_directories(bonsai_cuda_kernels PRIVATE ${CMAKE_SOURCE_DIR}/include)
    # NVIDIA's headers #error on libc++ hosts on x86_64; this macro is their
    # sanctioned bypass (parity-validated on A100/sm_80). No-op on aarch64.
    target_compile_definitions(bonsai_cuda_kernels PRIVATE _ALLOW_UNSUPPORTED_LIBCPP)
    # bonsai_parallel gives the CPU-fallback path across-node OpenMP; clang
    # compiles OpenMP host code and CUDA in the same TU without conflict.
    target_link_libraries(bonsai_cuda_kernels PRIVATE bonsai_parallel bonsai_warnings)
    target_link_libraries(bonsai_gpu INTERFACE bonsai_cuda_kernels CUDA::cudart)
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
    find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
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
    else()
        # GNU-ld equivalent: keep static-library symbols (libomp among
        # them) local to the module.
        target_link_options(_bonsai PRIVATE "-Wl,--exclude-libs,ALL")
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
    # A standalone target, not a _bonsai POST_BUILD: pure .py edits must
    # refresh the package even when the extension is up to date. Glob runs
    # at configure time — a NEW .py file needs one reconfigure.
    file(GLOB BONSAI_PY_SOURCES ${CMAKE_SOURCE_DIR}/python/bonsai/*.py)
    add_custom_target(bonsai_py_package
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${BONSAI_PY_SOURCES}
            ${CMAKE_BINARY_DIR}/python/bonsai/)
    add_dependencies(_bonsai bonsai_py_package)
    install(TARGETS _bonsai LIBRARY DESTINATION bonsai)
endif()
