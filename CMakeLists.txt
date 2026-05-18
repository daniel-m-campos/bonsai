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
    GIT_TAG        v2.4.2
)

# TOML config, header-only.
FetchContent_Declare(tomlplusplus
    GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
    GIT_TAG        v3.4.0
)

# Logging. Use std::format (C++23) instead of bundled fmt — sidesteps the
# FMT_STRING / consteval breakage in clang-trunk against bundled fmt.
set(SPDLOG_BUILD_SHARED  OFF CACHE BOOL "" FORCE)
set(SPDLOG_USE_STD_FORMAT ON CACHE BOOL "" FORCE)
FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        v1.15.0
)

# nlohmann/json — single-header, MIT, used for MessagePack model format
# via to_msgpack / from_msgpack. No transitive deps.
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)

FetchContent_MakeAvailable(CLI11 tomlplusplus spdlog nlohmann_json)

# ---- Subdirectories ---------------------------------------------------------

add_subdirectory(src)

if(BONSAI_TESTS)
    enable_testing()
    add_subdirectory(tests)
    add_subdirectory(benchmarks)
endif()
