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

# ---- Options ----------------------------------------------------------------

option(BONSAI_TESTS "Build tests" ON)

# ---- Dependencies via FetchContent ------------------------------------------
# Fetched lazily — only Catch2 (and toml++ etc. when first consumer lands).

include(FetchContent)

if(BONSAI_TESTS)
    FetchContent_Declare(Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.5.4
    )
    FetchContent_MakeAvailable(Catch2)
endif()

# ---- Subdirectories ---------------------------------------------------------

add_subdirectory(src)

if(BONSAI_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
