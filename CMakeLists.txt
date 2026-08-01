cmake_minimum_required(VERSION 3.28)

project(cuda_engineering LANGUAGES CXX CUDA)

include(CTest)
find_package(CUDAToolkit REQUIRED)

# ------------------------------------------------------------
# Language standards
# ------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# RTX 2080 Ti: Compute Capability 7.5
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()


# ------------------------------------------------------------
# Common compile options
# ------------------------------------------------------------

add_library(cuda_engineering_options INTERFACE)

if(MSVC)
    target_compile_options(cuda_engineering_options INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:
            /W4
            /permissive-
            /utf-8
        >

        $<$<COMPILE_LANGUAGE:CUDA>:
            -Xcompiler=/W4
            -Xcompiler=/permissive-
            -Xcompiler=/utf-8
        >

        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:
            -G
        >

        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<NOT:$<CONFIG:Debug>>>:
            -lineinfo
        >
    )
endif()


# ------------------------------------------------------------
# CUDA operator library
# ------------------------------------------------------------

add_library(cuda_ops STATIC
    kernels/vector_add.cu
    kernels/matrix_add.cu
    kernels/memory_access.cu
    kernels/transpose.cu
    kernels/reduction.cu
)

target_include_directories(cuda_ops
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(cuda_ops
    PUBLIC
        CUDA::cudart
    PRIVATE
        cuda_engineering_options
)

target_compile_features(cuda_ops
    PUBLIC
        cxx_std_17
        cuda_std_17
)

set_target_properties(cuda_ops PROPERTIES
    FOLDER "Libraries"
)


# ------------------------------------------------------------
# Applications
# ------------------------------------------------------------

add_executable(vector_add_app
    apps/vector_add_main.cpp
)

target_link_libraries(vector_add_app
    PRIVATE
        cuda_ops
        cuda_engineering_options
)

set_target_properties(vector_add_app PROPERTIES
    FOLDER "Applications"
)


add_executable(matrix_add_app
    apps/matrix_add_main.cpp
)

target_link_libraries(matrix_add_app
    PRIVATE
        cuda_ops
        cuda_engineering_options
)

set_target_properties(matrix_add_app PROPERTIES
    FOLDER "Applications"
)

add_executable(bench_vector_add
    benchmarks/bench_vector_add.cu
)

target_link_libraries(bench_vector_add
    PRIVATE
        cuda_ops
        cuda_engineering_options
)

set_target_properties(bench_vector_add PROPERTIES
    FOLDER "Benchmarks"
)

add_executable(bench_memory_access
    benchmarks/bench_memory_access.cu
)

target_link_libraries(bench_memory_access
    PRIVATE
        cuda_ops
        cuda_engineering_options
)

set_target_properties(bench_memory_access PROPERTIES
    FOLDER "Benchmarks"
)

add_executable(bench_transpose
    benchmarks/bench_transpose.cu
)

target_link_libraries(bench_transpose
    PRIVATE
        cuda_ops
        cuda_engineering_options
)

set_target_properties(bench_transpose PROPERTIES
    FOLDER "Benchmarks"
)

add_executable(bench_reduction
    benchmarks/bench_reduction.cu
)

target_link_libraries(bench_reduction
    PRIVATE
        cuda_ops
        cuda_engineering_options
)

set_target_properties(bench_reduction PROPERTIES
    FOLDER "Benchmarks"
)

add_executable(profile_copy
    kernels/profile_copy.cu
)

target_link_libraries(profile_copy
    PRIVATE
        cuda_ops
        cuda_engineering_options
)

set_target_properties(profile_copy PROPERTIES
    FOLDER "Benchmarks"
)

add_executable(stream_pipeline
    apps/stream_pipeline.cu
)

target_link_libraries(stream_pipeline
    PRIVATE
        cuda_ops
        cuda_engineering_options
)

# ------------------------------------------------------------
# Test executables
# ------------------------------------------------------------

add_executable(test_memory_access
    tests/test_memory_access.cu
)

target_link_libraries(test_memory_access
    PRIVATE
        cuda_ops
        cuda_engineering_options
)

add_executable(test_reduction
    tests/test_reduction.cu
)

target_link_libraries(test_reduction
    PRIVATE
        cuda_ops
        cuda_engineering_options
)

add_executable(test_common
    tests/test_common.cpp
)

target_link_libraries(test_common
    PRIVATE
        cuda_ops
        cuda_engineering_options
)

add_executable(test_api_error
    tests/test_api_error.cu
)

target_link_libraries(test_api_error
    PRIVATE
        cuda_ops
        cuda_engineering_options
)

add_executable(test_error_cases
    tests/test_error_cases.cu
)

target_link_libraries(test_error_cases
    PRIVATE
        cuda_ops
        cuda_engineering_options
)

set_target_properties(
    test_memory_access
    test_reduction
    test_common
    test_api_error
    test_error_cases
    PROPERTIES
        FOLDER "Tests"
)

# ------------------------------------------------------------
# Tests
# ------------------------------------------------------------

include(cmake/Tests.cmake)
