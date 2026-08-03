if(NOT BUILD_TESTING)
    return()
endif()

add_test(
    NAME vector_add_correctness
    COMMAND vector_add_app
)

add_test(
    NAME matrix_add_correctness
    COMMAND matrix_add_app
)

add_test(
    NAME memory_access_correctness
    COMMAND test_memory_access
)

add_test(
    NAME reduction_correctness
    COMMAND test_reduction
)

add_test(
    NAME common_components
    COMMAND test_common
)

add_test(
    NAME api_error_is_reported
    COMMAND test_api_error
)

set_tests_properties(
    vector_add_correctness
    matrix_add_correctness
    memory_access_correctness
    reduction_correctness
    common_components
    PROPERTIES
        LABELS "correctness"
)

# These examples deliberately trigger CUDA errors and return a non-zero
# process status after catching them. CTest therefore expects failure.
set_tests_properties(
    api_error_is_reported
    PROPERTIES
        LABELS "error-handling"
        WILL_FAIL TRUE
)

# An out-of-bounds GPU access is not guaranteed to be reported by the normal
# CUDA runtime because allocations have a larger granularity. Validate this
# example under Compute Sanitizer when it is installed.
find_program(
    COMPUTE_SANITIZER_EXECUTABLE
    NAMES compute-sanitizer
    HINTS "$ENV{CUDA_PATH}/compute-sanitizer"
)

if(COMPUTE_SANITIZER_EXECUTABLE)
    add_test(
        NAME kernel_error_is_reported
        COMMAND
            "${COMPUTE_SANITIZER_EXECUTABLE}"
            --tool memcheck
            --error-exitcode 1
            $<TARGET_FILE:test_error_cases>
    )
    set_tests_properties(
        kernel_error_is_reported
        PROPERTIES
            LABELS "error-handling;compute-sanitizer"
            WILL_FAIL TRUE
    )
endif()
