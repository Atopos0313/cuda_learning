#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>

// CUDA Runtime functions return an error code instead of throwing an
// exception. This macro turns a failed call into a readable diagnostic and a
// C++ exception, so already-constructed RAII objects can still clean up.
#define CUDA_CHECK(call)                                                               \
    do                                                                                 \
    {                                                                                  \
        const cudaError_t cuda_check_error = (call);                                   \
        if (cuda_check_error != cudaSuccess)                                           \
        {                                                                              \
            std::fprintf(stderr,                                                       \
                         "CUDA error\nAPI: %s\nFile: %s\nLine: %d\nError: %s\n",       \
                         #call,                                                        \
                         __FILE__,                                                     \
                         __LINE__,                                                     \
                         cudaGetErrorString(cuda_check_error));                        \
            throw std::runtime_error(cudaGetErrorString(cuda_check_error));            \
        }                                                                              \
    } while (false)

// Use this strong check at correctness/debug boundaries. The synchronization
// deliberately waits for execution-time failures and is not suitable inside
// a performance timing loop.
#define CUDA_KERNEL_CHECK()                                                            \
    do                                                                                 \
    {                                                                                  \
        CUDA_CHECK(cudaGetLastError());                                                \
        CUDA_CHECK(cudaDeviceSynchronize());                                           \
    } while (false)
