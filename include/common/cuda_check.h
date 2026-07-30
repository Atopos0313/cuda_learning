/*
异常的执行流程
CUDA API 返回错误
→ CUDA_CHECK 执行 throw
→ 当前代码立即停止
→ 沿调用链向外寻找 catch
→ 途中销毁已经构造的局部对象
→ DeviceBuffer 析构并执行 cudaFree
→ catch 统一处理错误
*/

#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>

#define CUDA_CHECK(call)                                      \
do                                                            \
{                                                             \
    cudaError_t err = (call);                                   \
                                                              \
    if(err != cudaSuccess)                                    \
    {                                                         \
        fprintf(stderr,                                      \
            "CUDA Error\n"                                   \
            "API: %s\n"                                     \
            "File: %s\n"                                     \
            "Line: %d\n"                                     \
            "Error: %s\n",                                   \
            #call,                                            \
            __FILE__,                                        \
            __LINE__,                                        \
            cudaGetErrorString(err));                        \
                                                              \
        throw std::runtime_error(cudaGetErrorString(err));                                  \
    }                                                         \
} while(0)


#define CUDA_KERNEL_CHECK()                                  \
do                                                            \
{                                                             \
    CUDA_CHECK(cudaGetLastError());                           \
    CUDA_CHECK(cudaDeviceSynchronize());                      \
} while(0)