#include "memory_access.h"

#include "common/cuda_check.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <algorithm>
#include <cstdint>

// 匿名命名空间，不把函数暴露给外界
namespace
{
constexpr int kWarpSize = 32;

void validate_launch_config(int threads_per_block)
{
    if (threads_per_block <= 0 || threads_per_block > 1024) {
        throw std::invalid_argument(
            "threads_per_block must be in [1, 1024]"
        );
    }
}
constexpr int KMaxBlocks = 4096;

/*
一个float4 中包含4个float 所以所占内存为连续的16字节，
从 pointer 指向的位置开始，每次按照一个 16 字节的 float4 读取数据，
pointer 最好位于一个能被 16 整除的地址上
*/
bool is_float4_aligned(const void* pointer)
{
    // 指针转换成整数，std::uintptr_t是专门用于保存指针地址数值的无符号整数类型
    // 使用 reinterpret_cast把这个指针保存的地址，重新解释成整数
    const auto address =
        reinterpret_cast<std::uintptr_t>(pointer);
    // alignof 用于查询一个类型的对齐要求
    return address % alignof(float4) == 0;
}

}

__global__ void axpy_contiguous_kernel(
    const float* x,
    float* y,
    int n,
    float alpha
)
{
    const int index =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    if (index < n) {
        y[index] = alpha * x[index] + y[index];
    }
}

__global__ void axpy_strided_kernel(
    const float* x,
    float* y,
    int n,
    float alpha
)
{
    const int linear_index =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    const int bulk_size =
        n - n % kWarpSize;

    int index = linear_index;

    if (linear_index < bulk_size) {
        const int rows = bulk_size / kWarpSize;
        const int lane = linear_index % kWarpSize;
        const int row = linear_index / kWarpSize;

        // Adjacent lanes access different columns of a conceptual
        // rows-by-32 matrix, so their addresses are rows floats apart.
        index = lane * rows + row;
    }

    if (index < n) {
        y[index] = alpha * x[index] + y[index];
    }
}

__global__ void axpy_float4_kernel(
    const float* x,
    float* y,
    int n,
    float alpha
)
{
    const int start =
        static_cast<int>(
            blockIdx.x * blockDim.x + threadIdx.x
        );
    // 计算有多少个线程，作为步长，这样一个线程处理多个值的时候不会遗漏不会重复
    const int stride =
        static_cast<int>(
            gridDim.x * blockDim.x
        );
    // 计算有多少个float4
    const int vector_count = n / 4;
    // 四个四个为一组来看待
    const float4* x4 =
        reinterpret_cast<const float4*>(x);

    float4* y4 =
        reinterpret_cast<float4*>(y);
    // vector_index已经是float4的索引了
    for (int vector_index = start;
         vector_index < vector_count;
         vector_index += stride)
    {
        const float4 x_value = x4[vector_index];
        float4 y_value = y4[vector_index];

        y_value.x = alpha * x_value.x + y_value.x;
        y_value.y = alpha * x_value.y + y_value.y;
        y_value.z = alpha * x_value.z + y_value.z;
        y_value.w = alpha * x_value.w + y_value.w;

        y4[vector_index] = y_value;
    }

    // 只让线程 0 处理不足 4 个元素的尾部。
    if (start == 0) {
        const int tail_begin = vector_count * 4;

        for (int index = tail_begin;
             index < n;
             ++index)
        {
            y[index] = alpha * x[index] + y[index];
        }
    }
}

void axpy_contiguous_device(
    const float* d_x,
    float* d_y,
    int n,
    float alpha,
    int threads_per_block,
    cudaStream_t stream
)
{
    if (n <= 0) {
        return;
    }

    validate_launch_config(threads_per_block);

    const int blocks =
        1 + (n - 1) / threads_per_block;

    axpy_contiguous_kernel<<<
        blocks,
        threads_per_block,
        0,
        stream
    >>>(
        d_x,
        d_y,
        n,
        alpha
    );

    CUDA_CHECK(cudaGetLastError());
}

void axpy_strided_device(
    const float* d_x,
    float* d_y,
    int n,
    float alpha,
    int threads_per_block,
    cudaStream_t stream
)
{
    if (n <= 0) {
        return;
    }

    validate_launch_config(threads_per_block);

    if (threads_per_block % kWarpSize != 0) {
        throw std::invalid_argument(
            "strided AXPY requires a multiple of 32 threads per block"
        );
    }

    const int blocks =
        1 + (n - 1) / threads_per_block;

    axpy_strided_kernel<<<
        blocks,
        threads_per_block,
        0,
        stream
    >>>(
        d_x,
        d_y,
        n,
        alpha
    );

    CUDA_CHECK(cudaGetLastError());
}

void axpy_float4_device(
    const float* d_x,
    float* d_y,
    int n,
    float alpha,
    int threads_per_block,
    cudaStream_t stream
)
{
    if (n <= 0) {
        return;
    }

    validate_launch_config(threads_per_block);

    if (!is_float4_aligned(d_x) ||
        !is_float4_aligned(d_y))
    {
        throw std::invalid_argument(
            "float4 AXPY requires 16-byte aligned pointers"
        );
    }

    const int vector_count = n / 4;

    int required_blocks = 1;

    if (vector_count > 0) {
        required_blocks =
            (vector_count + threads_per_block - 1) / threads_per_block;
    }
    const int blocks =
        std::min(required_blocks, KMaxBlocks);

    axpy_float4_kernel<<<
        blocks,
        threads_per_block,
        0,
        stream
    >>>(
        d_x,
        d_y,
        n,
        alpha
    );

    CUDA_CHECK(cudaGetLastError());
}
