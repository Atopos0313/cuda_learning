#include "memory_access.h"

#include "common/cuda_check.h"
#include "common/launch_config.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace
{
constexpr int kWarpSize = 32;

constexpr int kMaximumBlocks = 4096;

bool is_float4_aligned(const void* pointer)
{
    // Convert the address to an integer so its byte alignment can be tested.
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    return address % alignof(float4) == 0;
}

} // namespace

__global__ void axpy_contiguous_kernel(const float* x, float* y, int n, float alpha)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    if (index < n)
    {
        y[index] = alpha * x[index] + y[index];
    }
}

__global__ void axpy_strided_kernel(const float* x, float* y, int n, float alpha)
{
    const int linear_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    const int bulk_size = n - n % kWarpSize;

    int index = linear_index;

    if (linear_index < bulk_size)
    {
        const int rows = bulk_size / kWarpSize;
        const int lane = linear_index % kWarpSize;
        const int row = linear_index / kWarpSize;

        // Adjacent lanes access different columns of a conceptual
        // rows-by-32 matrix, so their addresses are rows floats apart.
        index = lane * rows + row;
    }

    if (index < n)
    {
        y[index] = alpha * x[index] + y[index];
    }
}

__global__ void axpy_float4_kernel(const float* x, float* y, int n, float alpha)
{
    const int start = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    // The full-grid stride lets each thread process multiple float4 values.
    const int stride = static_cast<int>(gridDim.x * blockDim.x);
    const int vector_count = n / 4;
    const float4* x4 = reinterpret_cast<const float4*>(x);

    float4* y4 = reinterpret_cast<float4*>(y);
    for (int vector_index = start; vector_index < vector_count; vector_index += stride)
    {
        const float4 x_value = x4[vector_index];
        float4 y_value = y4[vector_index];

        y_value.x = alpha * x_value.x + y_value.x;
        y_value.y = alpha * x_value.y + y_value.y;
        y_value.z = alpha * x_value.z + y_value.z;
        y_value.w = alpha * x_value.w + y_value.w;

        y4[vector_index] = y_value;
    }

    // Only one thread handles the final zero to three scalar elements.
    if (start == 0)
    {
        const int tail_begin = vector_count * 4;

        for (int index = tail_begin; index < n; ++index)
        {
            y[index] = alpha * x[index] + y[index];
        }
    }
}

void axpy_contiguous_device(const float* d_x,
                            float* d_y,
                            int n,
                            float alpha,
                            int threads_per_block,
                            cudaStream_t stream)
{
    if (n <= 0)
    {
        return;
    }

    validate_threads_per_block(threads_per_block);

    const int blocks = ceil_div(n, threads_per_block);

    axpy_contiguous_kernel<<<blocks, threads_per_block, 0, stream>>>(
        d_x, d_y, n, alpha);

    CUDA_CHECK(cudaGetLastError());
}

void axpy_strided_device(const float* d_x,
                         float* d_y,
                         int n,
                         float alpha,
                         int threads_per_block,
                         cudaStream_t stream)
{
    if (n <= 0)
    {
        return;
    }

    validate_threads_per_block(threads_per_block);

    if (threads_per_block % kWarpSize != 0)
    {
        throw std::invalid_argument(
            "strided AXPY requires a multiple of 32 threads per block");
    }

    const int blocks = ceil_div(n, threads_per_block);

    axpy_strided_kernel<<<blocks, threads_per_block, 0, stream>>>(d_x, d_y, n, alpha);

    CUDA_CHECK(cudaGetLastError());
}

void axpy_float4_device(const float* d_x,
                        float* d_y,
                        int n,
                        float alpha,
                        int threads_per_block,
                        cudaStream_t stream)
{
    if (n <= 0)
    {
        return;
    }

    validate_threads_per_block(threads_per_block);

    if (!is_float4_aligned(d_x) || !is_float4_aligned(d_y))
    {
        throw std::invalid_argument("float4 AXPY requires 16-byte aligned pointers");
    }

    const int vector_count = n / 4;

    int required_blocks = 1;

    if (vector_count > 0)
    {
        required_blocks = ceil_div(vector_count, threads_per_block);
    }
    const int blocks = std::min(required_blocks, kMaximumBlocks);

    axpy_float4_kernel<<<blocks, threads_per_block, 0, stream>>>(d_x, d_y, n, alpha);

    CUDA_CHECK(cudaGetLastError());
}
