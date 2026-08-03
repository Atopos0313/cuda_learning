#include "reduction.h"

#include "common/cuda_check.h"
#include "common/launch_config.h"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <stdexcept>

namespace
{
constexpr int kWarpSize = 32;

bool is_power_of_two(int value)
{
    return value > 0 && (value & (value - 1)) == 0;
}

void validate_reduction_block_size(int threads_per_block)
{
    validate_threads_per_block(threads_per_block);

    // The tree halves its active range on every step, so the teaching kernel
    // requires a power-of-two block size.
    if (!is_power_of_two(threads_per_block))
    {
        throw std::invalid_argument("reduction block size must be a power of two");
    }
}

void validate_warp_reduction_block_size(int threads_per_block)
{
    validate_reduction_block_size(threads_per_block);

    if (threads_per_block < kWarpSize)
    {
        throw std::invalid_argument("warp reduction block size must be in [32, 1024]");
    }
}

__device__ __forceinline__ float warp_reduce_sum(float value)
{
    constexpr unsigned int kFullWarpMask = 0xffffffffU;

    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        value += __shfl_down_sync(kFullWarpMask, value, offset);
    }

    return value;
}

__device__ __forceinline__ float warp_reduce_max(float value)
{
    constexpr unsigned int kFullWarpMask = 0xffffffffU;

    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        const float other = __shfl_down_sync(kFullWarpMask, value, offset);
        value = fmaxf(value, other);
    }

    return value;
}

__global__ void reduce_sum_blocks_kernel(const float* input, float* block_sums, int n)
{
    // The third kernel-launch argument supplies this dynamic shared memory.
    extern __shared__ float shared_values[];

    const int global_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int local_index = static_cast<int>(threadIdx.x);

    shared_values[local_index] = global_index < n ? input[global_index] : 0.0F;
    __syncthreads();

    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride /= 2)
    {
        if (local_index < stride)
        {
            shared_values[local_index] += shared_values[local_index + stride];
        }
        __syncthreads();
    }

    if (local_index == 0)
    {
        block_sums[blockIdx.x] = shared_values[0];
    }
}

__global__ void reduce_max_blocks_kernel(const float* input, float* block_maxes, int n)
{
    extern __shared__ float shared_values[];

    const int global_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int local_index = static_cast<int>(threadIdx.x);

    shared_values[local_index] = global_index < n ? input[global_index] : -CUDART_INF_F;
    __syncthreads();

    for (int stride = static_cast<int>(blockDim.x) / 2; stride > 0; stride /= 2)
    {
        if (local_index < stride)
        {
            shared_values[local_index] =
                fmaxf(shared_values[local_index], shared_values[local_index + stride]);
        }
        __syncthreads();
    }

    if (local_index == 0)
    {
        block_maxes[blockIdx.x] = shared_values[0];
    }
}

__global__ void
reduce_sum_blocks_warp_kernel(const float* input, float* block_sums, int n)
{
    extern __shared__ float shared_values[];

    const int global_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int local_index = static_cast<int>(threadIdx.x);

    shared_values[local_index] = global_index < n ? input[global_index] : 0.0F;
    __syncthreads();

    // Stop the shared-memory tree at 64 active values. The first warp then
    // combines the two remaining 32-value halves with shuffle instructions.
    for (int stride = static_cast<int>(blockDim.x) / 2; stride > kWarpSize; stride /= 2)
    {
        if (local_index < stride)
        {
            shared_values[local_index] += shared_values[local_index + stride];
        }
        __syncthreads();
    }

    if (local_index < kWarpSize)
    {
        float value = shared_values[local_index];

        if (blockDim.x > kWarpSize)
        {
            value += shared_values[local_index + kWarpSize];
        }

        value = warp_reduce_sum(value);

        if (local_index == 0)
        {
            block_sums[blockIdx.x] = value;
        }
    }
}

__global__ void
reduce_max_blocks_warp_kernel(const float* input, float* block_maxes, int n)
{
    extern __shared__ float shared_values[];

    const int global_index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int local_index = static_cast<int>(threadIdx.x);

    shared_values[local_index] = global_index < n ? input[global_index] : -CUDART_INF_F;
    __syncthreads();

    for (int stride = static_cast<int>(blockDim.x) / 2; stride > kWarpSize; stride /= 2)
    {
        if (local_index < stride)
        {
            shared_values[local_index] =
                fmaxf(shared_values[local_index], shared_values[local_index + stride]);
        }
        __syncthreads();
    }

    if (local_index < kWarpSize)
    {
        float value = shared_values[local_index];

        if (blockDim.x > kWarpSize)
        {
            value = fmaxf(value, shared_values[local_index + kWarpSize]);
        }

        value = warp_reduce_max(value);

        if (local_index == 0)
        {
            block_maxes[blockIdx.x] = value;
        }
    }
}
} // namespace

int reduction_block_count(int n, int threads_per_block)
{
    if (n <= 0)
    {
        return 0;
    }

    validate_reduction_block_size(threads_per_block);
    return ceil_div(n, threads_per_block);
}

void reduce_sum_blocks_device(const float* d_input,
                              float* d_block_sums,
                              int n,
                              int threads_per_block,
                              cudaStream_t stream)
{
    if (n <= 0)
    {
        return;
    }

    const int blocks = reduction_block_count(n, threads_per_block);
    const std::size_t shared_bytes =
        static_cast<std::size_t>(threads_per_block) * sizeof(float);

    reduce_sum_blocks_kernel<<<blocks, threads_per_block, shared_bytes, stream>>>(
        d_input, d_block_sums, n);

    CUDA_CHECK(cudaGetLastError());
}

void reduce_max_blocks_device(const float* d_input,
                              float* d_block_maxes,
                              int n,
                              int threads_per_block,
                              cudaStream_t stream)
{
    if (n <= 0)
    {
        return;
    }

    const int blocks = reduction_block_count(n, threads_per_block);
    const std::size_t shared_bytes =
        static_cast<std::size_t>(threads_per_block) * sizeof(float);

    reduce_max_blocks_kernel<<<blocks, threads_per_block, shared_bytes, stream>>>(
        d_input, d_block_maxes, n);

    CUDA_CHECK(cudaGetLastError());
}

void reduce_sum_blocks_warp_device(const float* d_input,
                                   float* d_block_sums,
                                   int n,
                                   int threads_per_block,
                                   cudaStream_t stream)
{
    if (n <= 0)
    {
        return;
    }

    validate_warp_reduction_block_size(threads_per_block);
    const int blocks = reduction_block_count(n, threads_per_block);
    const std::size_t shared_bytes =
        static_cast<std::size_t>(threads_per_block) * sizeof(float);

    reduce_sum_blocks_warp_kernel<<<blocks, threads_per_block, shared_bytes, stream>>>(
        d_input, d_block_sums, n);

    CUDA_CHECK(cudaGetLastError());
}

void reduce_max_blocks_warp_device(const float* d_input,
                                   float* d_block_maxes,
                                   int n,
                                   int threads_per_block,
                                   cudaStream_t stream)
{
    if (n <= 0)
    {
        return;
    }

    validate_warp_reduction_block_size(threads_per_block);
    const int blocks = reduction_block_count(n, threads_per_block);
    const std::size_t shared_bytes =
        static_cast<std::size_t>(threads_per_block) * sizeof(float);

    reduce_max_blocks_warp_kernel<<<blocks, threads_per_block, shared_bytes, stream>>>(
        d_input, d_block_maxes, n);

    CUDA_CHECK(cudaGetLastError());
}
