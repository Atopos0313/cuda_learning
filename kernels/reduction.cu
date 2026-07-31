#include "reduction.h"
#include "common/cuda_check.h"

#include <cuda_runtime.h>
#include <math_constants.h>
#include <stdexcept>

__device__ __forceinline__
float warp_reduce_sum(float value)
{
    // 二进制中是 32 个 1，表示这个 warp 的 32 个 lane 都参与 shuffle
    constexpr unsigned int full_mask =
        0xffffffffu;

    for (
        int offset = warpSize / 2;
        offset > 0;
        offset /= 2
    ){
        value += __shfl_down_sync(
            full_mask,
            value,
            offset
        );
    }

    return value;
}
__device__ __forceinline__
float warp_reduce_max(float value)
{
    constexpr unsigned int full_mask =
        0xffffffffu;

    for (
        int offset = warpSize / 2;
        offset > 0;
        offset /= 2
    ){
        const float other =
            __shfl_down_sync(
                full_mask,
                value,
                offset
            );

        value = fmaxf(
            value,
            other
        );
    }

    return value;
}

__global__ void reduce_sum_blocks_kernel(
    const float* d_input,
    float* d_block_sums,
    int n
){
    // extern __shared__ 不是引用外部变量，而是告诉 CUDA：
    // “这块 shared memory 的大小由 kernel 启动时决定”。
    // 在核函数启动时，第三个参数 num * sizeof(float)
    extern __shared__ float shared_values[];
    const int global_index = blockDim.x * blockIdx.x +threadIdx.x;
    const int local_index = threadIdx.x;
    if (global_index < n){
        shared_values[local_index] = d_input[global_index];

    }else{
        shared_values[local_index] = 0.0f;
    }
    __syncthreads();
    for(int stride = blockDim.x / 2; stride > 0; stride /= 2){

        if (local_index < stride){
            shared_values[local_index] += shared_values[local_index + stride];

        }
        __syncthreads();
    }
    if (local_index == 0){
        // 为什么是d_block_sums[blockIdx.x]，因为一个block负责计算一块数据，保存到自己blockid的位置上
        d_block_sums[blockIdx.x] = shared_values[0];
    }
}

__global__ void reduce_max_blocks_kernel(
    const float* d_input,
    float* d_block_maxes,
    int n
){
    extern __shared__ float shared_values[];

    const int global_index =
        blockDim.x * blockIdx.x + threadIdx.x;
    const int local_index = threadIdx.x;

    shared_values[local_index] =
        global_index < n
            ? d_input[global_index]
            : -CUDART_INF_F;

    __syncthreads();

    for (
        int stride = blockDim.x / 2;
        stride > 0;
        stride /= 2
    ){
        if (local_index < stride){
            shared_values[local_index] = fmaxf(
                shared_values[local_index],
                shared_values[local_index + stride]
            );
        }

        __syncthreads();
    }

    if (local_index == 0){
        d_block_maxes[blockIdx.x] =
            shared_values[0];
    }
}

int reduction_block_count(
    const int n,
    const int threads_per_block
){
    if (n <= 0){
        return 0;
    }
    if (threads_per_block <= 0) {
        throw std::invalid_argument(
        "threads_per_block must be positive"
    );
}
    const int grid = (n + threads_per_block - 1) / threads_per_block;
    return grid;
}

void reduce_sum_blocks_device(
    const float* d_input,
    float* d_block_sums,
    int n,
    int threads_per_block,
    cudaStream_t stream
){
    if (n <= 0) {
        return;
    }
    int grid = reduction_block_count(n, threads_per_block);
    reduce_sum_blocks_kernel<<<grid, threads_per_block, threads_per_block * sizeof(float), stream>>>(d_input, d_block_sums, n);
    CUDA_CHECK(cudaGetLastError());
}

void reduce_max_blocks_device(
    const float* d_input,
    float* d_block_maxes,
    int n,
    int threads_per_block,
    cudaStream_t stream
){
    if (n <= 0) {
        return;
    }

    const int grid =
        reduction_block_count(n, threads_per_block);

    reduce_max_blocks_kernel<<<
        grid,
        threads_per_block,
        threads_per_block * sizeof(float),
        stream
    >>>(
        d_input,
        d_block_maxes,
        n
    );

    CUDA_CHECK(cudaGetLastError());
}

__global__ void reduce_sum_blocks_warp_kernel(
    const float* d_input,
    float* d_block_sums,
    int n
)
{
    extern __shared__ float shared_values[];

    const int global_index =
        blockDim.x * blockIdx.x + threadIdx.x;

    const int local_index =
        threadIdx.x;

    shared_values[local_index] =
        global_index < n
            ? d_input[global_index]
            : 0.0F;

    __syncthreads();

    for (
        int stride = blockDim.x / 2;
        stride > warpSize;
        stride /= 2
    ){
        if (local_index < stride) {
            shared_values[local_index] +=
                shared_values[
                    local_index + stride
                ];
        }

        __syncthreads();
    }

    if (local_index < warpSize) {
        float value =
            shared_values[local_index];

        if (blockDim.x > warpSize) {
            value +=
                shared_values[
                    local_index + warpSize
                ];
        }

        value = warp_reduce_sum(value);

        if (local_index == 0) {
            d_block_sums[blockIdx.x] =
                value;
        }
    }
}

__global__ void reduce_max_blocks_warp_kernel(
    const float* d_input,
    float* d_block_maxes,
    int n
)
{
    extern __shared__ float shared_values[];

    const int global_index =
        blockDim.x * blockIdx.x + threadIdx.x;
    const int local_index = threadIdx.x;

    shared_values[local_index] =
        global_index < n
            ? d_input[global_index]
            : -CUDART_INF_F;

    __syncthreads();

    for (
        int stride = blockDim.x / 2;
        stride > warpSize;
        stride /= 2
    ){
        if (local_index < stride) {
            shared_values[local_index] = fmaxf(
                shared_values[local_index],
                shared_values[local_index + stride]
            );
        }

        __syncthreads();
    }

    if (local_index < warpSize) {
        float value =
            shared_values[local_index];

        if (blockDim.x > warpSize) {
            value = fmaxf(
                value,
                shared_values[
                    local_index + warpSize
                ]
            );
        }

        value = warp_reduce_max(value);

        if (local_index == 0) {
            d_block_maxes[blockIdx.x] =
                value;
        }
    }
}

void validate_warp_reduction_block_size(
    int threads_per_block
)
{
    constexpr int warp_width = 32;
    constexpr int maximum_threads_per_block = 1024;

    const bool is_power_of_two =
        threads_per_block > 0 &&
        (
            threads_per_block &
            (threads_per_block - 1)
        ) == 0;

    if (
        threads_per_block < warp_width ||
        threads_per_block >
            maximum_threads_per_block ||
        !is_power_of_two
    ){
        throw std::invalid_argument(
            "warp reduction block size must be "
            "a power of two in [32, 1024]"
        );
    }
}

void reduce_sum_blocks_warp_device(
    const float* d_input,
    float* d_block_sums,
    int n,
    int threads_per_block,
    cudaStream_t stream
)
{
    if (n <= 0) {
        return;
    }

    validate_warp_reduction_block_size(
        threads_per_block
    );

    const int grid =
        reduction_block_count(n, threads_per_block);

    reduce_sum_blocks_warp_kernel<<<
        grid,
        threads_per_block,
        threads_per_block * sizeof(float),
        stream
    >>>(
        d_input,
        d_block_sums,
        n
    );

    CUDA_CHECK(cudaGetLastError());
}

void reduce_max_blocks_warp_device(
    const float* d_input,
    float* d_block_maxes,
    int n,
    int threads_per_block,
    cudaStream_t stream
)
{
    if (n <= 0) {
        return;
    }

    validate_warp_reduction_block_size(
        threads_per_block
    );

    const int grid =
        reduction_block_count(n, threads_per_block);

    reduce_max_blocks_warp_kernel<<<
        grid,
        threads_per_block,
        threads_per_block * sizeof(float),
        stream
    >>>(
        d_input,
        d_block_maxes,
        n
    );

    CUDA_CHECK(cudaGetLastError());
}
