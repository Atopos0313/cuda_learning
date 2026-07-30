#include "reduction.h"
#include "common/cuda_check.h"

#include <cuda_runtime.h>
#include <stdexcept>

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
