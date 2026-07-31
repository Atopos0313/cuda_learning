#pragma once

#include <cuda_runtime.h>

int reduction_block_count(
    int n,
    int threads_per_block
);

void reduce_sum_blocks_device(
    const float* d_input,
    float* d_block_sums,
    int n,
    int threads_per_block,
    cudaStream_t stream = nullptr
);

void reduce_max_blocks_device(
    const float* d_input,
    float* d_block_maxes,
    int n,
    int threads_per_block,
    cudaStream_t stream = nullptr
);

void reduce_sum_blocks_warp_device(
    const float* d_input,
    float* d_block_sums,
    int n,
    int threads_per_block,
    cudaStream_t stream = nullptr
);

void reduce_max_blocks_warp_device(
    const float* d_input,
    float* d_block_maxes,
    int n,
    int threads_per_block,
    cudaStream_t stream = nullptr
);
