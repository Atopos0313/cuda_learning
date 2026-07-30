#pragma once

#include <cuda_runtime.h>

void vector_add(
    const float* h_a,
    const float* h_b,
    float* h_c,
    int N,
    int thread_per_block
);

void vector_add_device(
    const float* d_a,
    const float* d_b,
    float* d_c,
    int N,
    int thread_per_block,
    cudaStream_t stream = nullptr
);
