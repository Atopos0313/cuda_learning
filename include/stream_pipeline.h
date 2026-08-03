#pragma once

#include <cuda_runtime.h>

__global__ void simpleKernel(
    const float* d_input,
    float* d_output,
    int n
);