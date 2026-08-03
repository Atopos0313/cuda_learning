#pragma once

#include <cuda_runtime.h>

void transpose_naive_device(const float* d_input,
                            float* d_output,
                            int height,
                            int width,
                            cudaStream_t stream = nullptr);

void transpose_tiled_device(const float* d_input,
                            float* d_output,
                            int height,
                            int width,
                            cudaStream_t stream = nullptr);

void transpose_padded_device(const float* d_input,
                             float* d_output,
                             int height,
                             int width,
                             cudaStream_t stream = nullptr);