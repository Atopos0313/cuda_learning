#pragma once

#include <cuda_runtime.h>

void axpy_contiguous_device(const float* d_x,
                            float* d_y,
                            int n,
                            float alpha,
                            int threads_per_block,
                            cudaStream_t stream = nullptr);

void axpy_strided_device(const float* d_x,
                         float* d_y,
                         int n,
                         float alpha,
                         int threads_per_block,
                         cudaStream_t stream = nullptr);

void axpy_float4_device(const float* d_x,
                        float* d_y,
                        int n,
                        float alpha,
                        int threads_per_block,
                        cudaStream_t stream = nullptr);
