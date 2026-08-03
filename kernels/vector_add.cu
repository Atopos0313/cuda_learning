#include "vector_add.h"

#include "common/cuda_check.h"
#include "common/device_buffer.h"
#include "common/launch_config.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>

namespace
{
constexpr int kMaximumBlocks = 4096;

__global__ void
vector_add_stride_kernel(const float* a, const float* b, float* c, int n)
{
    const int start = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int stride = static_cast<int>(gridDim.x * blockDim.x);

    for (int index = start; index < n; index += stride)
    {
        c[index] = a[index] + b[index];
    }
}
} // namespace

void vector_add_device(const float* d_a,
                       const float* d_b,
                       float* d_c,
                       int n,
                       int threads_per_block,
                       cudaStream_t stream)
{
    if (n <= 0)
    {
        return;
    }

    validate_threads_per_block(threads_per_block);

    const int required_blocks = ceil_div(n, threads_per_block);
    const int blocks = std::min(required_blocks, kMaximumBlocks);

    vector_add_stride_kernel<<<blocks, threads_per_block, 0, stream>>>(
        d_a, d_b, d_c, n);

    CUDA_CHECK(cudaGetLastError());
}

void vector_add(
    const float* h_a, const float* h_b, float* h_c, int n, int threads_per_block)
{
    if (n <= 0)
    {
        return;
    }

    validate_threads_per_block(threads_per_block);

    const std::size_t element_count = static_cast<std::size_t>(n);
    const std::size_t bytes = element_count * sizeof(float);

    DeviceBuffer<float> d_a(element_count);
    DeviceBuffer<float> d_b(element_count);
    DeviceBuffer<float> d_c(element_count);

    CUDA_CHECK(cudaMemcpy(d_a.data(), h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b.data(), h_b, bytes, cudaMemcpyHostToDevice));

    vector_add_device(d_a.data(), d_b.data(), d_c.data(), n, threads_per_block);

    CUDA_CHECK(cudaMemcpy(h_c, d_c.data(), bytes, cudaMemcpyDeviceToHost));
}
