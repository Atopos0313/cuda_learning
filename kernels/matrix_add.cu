#include "matrix_add.h"

#include "common/cuda_check.h"
#include "common/device_buffer.h"
#include "common/launch_config.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace
{
__global__ void
matrix_add_kernel(const float* a, const float* b, float* c, int height, int width)
{
    // x identifies a column and y identifies a row in this project.
    const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);

    if (row < height && col < width)
    {
        // Row-major storage: skip row complete rows, then move col elements.
        const int index = row * width + col;
        c[index] = a[index] + b[index];
    }
}
} // namespace

void matrix_add(const float* h_a, const float* h_b, float* h_c, int height, int width)
{
    if (height <= 0 || width <= 0)
    {
        return;
    }

    const std::size_t element_count =
        static_cast<std::size_t>(height) * static_cast<std::size_t>(width);
    const std::size_t bytes = element_count * sizeof(float);

    DeviceBuffer<float> d_a(element_count);
    DeviceBuffer<float> d_b(element_count);
    DeviceBuffer<float> d_c(element_count);

    CUDA_CHECK(cudaMemcpy(d_a.data(), h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b.data(), h_b, bytes, cudaMemcpyHostToDevice));

    constexpr int kBlockWidth = 16;
    constexpr int kBlockHeight = 16;
    const dim3 block(kBlockWidth, kBlockHeight);
    const dim3 grid(ceil_div(width, kBlockWidth), ceil_div(height, kBlockHeight));

    matrix_add_kernel<<<grid, block>>>(
        d_a.data(), d_b.data(), d_c.data(), height, width);
    CUDA_KERNEL_CHECK();

    CUDA_CHECK(cudaMemcpy(h_c, d_c.data(), bytes, cudaMemcpyDeviceToHost));
}
