#include "transpose.h"

#include "common/cuda_check.h"
#include "common/launch_config.h"

#include <cuda_runtime.h>

namespace
{
constexpr int kTileDimension = 32;
constexpr int kBlockRows = 8;
} // namespace

__global__ void
transpose_naive_kernel(const float* input, float* output, int height, int width)
{
    const int row = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);

    const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    if (row < height && col < width)
    {
        output[col * height + row] = input[row * width + col];
    }
}

__global__ void
transpose_tiled_kernel(const float* input, float* output, int height, int width)
{
    __shared__ float tile[kTileDimension][kTileDimension];

    const int input_col =
        static_cast<int>(blockIdx.x) * kTileDimension + static_cast<int>(threadIdx.x);
    // A 32x8 block cooperatively covers a 32x32 tile. Each thread therefore
    // processes four rows, separated by kBlockRows.
    const int input_row =
        static_cast<int>(blockIdx.y) * kTileDimension + static_cast<int>(threadIdx.y);
    for (int offset = 0; offset < kTileDimension; offset += kBlockRows)
    {
        if (input_col < width && input_row + offset < height)
        {
            tile[threadIdx.y + offset][threadIdx.x] =
                input[(input_row + offset) * width + input_col];
        }
    }

    __syncthreads();

    const int output_col =
        static_cast<int>(blockIdx.y) * kTileDimension + static_cast<int>(threadIdx.x);

    const int output_row =
        static_cast<int>(blockIdx.x) * kTileDimension + static_cast<int>(threadIdx.y);

    for (int offset = 0; offset < kTileDimension; offset += kBlockRows)
    {
        if (output_col < height && output_row + offset < width)
        {
            output[(output_row + offset) * height + output_col] =
                tile[threadIdx.x][threadIdx.y + offset];
        }
    }
}

__global__ void
transpose_padded_kernel(const float* input, float* output, int height, int width)
{
    __shared__ float tile[kTileDimension][kTileDimension + 1];

    const int input_col =
        static_cast<int>(blockIdx.x) * kTileDimension + static_cast<int>(threadIdx.x);

    const int input_row =
        static_cast<int>(blockIdx.y) * kTileDimension + static_cast<int>(threadIdx.y);

    for (int offset = 0; offset < kTileDimension; offset += kBlockRows)
    {
        if (input_col < width && input_row + offset < height)
        {
            tile[threadIdx.y + offset][threadIdx.x] =
                input[(input_row + offset) * width + input_col];
        }
    }

    __syncthreads();

    const int output_col =
        static_cast<int>(blockIdx.y) * kTileDimension + static_cast<int>(threadIdx.x);

    const int output_row =
        static_cast<int>(blockIdx.x) * kTileDimension + static_cast<int>(threadIdx.y);

    for (int offset = 0; offset < kTileDimension; offset += kBlockRows)
    {
        if (output_col < height && output_row + offset < width)
        {
            output[(output_row + offset) * height + output_col] =
                tile[threadIdx.x][threadIdx.y + offset];
        }
    }
}

void transpose_naive_device(
    const float* d_input, float* d_output, int height, int width, cudaStream_t stream)
{
    if (height <= 0 || width <= 0)
    {
        return;
    }

    const dim3 block(kTileDimension, kBlockRows);

    const dim3 grid(ceil_div(width, static_cast<int>(block.x)),
                    ceil_div(height, static_cast<int>(block.y)));

    transpose_naive_kernel<<<grid, block, 0, stream>>>(
        d_input, d_output, height, width);

    CUDA_CHECK(cudaGetLastError());
}

void transpose_tiled_device(
    const float* d_input, float* d_output, int height, int width, cudaStream_t stream)
{
    if (height <= 0 || width <= 0)
    {
        return;
    }

    const dim3 block(kTileDimension, kBlockRows);

    const dim3 grid(ceil_div(width, kTileDimension), ceil_div(height, kTileDimension));

    transpose_tiled_kernel<<<grid, block, 0, stream>>>(
        d_input, d_output, height, width);

    CUDA_CHECK(cudaGetLastError());
}

// On this architecture, successive 32-bit shared-memory words map across 32
// banks by address. Reading a column from a 32x32 tile gives all lanes the
// same address modulo 32, causing a bank conflict. The extra padding column
// changes that address stride from 32 to 33 words and distributes lanes over
// different banks.
void transpose_padded_device(
    const float* d_input, float* d_output, int height, int width, cudaStream_t stream)
{
    if (height <= 0 || width <= 0)
    {
        return;
    }

    const dim3 block(kTileDimension, kBlockRows);

    const dim3 grid(ceil_div(width, kTileDimension), ceil_div(height, kTileDimension));

    transpose_padded_kernel<<<grid, block, 0, stream>>>(
        d_input, d_output, height, width);

    CUDA_CHECK(cudaGetLastError());
}
