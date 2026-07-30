/*
记住:
threadIdx.x / threadIdx.y
线程在当前 block 内的位置

blockIdx.x / blockIdx.y
当前 block 在 grid 中的位置

col / row
当前线程负责的全局矩阵坐标

idx
该矩阵坐标对应的一维内存下标

*/

#include "matrix_add.h"
#include "common/cuda_check.h"
#include "common/device_buffer.h"

#include <cuda_runtime.h>


__global__ void matrix_add_kernel(
    const float* A,
    const float* B,
    float* C,
    int height,
    int width)
{
    // 当前线程负责的全局矩阵位置
    // threadIdx.x 管列
    // threadIdx.y 管行
    // matrix[row][col]
    /*
    x 横向 → 列 col是全局位置的列号 → width是矩阵的列数
    y 纵向 → 行 row → height
    */
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // 这个矩阵位置对应的一维内存下标
    if(col < width && row < height){
        int idx = row * width + col;
        C[idx] = A[idx] + B[idx];
    }

}
void matrix_add(
    const float* h_A,
    const float* h_B,
    float* h_C,
    int height,
    int width)
{
    if (height <= 0 || width <= 0) {
        return;
    }

    const std::size_t element_count =
        static_cast<std::size_t>(height) *
        static_cast<std::size_t>(width);

    const std::size_t bytes =
        element_count * sizeof(float);

    // float* d_A = nullptr;
    // float* d_B = nullptr;
    // float* d_C = nullptr;
    DeviceBuffer<float> d_A(element_count);
    DeviceBuffer<float> d_B(element_count);
    DeviceBuffer<float> d_C(element_count);


    // CUDA_CHECK(cudaMalloc(&d_A, bytes));
    // CUDA_CHECK(cudaMalloc(&d_B, bytes));
    // CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(
        d_A.data(),
        h_A,
        bytes,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_B.data(),
        h_B,
        bytes,
        cudaMemcpyHostToDevice
    ));
    const int block_x = 16;
    const int block_y = 16;
    const dim3 block(block_x, block_y);

    const dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );

    matrix_add_kernel<<<grid, block>>>(
        d_A.data(),
        d_B.data(),
        d_C.data(),
        height,
        width
    );

    CUDA_KERNEL_CHECK();

    CUDA_CHECK(cudaMemcpy(
        h_C,
        d_C.data(),
        bytes,
        cudaMemcpyDeviceToHost
    ));

    // CUDA_CHECK(cudaFree(d_A));
    // CUDA_CHECK(cudaFree(d_B));
    // CUDA_CHECK(cudaFree(d_C));
}
