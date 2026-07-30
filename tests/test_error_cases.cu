#include "common/cuda_check.h"
#include "common/device_buffer.h"

#include <cuda_runtime.h>
#include <exception>
#include <cstdio>

__global__ void out_of_bounds_kernel(float* data)
{
    const int index =
        blockIdx.x * blockDim.x + threadIdx.x;

    // 故意写到分配范围之外。
    data[index + 1024] = 1.0F;
}

int main()
{
    try {
        DeviceBuffer<float> buffer(1);

        out_of_bounds_kernel<<<1, 1>>>(
            buffer.data()
        );

        CUDA_KERNEL_CHECK();

        return 0;
    }
    catch (const std::exception& error) {
        std::fprintf(
            stderr,
            "Caught exception: %s\n",
            error.what()
        );

        return 1;
    }
}