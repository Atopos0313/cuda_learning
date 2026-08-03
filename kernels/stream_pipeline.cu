#include "stream_pipeline.h"

__global__ void simpleKernel(
    const float* d_input,
    float* d_output,
    int n
)
{
    const int index =
        blockDim.x * blockIdx.x + threadIdx.x;

    if (index < n)
    {
        d_output[index] =
            d_input[index] * 2.0F + 1.0F;
    }
}