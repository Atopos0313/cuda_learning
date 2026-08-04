#include <cuda_runtime.h>

__global__ void rmsnorm_f32_kernel(const float* input,
                                   const float* weight,
                                   float* output,
                                   int hidden_size,
                                   float epsilon)
{
    extern __shared__ float shared_sum[];

    const int row = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);

    const float* row_input = input + row * hidden_size;
    float* row_output = output + row * hidden_size;

    // 1. 每个线程计算自己负责元素的平方和。
    float local_sum = 0.0F;

    for (int col = tid; col < hidden_size; col += blockDim.x)
    {
        // 在这里读取 input，并累加 value²。
    }

    // 2. 把每个线程的局部结果放进 shared memory。
    shared_sum[tid] = local_sum;
    __syncthreads();

    // 3. 对 shared memory 做 tree reduction。
    for (int stride = static_cast<int>(blockDim.x) / 2;
         stride > 0;
         stride /= 2)
    {
        if (tid < stride)
        {
            // 合并两个部分和。
        }

        __syncthreads();
    }

    // 4. shared_sum[0] 是整行平方和。
    const float inverse_rms =
        rsqrtf(shared_sum[0] / static_cast<float>(hidden_size) + epsilon);

    // 5. 第二次遍历，完成归一化和 weight 缩放。
    for (int col = tid; col < hidden_size; col += blockDim.x)
    {
        // output = input × inverse_rms × weight
    }
}