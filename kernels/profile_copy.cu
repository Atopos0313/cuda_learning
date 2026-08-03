#include "common/cuda_check.h"
#include "common/device_buffer.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <exception>
#include <iostream>
#include <vector>

namespace
{
constexpr int kMatrixExtent = 4096;
constexpr int kElementCount = kMatrixExtent * kMatrixExtent;
constexpr int kThreadsPerBlock = 256;
} // namespace

__global__ void copy_contiguous_kernel(const float* input, float* output, int n)
{
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    if (index < n)
    {
        output[index] = input[index];
    }
}

int main()
{
    try
    {
        std::vector<float> input(static_cast<std::size_t>(kElementCount));
        std::vector<float> output(static_cast<std::size_t>(kElementCount));

        for (std::size_t index = 0; index < input.size(); ++index)
        {
            const int pattern = static_cast<int>(index % 2048) - 1024;
            input[index] = static_cast<float>(pattern);
        }

        DeviceBuffer<float> d_input(input.size());
        DeviceBuffer<float> d_output(output.size());

        const std::size_t bytes = input.size() * sizeof(float);

        CUDA_CHECK(
            cudaMemcpy(d_input.data(), input.data(), bytes, cudaMemcpyHostToDevice));

        const int blocks = (kElementCount + kThreadsPerBlock - 1) / kThreadsPerBlock;

        copy_contiguous_kernel<<<blocks, kThreadsPerBlock>>>(
            d_input.data(), d_output.data(), kElementCount);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(
            cudaMemcpy(output.data(), d_output.data(), bytes, cudaMemcpyDeviceToHost));

        std::size_t mismatch_count = 0;
        std::size_t first_mismatch = output.size();

        for (std::size_t index = 0; index < output.size(); ++index)
        {
            if (output[index] != input[index])
            {
                if (mismatch_count == 0)
                {
                    first_mismatch = index;
                }

                ++mismatch_count;
            }
        }

        const bool passed = mismatch_count == 0;

        std::cout << "case=copy"
                  << ",N=" << kElementCount << ",block=" << kThreadsPerBlock
                  << ",grid=" << blocks << ",bytes=" << 2 * bytes
                  << ",mismatches=" << mismatch_count;

        if (!passed)
        {
            std::cout << ",first_mismatch=" << first_mismatch
                      << ",expected=" << input[first_mismatch]
                      << ",actual=" << output[first_mismatch];
        }

        std::cout << ",status=" << (passed ? "PASS" : "FAIL") << '\n';

        return passed ? 0 : 1;
    }
    catch (const std::exception& error)
    {
        std::cerr << "error=" << error.what() << '\n';
        return 1;
    }
}
