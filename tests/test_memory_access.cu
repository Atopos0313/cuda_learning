#include "common/compare.h"
#include "common/cuda_check.h"
#include "common/device_buffer.h"
#include "common/init_data.h"
#include "memory_access.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <iostream>
#include <vector>
// A scoped enum keeps the three teaching variants type-safe and explicit.
enum class AxpyVariant
{
    Contiguous,
    Strided,
    Float4
};
const char* variant_name(AxpyVariant variant)
{
    if (variant == AxpyVariant::Contiguous)
    {
        return "contiguous";
    }

    if (variant == AxpyVariant::Strided)
    {
        return "strided";
    }

    return "float4";
}

bool run_test(AxpyVariant variant, int n, int threads_per_block)
{
    constexpr float alpha = 1.1F;

    std::vector<float> h_x(n);
    std::vector<float> h_y(n);
    std::vector<float> expected(n);

    init_random(h_x, 20260729);
    init_random(h_y, 20260730);

    // Store the expected result of exactly one AXPY operation.
    for (int index = 0; index < n; ++index)
    {
        expected[index] = alpha * h_x[index] + h_y[index];
    }

    DeviceBuffer<float> d_x(n);
    DeviceBuffer<float> d_y(n);

    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);

    // Empty buffers have no data to copy.
    if (n > 0)
    {
        CUDA_CHECK(cudaMemcpy(d_x.data(), h_x.data(), bytes, cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(d_y.data(), h_y.data(), bytes, cudaMemcpyHostToDevice));
    }

    // A correctness test executes AXPY exactly once.
    // Calling with n == 0 also verifies the early-return path.
    if (variant == AxpyVariant::Contiguous)
    {
        axpy_contiguous_device(d_x.data(), d_y.data(), n, alpha, threads_per_block);
    }
    else if (variant == AxpyVariant::Strided)
    {
        axpy_strided_device(d_x.data(), d_y.data(), n, alpha, threads_per_block);
    }
    else
    {
        axpy_float4_device(d_x.data(), d_y.data(), n, alpha, threads_per_block);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    if (n > 0)
    {
        CUDA_CHECK(cudaMemcpy(h_y.data(), d_y.data(), bytes, cudaMemcpyDeviceToHost));
    }

    const CompareResult result = compare_results(h_y, expected, 1.0e-6F, 1.0e-5F);

    if (!result.passed)
    {
        std::cout << " first_mismatch=" << result.first_mismatch
                  << " max_abs_error=" << result.max_abs_error
                  << " max_rel_error=" << result.max_rel_error;
    }

    return result.passed;
}

int main()
{
    const std::vector<AxpyVariant> variants = {
        AxpyVariant::Contiguous, AxpyVariant::Strided, AxpyVariant::Float4};
    const std::vector<int> sizes = {0, 1, 3, 4, 5, 1000003};

    const std::vector<int> block_sizes = {64, 128, 256};

    bool all_passed = true;

    for (const AxpyVariant variant : variants)
    {
        for (const int n : sizes)
        {
            for (const int block_size : block_sizes)
            {
                const bool passed = run_test(variant, n, block_size);

                std::cout << "variant=" << variant_name(variant) << ", N=" << n
                          << ", block=" << block_size << ": "
                          << (passed ? "PASS" : "FAIL") << '\n';

                if (!passed)
                {
                    all_passed = false;
                }
            }
        }
    }

    // A non-zero exit code lets CTest detect any failed case.
    return all_passed ? 0 : 1;
}
