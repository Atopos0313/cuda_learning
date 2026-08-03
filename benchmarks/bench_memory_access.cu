#include "common/compare.h"
#include "common/cuda_check.h"
#include "common/cuda_timer.h"
#include "common/device_buffer.h"
#include "common/init_data.h"
#include "memory_access.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

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

void launch_axpy(AxpyVariant variant,
                 const float* d_x,
                 float* d_y,
                 int n,
                 float alpha,
                 int threads_per_block)
{
    if (variant == AxpyVariant::Contiguous)
    {
        axpy_contiguous_device(d_x, d_y, n, alpha, threads_per_block);
    }
    else if (variant == AxpyVariant::Strided)
    {
        axpy_strided_device(d_x, d_y, n, alpha, threads_per_block);
    }
    else
    {
        axpy_float4_device(d_x, d_y, n, alpha, threads_per_block);
    }
}

float median(std::vector<float> values)
{
    std::sort(values.begin(), values.end());

    const std::size_t middle = values.size() / 2;

    if (values.size() % 2 == 0)
    {
        return (values[middle - 1] + values[middle]) * 0.5F;
    }

    return values[middle];
}

struct BenchmarkResult
{
    float kernel_ms = 0.0F;
    double bandwidth_gbs = 0.0;
    bool passed = false;
    std::size_t first_mismatch = std::numeric_limits<std::size_t>::max();
    float max_abs_error = 0.0F;
    float max_rel_error = 0.0F;
};

BenchmarkResult run_benchmark(AxpyVariant variant,
                              int n,
                              float alpha,
                              int threads_per_block,
                              int warmup_iterations,
                              int benchmark_iterations)
{
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);

    std::vector<float> h_x(n);
    std::vector<float> h_y(n);
    std::vector<float> h_output(n);
    std::vector<float> expected(n);

    init_random(h_x, 20260729);
    init_random(h_y, 20260730);

    for (int index = 0; index < n; ++index)
    {
        expected[index] = alpha * h_x[index] + h_y[index];
    }

    DeviceBuffer<float> d_x(n);
    DeviceBuffer<float> d_y(n);

    CUDA_CHECK(cudaMemcpy(d_x.data(), h_x.data(), bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_y.data(), h_y.data(), bytes, cudaMemcpyHostToDevice));

    // Check one AXPY operation independently from the timed loop.
    launch_axpy(variant, d_x.data(), d_y.data(), n, alpha, threads_per_block);

    CUDA_CHECK(cudaMemcpy(h_output.data(), d_y.data(), bytes, cudaMemcpyDeviceToHost));

    const CompareResult comparison =
        compare_results(h_output, expected, 1.0e-6F, 1.0e-5F);

    // Restore y before warmup so both variants start from the same data.
    CUDA_CHECK(cudaMemcpy(d_y.data(), h_y.data(), bytes, cudaMemcpyHostToDevice));

    for (int iteration = 0; iteration < warmup_iterations; ++iteration)
    {
        launch_axpy(variant, d_x.data(), d_y.data(), n, alpha, threads_per_block);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> samples;
    samples.reserve(benchmark_iterations);

    CudaEventTimer timer;

    for (int iteration = 0; iteration < benchmark_iterations; ++iteration)
    {
        timer.start();

        launch_axpy(variant, d_x.data(), d_y.data(), n, alpha, threads_per_block);

        timer.stop();
        samples.push_back(timer.elapsed_ms());
    }

    const float kernel_ms = median(samples);

    // AXPY reads x and y, then writes y: 3 * n * sizeof(float).
    const double transferred_bytes = 3.0 * static_cast<double>(bytes);

    const double bandwidth_gbs =
        transferred_bytes / (static_cast<double>(kernel_ms) * 1.0e6);

    return {kernel_ms,
            bandwidth_gbs,
            comparison.passed,
            comparison.first_mismatch,
            comparison.max_abs_error,
            comparison.max_rel_error};
}

int main()
{
    const std::vector<AxpyVariant> variants = {
        AxpyVariant::Contiguous, AxpyVariant::Strided, AxpyVariant::Float4};

    const std::vector<int> sizes = {1000003, 4194304, 16777217};

    const std::vector<int> block_sizes = {64, 128, 256};

    constexpr float alpha = 1.1F;
    constexpr int warmup_iterations = 20;
    constexpr int benchmark_iterations = 50;

    bool all_passed = true;

    std::cout << "variant,size,block,warmup,iters,"
              << "kernel_ms_median,GB_per_s,correct,"
              << "max_abs_error,max_rel_error,first_mismatch\n";

    for (const AxpyVariant variant : variants)
    {
        for (const int n : sizes)
        {
            for (const int block_size : block_sizes)
            {
                const BenchmarkResult result = run_benchmark(variant,
                                                             n,
                                                             alpha,
                                                             block_size,
                                                             warmup_iterations,
                                                             benchmark_iterations);

                std::cout << variant_name(variant) << "," << n << "," << block_size
                          << "," << warmup_iterations << "," << benchmark_iterations
                          << "," << std::fixed << std::setprecision(6)
                          << result.kernel_ms << "," << std::setprecision(2)
                          << result.bandwidth_gbs << ","
                          << (result.passed ? "PASS" : "FAIL") << "," << std::scientific
                          << result.max_abs_error << "," << result.max_rel_error << ",";

                if (result.passed)
                {
                    std::cout << "none";
                }
                else
                {
                    std::cout << result.first_mismatch;
                    all_passed = false;
                }

                std::cout << '\n';
            }
        }
    }

    return all_passed ? 0 : 1;
}
