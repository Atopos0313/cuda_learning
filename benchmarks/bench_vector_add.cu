#include "vector_add.h"

#include "common/compare.h"
#include "common/cuda_check.h"
#include "common/cuda_timer.h"
#include "common/device_buffer.h"
#include "common/init_data.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

namespace
{
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
    float e2e_ms = 0.0F;
    double bandwidth_gbs = 0.0;
    bool passed = false;
    std::size_t first_mismatch = std::numeric_limits<std::size_t>::max();
    float max_abs_error = 0.0F;
    float max_rel_error = 0.0F;
};

BenchmarkResult run_benchmark(int n,
                              int threads_per_block,
                              int warmup_iterations,
                              int benchmark_iterations)
{
    const std::size_t element_count = static_cast<std::size_t>(n);
    const std::size_t bytes = element_count * sizeof(float);

    std::vector<float> h_a(element_count);
    std::vector<float> h_b(element_count);
    std::vector<float> h_c(element_count, 0.0F);
    std::vector<float> reference(element_count);

    init_random(h_a, 20260729);
    init_random(h_b, 20260730);

    for (std::size_t index = 0; index < element_count; ++index)
    {
        reference[index] = h_a[index] + h_b[index];
    }

    DeviceBuffer<float> d_a(element_count);
    DeviceBuffer<float> d_b(element_count);
    DeviceBuffer<float> d_c(element_count);

    CUDA_CHECK(cudaMemcpy(d_a.data(), h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b.data(), h_b.data(), bytes, cudaMemcpyHostToDevice));

    // Warmup excludes one-time context/module work from measured samples.
    for (int iteration = 0; iteration < warmup_iterations; ++iteration)
    {
        vector_add_device(d_a.data(), d_b.data(), d_c.data(), n, threads_per_block);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> kernel_samples;
    kernel_samples.reserve(static_cast<std::size_t>(benchmark_iterations));

    CudaEventTimer gpu_timer;
    for (int iteration = 0; iteration < benchmark_iterations; ++iteration)
    {
        gpu_timer.start();
        vector_add_device(d_a.data(), d_b.data(), d_c.data(), n, threads_per_block);
        gpu_timer.stop();
        kernel_samples.push_back(gpu_timer.elapsed_ms());
    }

    std::vector<float> e2e_samples;
    e2e_samples.reserve(static_cast<std::size_t>(benchmark_iterations));

    for (int iteration = 0; iteration < benchmark_iterations; ++iteration)
    {
        const auto start = std::chrono::steady_clock::now();

        CUDA_CHECK(cudaMemcpy(d_a.data(), h_a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b.data(), h_b.data(), bytes, cudaMemcpyHostToDevice));

        vector_add_device(d_a.data(), d_b.data(), d_c.data(), n, threads_per_block);

        CUDA_CHECK(cudaMemcpy(h_c.data(), d_c.data(), bytes, cudaMemcpyDeviceToHost));

        const auto stop = std::chrono::steady_clock::now();
        const float milliseconds =
            std::chrono::duration<float, std::milli>(stop - start).count();
        e2e_samples.push_back(milliseconds);
    }

    const float kernel_median = median(kernel_samples);
    const float e2e_median = median(e2e_samples);

    // Vector add reads A and B and writes C: 3 * n * sizeof(float).
    const double transferred_bytes = 3.0 * static_cast<double>(bytes);
    const double bandwidth_gbs =
        transferred_bytes / (static_cast<double>(kernel_median) * 1.0e6);

    const CompareResult comparison = compare_results(h_c, reference, 1.0e-6F, 1.0e-5F);

    return {kernel_median,
            e2e_median,
            bandwidth_gbs,
            comparison.passed,
            comparison.first_mismatch,
            comparison.max_abs_error,
            comparison.max_rel_error};
}
} // namespace

int main()
{
    const std::vector<int> sizes = {1024, 65536, 1000003, 4194304, 16777217};
    const std::vector<int> block_sizes = {64, 128, 256};

    constexpr int kWarmupIterations = 20;
    constexpr int kBenchmarkIterations = 50;

    bool all_passed = true;

    std::cout << "size,block,warmup,iters,"
              << "kernel_ms_median,e2e_ms_median,GB_per_s,"
              << "correct,max_abs_error,max_rel_error,"
              << "first_mismatch\n";

    for (const int n : sizes)
    {
        for (const int block_size : block_sizes)
        {
            const BenchmarkResult result =
                run_benchmark(n, block_size, kWarmupIterations, kBenchmarkIterations);

            std::cout << n << ',' << block_size << ',' << kWarmupIterations << ','
                      << kBenchmarkIterations << ',' << std::fixed
                      << std::setprecision(6) << result.kernel_ms << ','
                      << result.e2e_ms << ',' << std::setprecision(2)
                      << result.bandwidth_gbs << ','
                      << (result.passed ? "PASS" : "FAIL") << ',' << std::scientific
                      << result.max_abs_error << ',' << result.max_rel_error << ',';

            if (result.passed)
            {
                std::cout << "none";
            }
            else
            {
                std::cout << result.first_mismatch;
            }

            std::cout << '\n';
            all_passed = result.passed && all_passed;
        }
    }

    return all_passed ? 0 : 1;
}
