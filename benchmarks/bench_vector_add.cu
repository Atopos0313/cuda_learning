#include "vector_add.h"
#include "common/cuda_check.h"
#include "common/cuda_timer.h"
#include "common/device_buffer.h"
#include "common/compare.h"
#include "common/init_data.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

float median(std::vector<float> values)
{
    std::sort(values.begin(), values.end());

    const std::size_t middle = values.size() / 2;

    if (values.size() % 2 == 0) {
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
    std::size_t first_mismatch =
        std::numeric_limits<std::size_t>::max();
    float max_abs_error = 0.0F;
    float max_rel_error = 0.0F;
};

BenchmarkResult run_benchmark(
    int N,
    int threads_per_block,
    int warmup_iterations,
    int benchmark_iterations
)
{
    const std::size_t bytes =
        static_cast<std::size_t>(N) * sizeof(float);

    std::vector<float> h_A(N);
    std::vector<float> h_B(N);
    std::vector<float> h_C(N, 0.0f);
    std::vector<float> reference(N);

    init_random(h_A, 20260729);
    init_random(h_B, 20260730);

    for (int index = 0; index < N; ++index) {
        reference[index] = h_A[index] + h_B[index];
    }

    DeviceBuffer<float> d_A(N);
    DeviceBuffer<float> d_B(N);
    DeviceBuffer<float> d_C(N);

    CUDA_CHECK(cudaMemcpy(
        d_A.data(),
        h_A.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_B.data(),
        h_B.data(),
        bytes,
        cudaMemcpyHostToDevice
    ));

    // 预热：排除首次 CUDA Context 初始化、缓存和频率变化的影响。
    for (int iteration = 0;
         iteration < warmup_iterations;
         ++iteration)
    {
        vector_add_device(
            d_A.data(),
            d_B.data(),
            d_C.data(),
            N,
            threads_per_block
        );
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> kernel_samples;
    // reserve() 用于提前给 std::vector 预留内存，避免后续添加元素时频繁扩容。但是此时里面还没有元素，不能使用下标访问
    kernel_samples.reserve(benchmark_iterations);

    CudaEventTimer gpu_timer;

    for (int iteration = 0;
         iteration < benchmark_iterations;
         ++iteration)
    {
        gpu_timer.start();

        vector_add_device(
            d_A.data(),
            d_B.data(),
            d_C.data(),
            N,
            threads_per_block
        );

        gpu_timer.stop();

        kernel_samples.push_back(gpu_timer.elapsed_ms());
    }

    std::vector<float> e2e_samples;
    e2e_samples.reserve(benchmark_iterations);

    for (int iteration = 0;
         iteration < benchmark_iterations;
         ++iteration)
    {
        const auto start = std::chrono::steady_clock::now();

        CUDA_CHECK(cudaMemcpy(
            d_A.data(),
            h_A.data(),
            bytes,
            cudaMemcpyHostToDevice
        ));

        CUDA_CHECK(cudaMemcpy(
            d_B.data(),
            h_B.data(),
            bytes,
            cudaMemcpyHostToDevice
        ));

        vector_add_device(
            d_A.data(),
            d_B.data(),
            d_C.data(),
            N,
            threads_per_block
        );

        CUDA_CHECK(cudaMemcpy(
            h_C.data(),
            d_C.data(),
            bytes,
            cudaMemcpyDeviceToHost
        ));

        const auto stop = std::chrono::steady_clock::now();

        const float milliseconds =
            std::chrono::duration<float, std::milli>(
                stop - start
            ).count();

        e2e_samples.push_back(milliseconds);
    }

    const float kernel_median = median(kernel_samples);
    const float e2e_median = median(e2e_samples);

    // vector add：读取 A、读取 B、写入 C，共移动 3 × N × sizeof(float)。
    const double transferred_bytes =
        3.0 * static_cast<double>(bytes);

    const double bandwidth_gbs =
        transferred_bytes /
        (static_cast<double>(kernel_median) * 1.0e6);

    const CompareResult comparison = compare_results(
        h_C,
        reference,
        1.0e-6F,
        1.0e-5F
    );

    return {
        kernel_median,
        e2e_median,
        bandwidth_gbs,
        comparison.passed,
        comparison.first_mismatch,
        comparison.max_abs_error,
        comparison.max_rel_error
    };
}

int main()
{
    const std::vector<int> sizes = {
        1024,
        65536,
        1000003,
        4194304,
        16777217
    };

    const std::vector<int> block_sizes = {
        64,
        128,
        256
    };

    constexpr int warmup_iterations = 20;
    constexpr int benchmark_iterations = 50;

    bool all_passed = true;

    std::cout
        << "size,block,warmup,iters,"
        << "kernel_ms_median,e2e_ms_median,GB_per_s,"
        << "correct,max_abs_error,max_rel_error,"
        << "first_mismatch\n";

    for (const int N : sizes) {
        for (const int block_size : block_sizes) {
            const BenchmarkResult result = run_benchmark(
                N,
                block_size,
                warmup_iterations,
                benchmark_iterations
            );

            std::cout
                << N << ","
                << block_size << ","
                << warmup_iterations << ","
                << benchmark_iterations << ","
                << std::fixed << std::setprecision(6)
                << result.kernel_ms << ","
                << result.e2e_ms << ","
                << std::setprecision(2)
                << result.bandwidth_gbs << ","
                << (result.passed ? "PASS" : "FAIL") << ","
                << std::scientific
                << result.max_abs_error << ","
                << result.max_rel_error << ",";

            if (result.passed) {
                std::cout << "none";
            }
            else {
                std::cout << result.first_mismatch;
                all_passed = false;
            }

            std::cout << "\n";
        }
    }

    return all_passed ? 0 : 1;
}
