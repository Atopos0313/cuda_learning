#include "reduction.h"

#include "common/benchmark_stats.h"
#include "common/cuda_check.h"
#include "common/cuda_timer.h"
#include "common/device_buffer.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

enum class ReductionOperation
{
    Sum,
    Max
};

enum class ReductionVariant
{
    Shared,
    Warp
};

const char* operation_name(ReductionOperation operation)
{
    return operation == ReductionOperation::Sum ? "sum" : "max";
}

const char* variant_name(ReductionVariant variant)
{
    return variant == ReductionVariant::Shared ? "shared" : "warp";
}

int reduction_rounds(int n, int threads_per_block)
{
    int rounds = 0;

    while (n > 1)
    {
        n = reduction_block_count(n, threads_per_block);
        ++rounds;
    }

    return rounds;
}

double effective_global_bytes(int n, int threads_per_block)
{
    double bytes = 0.0;

    while (n > 1)
    {
        const int next_count = reduction_block_count(n, threads_per_block);

        bytes += static_cast<double>(n + next_count) * sizeof(float);

        n = next_count;
    }

    return bytes;
}

void launch_blocks(ReductionOperation operation,
                   ReductionVariant variant,
                   const float* d_input,
                   float* d_output,
                   int n,
                   int threads_per_block)
{
    if (operation == ReductionOperation::Sum)
    {
        if (variant == ReductionVariant::Shared)
        {
            reduce_sum_blocks_device(d_input, d_output, n, threads_per_block);
        }
        else
        {
            reduce_sum_blocks_warp_device(d_input, d_output, n, threads_per_block);
        }
    }
    else
    {
        if (variant == ReductionVariant::Shared)
        {
            reduce_max_blocks_device(d_input, d_output, n, threads_per_block);
        }
        else
        {
            reduce_max_blocks_warp_device(d_input, d_output, n, threads_per_block);
        }
    }
}

const float* launch_pipeline(ReductionOperation operation,
                             ReductionVariant variant,
                             const float* d_input,
                             float* d_ping,
                             float* d_pong,
                             int n,
                             int threads_per_block)
{
    const float* current_input = d_input;
    float* current_output = d_ping;
    float* alternate_output = d_pong;
    int current_count = n;

    while (current_count > 1)
    {
        launch_blocks(operation,
                      variant,
                      current_input,
                      current_output,
                      current_count,
                      threads_per_block);

        current_count = reduction_block_count(current_count, threads_per_block);
        current_input = current_output;

        std::swap(current_output, alternate_output);
    }

    return current_input;
}

std::vector<float> make_input(int n)
{
    std::mt19937 generator(20260731 + n);
    std::uniform_real_distribution<float> distribution(-1.0F, 1.0F);

    std::vector<float> values(static_cast<std::size_t>(n));

    for (float& value : values)
    {
        value = distribution(generator);
    }

    return values;
}

double cpu_reference(ReductionOperation operation, const std::vector<float>& values)
{
    if (operation == ReductionOperation::Sum)
    {
        double result = 0.0;

        for (const float value : values)
        {
            result += static_cast<double>(value);
        }

        return result;
    }

    float result = std::numeric_limits<float>::lowest();

    for (const float value : values)
    {
        result = std::max(result, value);
    }

    return static_cast<double>(result);
}

bool result_is_correct(ReductionOperation operation, float actual, double expected)
{
    if (operation == ReductionOperation::Max)
    {
        return actual == static_cast<float>(expected);
    }

    constexpr double absolute_tolerance = 1.0e-4;
    constexpr double relative_tolerance = 1.0e-6;

    const double allowed_error =
        absolute_tolerance + relative_tolerance * std::abs(expected);

    return std::abs(static_cast<double>(actual) - expected) <= allowed_error;
}

struct BenchmarkResult
{
    float pipeline_ms = 0.0F;
    double million_elements_per_second = 0.0;
    double effective_bandwidth_gbs = 0.0;
    double reference = 0.0;
    float result = 0.0F;
    double absolute_error = 0.0;
    bool passed = false;
};

BenchmarkResult run_benchmark(ReductionOperation operation,
                              ReductionVariant variant,
                              int n,
                              int threads_per_block,
                              int warmup_iterations,
                              int benchmark_iterations)
{
    const std::vector<float> h_input = make_input(n);
    const double reference = cpu_reference(operation, h_input);

    DeviceBuffer<float> d_input(static_cast<std::size_t>(n));

    const int first_output_count = reduction_block_count(n, threads_per_block);

    DeviceBuffer<float> d_ping(static_cast<std::size_t>(first_output_count));
    DeviceBuffer<float> d_pong(static_cast<std::size_t>(first_output_count));

    CUDA_CHECK(cudaMemcpy(d_input.data(),
                          h_input.data(),
                          h_input.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    const float* d_result = launch_pipeline(operation,
                                            variant,
                                            d_input.data(),
                                            d_ping.data(),
                                            d_pong.data(),
                                            n,
                                            threads_per_block);

    float result = 0.0F;

    CUDA_CHECK(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

    const bool passed = result_is_correct(operation, result, reference);

    for (int iteration = 0; iteration < warmup_iterations; ++iteration)
    {
        launch_pipeline(operation,
                        variant,
                        d_input.data(),
                        d_ping.data(),
                        d_pong.data(),
                        n,
                        threads_per_block);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> samples;
    samples.reserve(static_cast<std::size_t>(benchmark_iterations));

    CudaEventTimer timer;

    for (int iteration = 0; iteration < benchmark_iterations; ++iteration)
    {
        timer.start();

        launch_pipeline(operation,
                        variant,
                        d_input.data(),
                        d_ping.data(),
                        d_pong.data(),
                        n,
                        threads_per_block);

        timer.stop();
        samples.push_back(timer.elapsed_ms());
    }

    const float pipeline_ms = benchmark_median(samples);

    const double million_elements_per_second =
        static_cast<double>(n) / (static_cast<double>(pipeline_ms) * 1.0e3);

    const double effective_bandwidth_gbs =
        effective_global_bytes(n, threads_per_block) /
        (static_cast<double>(pipeline_ms) * 1.0e6);

    return {pipeline_ms,
            million_elements_per_second,
            effective_bandwidth_gbs,
            reference,
            result,
            std::abs(static_cast<double>(result) - reference),
            passed};
}

void print_result(ReductionOperation operation,
                  ReductionVariant variant,
                  int n,
                  int threads_per_block,
                  int warmup_iterations,
                  int benchmark_iterations,
                  const BenchmarkResult& result,
                  double shared_pipeline_ms)
{
    const double speedup = shared_pipeline_ms / static_cast<double>(result.pipeline_ms);

    std::cout << operation_name(operation) << "," << variant_name(variant) << "," << n
              << "," << threads_per_block << ","
              << reduction_rounds(n, threads_per_block) << "," << warmup_iterations
              << "," << benchmark_iterations << "," << std::fixed
              << std::setprecision(6) << result.pipeline_ms << ","
              << std::setprecision(3) << speedup << ","
              << result.million_elements_per_second << ","
              << result.effective_bandwidth_gbs << "," << std::setprecision(9)
              << result.reference << "," << static_cast<double>(result.result) << ","
              << std::scientific << result.absolute_error << ","
              << (result.passed ? "PASS" : "FAIL") << '\n';
}

int main()
{
    const std::vector<ReductionOperation> operations = {ReductionOperation::Sum,
                                                        ReductionOperation::Max};

    const std::vector<int> sizes = {1003, 100003, 1048576, 16777216};

    const std::vector<int> block_sizes = {32, 64, 128, 256};

    constexpr int warmup_iterations = 20;
    constexpr int benchmark_iterations = 50;

    bool all_passed = true;

    std::cout << "operation,variant,N,block,rounds,"
              << "warmup,iters,pipeline_ms_median,"
              << "speedup_vs_shared,M_elements_per_s,"
              << "effective_GB_per_s,reference,result,"
              << "abs_error,correct\n";

    for (const ReductionOperation operation : operations)
    {
        for (const int n : sizes)
        {
            for (const int block_size : block_sizes)
            {
                const BenchmarkResult shared = run_benchmark(operation,
                                                             ReductionVariant::Shared,
                                                             n,
                                                             block_size,
                                                             warmup_iterations,
                                                             benchmark_iterations);

                const BenchmarkResult warp = run_benchmark(operation,
                                                           ReductionVariant::Warp,
                                                           n,
                                                           block_size,
                                                           warmup_iterations,
                                                           benchmark_iterations);

                print_result(operation,
                             ReductionVariant::Shared,
                             n,
                             block_size,
                             warmup_iterations,
                             benchmark_iterations,
                             shared,
                             shared.pipeline_ms);

                print_result(operation,
                             ReductionVariant::Warp,
                             n,
                             block_size,
                             warmup_iterations,
                             benchmark_iterations,
                             warp,
                             shared.pipeline_ms);

                all_passed = shared.passed && warp.passed && all_passed;
            }
        }
    }

    return all_passed ? 0 : 1;
}
