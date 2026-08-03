#include "transpose.h"

#include "common/benchmark_stats.h"
#include "common/compare.h"
#include "common/cuda_check.h"
#include "common/cuda_timer.h"
#include "common/device_buffer.h"
#include "common/init_data.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

enum class TransposeVariant
{
    Naive,
    Tiled,
    Padded
};

const char* variant_name(TransposeVariant variant)
{
    if (variant == TransposeVariant::Naive)
    {
        return "naive";
    }
    if (variant == TransposeVariant::Tiled)
    {
        return "tiled";
    }
    return "padded";
}

void launch_transpose(TransposeVariant variant,
                      const float* d_input,
                      float* d_output,
                      int height,
                      int width)
{
    if (variant == TransposeVariant::Naive)
    {
        transpose_naive_device(d_input, d_output, height, width);
    }
    else if (variant == TransposeVariant::Tiled)
    {
        transpose_tiled_device(d_input, d_output, height, width);
    }
    else
    {
        transpose_padded_device(d_input, d_output, height, width);
    }
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

BenchmarkResult run_benchmark(TransposeVariant variant,
                              int height,
                              int width,
                              int warmup_iterations,
                              int benchmark_iterations)
{
    const std::size_t element_count =
        static_cast<std::size_t>(height) * static_cast<std::size_t>(width);

    const std::size_t bytes = element_count * sizeof(float);

    std::vector<float> h_input(element_count);
    std::vector<float> h_output(element_count);
    std::vector<float> expected(element_count);

    init_random(h_input, 20260730);

    for (int row = 0; row < height; ++row)
    {
        for (int col = 0; col < width; ++col)
        {
            expected[static_cast<std::size_t>(col) * height + row] =
                h_input[static_cast<std::size_t>(row) * width + col];
        }
    }

    DeviceBuffer<float> d_input(element_count);
    DeviceBuffer<float> d_output(element_count);

    CUDA_CHECK(
        cudaMemcpy(d_input.data(), h_input.data(), bytes, cudaMemcpyHostToDevice));

    launch_transpose(variant, d_input.data(), d_output.data(), height, width);

    CUDA_CHECK(
        cudaMemcpy(h_output.data(), d_output.data(), bytes, cudaMemcpyDeviceToHost));

    const CompareResult comparison = compare_results(h_output, expected, 0.0F, 0.0F);

    for (int iteration = 0; iteration < warmup_iterations; ++iteration)
    {
        launch_transpose(variant, d_input.data(), d_output.data(), height, width);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> samples;
    samples.reserve(benchmark_iterations);

    CudaEventTimer timer;

    for (int iteration = 0; iteration < benchmark_iterations; ++iteration)
    {
        timer.start();

        launch_transpose(variant, d_input.data(), d_output.data(), height, width);

        timer.stop();
        samples.push_back(timer.elapsed_ms());
    }

    const float kernel_ms = benchmark_median(samples);

    // One global read and one global write per element.
    const double transferred_bytes = 2.0 * static_cast<double>(bytes);

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
    struct Shape
    {
        int height;
        int width;
    };

    const std::vector<TransposeVariant> variants = {
        TransposeVariant::Naive, TransposeVariant::Tiled, TransposeVariant::Padded};

    const std::vector<Shape> shapes = {
        {3, 5}, {17, 19}, {1003, 769}, {2048, 2048}, {4096, 4096}};

    constexpr int warmup_iterations = 20;
    constexpr int benchmark_iterations = 50;

    bool all_passed = true;

    std::cout << "variant,height,width,elements,warmup,iters,"
              << "kernel_ms_median,GB_per_s,correct,"
              << "max_abs_error,max_rel_error,first_mismatch\n";

    for (const TransposeVariant variant : variants)
    {
        for (const Shape shape : shapes)
        {
            const BenchmarkResult result = run_benchmark(variant,
                                                         shape.height,
                                                         shape.width,
                                                         warmup_iterations,
                                                         benchmark_iterations);

            const std::size_t element_count = static_cast<std::size_t>(shape.height) *
                                              static_cast<std::size_t>(shape.width);

            std::cout << variant_name(variant) << "," << shape.height << ","
                      << shape.width << "," << element_count << "," << warmup_iterations
                      << "," << benchmark_iterations << "," << std::fixed
                      << std::setprecision(6) << result.kernel_ms << ","
                      << std::setprecision(2) << result.bandwidth_gbs << ","
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

    return all_passed ? 0 : 1;
}
