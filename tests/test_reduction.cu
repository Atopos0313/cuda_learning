#include "common/cuda_check.h"
#include "common/device_buffer.h"
#include "reduction.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <iostream>
#include <random>
#include <string>
#include <vector>

std::vector<float> make_random_data(
    int n,
    int seed,
    float lower,
    float upper
)
{
    std::mt19937 generator(seed);

    std::uniform_real_distribution<float> distribution(
        lower,
        upper
    );

    std::vector<float> values(n);

    for (float& value : values) {
        value = distribution(generator);
    }

    return values;
}


double cpu_sum_reference(
    const std::vector<float>& values
)
{
    double result = 0.0;

    for (const float value : values) {
        result += static_cast<double>(value);
    }

    return result;
}
float cpu_max_reference(
    const std::vector<float>& values
)
{
    float result =
        std::numeric_limits<float>::lowest();

    for (const float value : values) {
        result = std::max(result, value);
    }

    return result;
}

double gpu_sum_first_stage(
    const std::vector<float>& values,
    int threads_per_block
)
{
    const int n = static_cast<int>(values.size());
    const int block_count =
        reduction_block_count(n, threads_per_block);

    if (block_count == 0) {
        reduce_sum_blocks_device(
            nullptr,
            nullptr,
            0,
            threads_per_block
        );
        return 0.0;
    }

    DeviceBuffer<float> d_input(values.size());
    DeviceBuffer<float> d_block_sums(
        static_cast<std::size_t>(block_count)
    );

    CUDA_CHECK(cudaMemcpy(
        d_input.data(),
        values.data(),
        values.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    reduce_sum_blocks_device(
        d_input.data(),
        d_block_sums.data(),
        n,
        threads_per_block
    );

    std::vector<float> block_sums(
        static_cast<std::size_t>(block_count)
    );

    CUDA_CHECK(cudaMemcpy(
        block_sums.data(),
        d_block_sums.data(),
        block_sums.size() * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    double result = 0.0;

    for (const float block_sum : block_sums) {
        result += static_cast<double>(block_sum);
    }

    return result;
}

bool sum_is_close(
    double actual,
    double expected
)
{
    constexpr double absolute_tolerance = 1.0e-4;
    constexpr double relative_tolerance = 1.0e-6;

    const double allowed_error =
        absolute_tolerance +
        relative_tolerance * std::abs(expected);

    return std::abs(actual - expected) <= allowed_error;
}

bool run_sum_case(
    const std::string& case_name,
    const std::vector<float>& values,
    int threads_per_block
)
{
    const double expected = cpu_sum_reference(values);
    const double actual =
        gpu_sum_first_stage(values, threads_per_block);
    const double absolute_error =
        std::abs(actual - expected);
    const bool passed =
        sum_is_close(actual, expected);

    std::cout
        << "case=" << case_name
        << ", N=" << values.size()
        << ", block=" << threads_per_block
        << ", partials="
        << reduction_block_count(
               static_cast<int>(values.size()),
               threads_per_block
           )
        << ", cpu_sum=" << expected
        << ", gpu_sum=" << actual
        << ", abs_error=" << absolute_error
        << ", status=" << (passed ? "PASS" : "FAIL")
        << '\n';

    return passed;
}

const std::vector<int> sizes = {
    0,
    1,
    31,
    32,
    33,
    100003
};

const std::vector<int> block_sizes = {
    64,
    128,
    256
};

int main()
{
    std::cout << std::setprecision(10);

    bool all_passed = true;

    for (const int n : sizes) {
        const std::vector<float> random_values =
            make_random_data(
                n,
                20260730 + n,
                -1.0F,
                1.0F
            );

        const std::vector<float> negative_values =
            make_random_data(
                n,
                20260731 + n,
                -10.0F,
                -0.1F
            );

        for (const int block_size : block_sizes) {
            all_passed =
                run_sum_case(
                    "random",
                    random_values,
                    block_size
                ) && all_passed;

            all_passed =
                run_sum_case(
                    "negative",
                    negative_values,
                    block_size
                ) && all_passed;
        }
    }

    const std::vector<float> extreme_values = {
        1.0e20F,
        -1.0e20F,
        1.0F,
        -2.0F
    };

    for (const int block_size : block_sizes) {
        const double cpu_sum =
            cpu_sum_reference(extreme_values);
        const double gpu_sum =
            gpu_sum_first_stage(
                extreme_values,
                block_size
            );

        std::cout
            << "case=extreme"
            << ", N=" << extreme_values.size()
            << ", block=" << block_size
            << ", cpu_sum=" << cpu_sum
            << ", gpu_sum=" << gpu_sum
            << ", max="
            << cpu_max_reference(extreme_values)
            << ", status=OBSERVE"
            << '\n';
    }

    return all_passed ? 0 : 1;
}
