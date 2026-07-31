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

double gpu_sum_all_gpu(
    const std::vector<float>& values,
    int threads_per_block,
    bool use_warp_shuffle = false
){
    const int n = static_cast<int>(values.size());
    if(n == 0){
        return 0.0;
    }
    DeviceBuffer<float> d_input(values.size());
    CUDA_CHECK(cudaMemcpy(
        d_input.data(),
        values.data(),
        values.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));
    int first_output_count = reduction_block_count(n, threads_per_block);
    DeviceBuffer<float> d_ping(first_output_count);
    DeviceBuffer<float> d_pong(first_output_count);
    const float* current_input =
        d_input.data();

    float* current_output =
        d_ping.data();

    float* alternate_output =
        d_pong.data();

    int current_count = n;

    while (current_count > 1) {
        const int next_count =
            reduction_block_count(
                current_count,
                threads_per_block
            );

        if (use_warp_shuffle) {
            reduce_sum_blocks_warp_device(
                current_input,
                current_output,
                current_count,
                threads_per_block
            );
        } else {
            reduce_sum_blocks_device(
                current_input,
                current_output,
                current_count,
                threads_per_block
            );
        }

        current_input = current_output;
        current_count = next_count;

        std::swap(
            current_output,
            alternate_output
        );
    }

    float result = 0.0F;

    CUDA_CHECK(cudaMemcpy(
        &result,
        current_input,
        sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    return static_cast<double>(result);

}

float gpu_max_all_gpu(
    const std::vector<float>& values,
    int threads_per_block,
    bool use_warp_shuffle = false
)
{
    const int n =
        static_cast<int>(values.size());

    if (n == 0) {
        return std::numeric_limits<float>::lowest();
    }

    DeviceBuffer<float> d_input(values.size());

    CUDA_CHECK(cudaMemcpy(
        d_input.data(),
        values.data(),
        values.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    const int first_output_count =
        reduction_block_count(
            n,
            threads_per_block
        );

    DeviceBuffer<float> d_ping(
        static_cast<std::size_t>(first_output_count)
    );
    DeviceBuffer<float> d_pong(
        static_cast<std::size_t>(first_output_count)
    );

    const float* current_input =
        d_input.data();
    float* current_output =
        d_ping.data();
    float* alternate_output =
        d_pong.data();
    int current_count = n;

    while (current_count > 1) {
        const int next_count =
            reduction_block_count(
                current_count,
                threads_per_block
            );

        if (use_warp_shuffle) {
            reduce_max_blocks_warp_device(
                current_input,
                current_output,
                current_count,
                threads_per_block
            );
        } else {
            reduce_max_blocks_device(
                current_input,
                current_output,
                current_count,
                threads_per_block
            );
        }

        current_input = current_output;
        current_count = next_count;

        std::swap(
            current_output,
            alternate_output
        );
    }

    float result = 0.0F;

    CUDA_CHECK(cudaMemcpy(
        &result,
        current_input,
        sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    return result;
}

int reduction_round_count(
    int n,
    int threads_per_block
)
{
    int rounds = 0;

    while (n > 1) {
        n = reduction_block_count(
            n,
            threads_per_block
        );

        ++rounds;
    }

    return rounds;
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
    const int n =
        static_cast<int>(values.size());

    const double expected =
        cpu_sum_reference(values);

    const double first_stage_sum =
        gpu_sum_first_stage(
            values,
            threads_per_block
        );

    const double all_gpu_sum =
        gpu_sum_all_gpu(
            values,
            threads_per_block
        );
    const double warp_sum =
        gpu_sum_all_gpu(
            values,
            threads_per_block,
            true
        );

    const double first_stage_error =
        std::abs(first_stage_sum - expected);

    const double all_gpu_error =
        std::abs(all_gpu_sum - expected);
    const double warp_error =
        std::abs(warp_sum - expected);

    const bool first_stage_passed =
        sum_is_close(
            first_stage_sum,
            expected
        );

    const bool all_gpu_passed =
        sum_is_close(
            all_gpu_sum,
            expected
        );
    const bool warp_passed =
        sum_is_close(
            warp_sum,
            expected
        );

    const bool passed =
        first_stage_passed &&
        all_gpu_passed &&
        warp_passed;

    std::cout
        << "case=" << case_name
        << ", N=" << n
        << ", block=" << threads_per_block
        << ", first_partials="
        << reduction_block_count(
               n,
               threads_per_block
           )
        << ", rounds="
        << reduction_round_count(
               n,
               threads_per_block
           )
        << ", cpu_sum=" << expected
        << ", first_stage_sum="
        << first_stage_sum
        << ", all_gpu_sum="
        << all_gpu_sum
        << ", warp_sum="
        << warp_sum
        << ", first_stage_error="
        << first_stage_error
        << ", all_gpu_error="
        << all_gpu_error
        << ", warp_error="
        << warp_error
        << ", status="
        << (passed ? "PASS" : "FAIL")
        << '\n';

    return passed;
}

bool run_max_case(
    const std::string& case_name,
    const std::vector<float>& values,
    int threads_per_block
)
{
    const int n =
        static_cast<int>(values.size());
    const float expected =
        cpu_max_reference(values);
    const float shared_actual =
        gpu_max_all_gpu(
            values,
            threads_per_block
        );
    const float warp_actual =
        gpu_max_all_gpu(
            values,
            threads_per_block,
            true
        );
    const bool passed =
        shared_actual == expected &&
        warp_actual == expected;

    std::cout
        << "operation=max"
        << ", case=" << case_name
        << ", N=" << n
        << ", block=" << threads_per_block
        << ", first_partials="
        << reduction_block_count(
               n,
               threads_per_block
           )
        << ", rounds="
        << reduction_round_count(
               n,
               threads_per_block
           )
        << ", cpu_max=" << expected
        << ", shared_max=" << shared_actual
        << ", warp_max=" << warp_actual
        << ", status="
        << (passed ? "PASS" : "FAIL")
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
    32,
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

            all_passed =
                run_max_case(
                    "random",
                    random_values,
                    block_size
                ) && all_passed;

            all_passed =
                run_max_case(
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
        const double first_stage_sum =
            gpu_sum_first_stage(
                extreme_values,
                block_size
            );
        const double all_gpu_sum =
            gpu_sum_all_gpu(
                extreme_values,
                block_size
            );
        const double warp_sum =
            gpu_sum_all_gpu(
                extreme_values,
                block_size,
                true
            );

        std::cout
            << "case=extreme"
            << ", N=" << extreme_values.size()
            << ", block=" << block_size
            << ", cpu_sum=" << cpu_sum
            << ", first_stage_sum="
            << first_stage_sum
            << ", all_gpu_sum="
            << all_gpu_sum
            << ", warp_sum="
            << warp_sum
            << ", first_stage_error="
            << std::abs(first_stage_sum - cpu_sum)
            << ", all_gpu_error="
            << std::abs(all_gpu_sum - cpu_sum)
            << ", warp_error="
            << std::abs(warp_sum - cpu_sum)
            << ", max="
            << cpu_max_reference(extreme_values)
            << ", status=OBSERVE"
            << '\n';
    }

    return all_passed ? 0 : 1;
}
