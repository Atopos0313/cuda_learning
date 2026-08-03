#include "stream_pipeline.h"

#include "common/benchmark_stats.h"
#include "common/compare.h"
#include "common/cuda_check.h"
#include "common/cuda_event.h"
#include "common/cuda_stream.h"
#include "common/device_buffer.h"
#include "common/init_data.h"
#include "common/pinned_buffer.h"

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

namespace
{
struct BenchmarkResult
{
    double e2e_ms = 0.0;
    std::size_t chunk_count = 0;
    CompareResult comparison;
};

void submit_pipeline(const float* h_input,
                     float* h_output,
                     std::size_t total_count,
                     std::size_t chunk_size,
                     int threads_per_block,
                     const std::vector<CudaStream>& streams,
                     std::vector<DeviceBuffer<float>>& chunk_inputs,
                     std::vector<DeviceBuffer<float>>& chunk_outputs,
                     const std::vector<CudaCompletionEvent>& completion_events)
{
    const std::size_t chunk_count =
        (total_count + chunk_size - 1) / chunk_size;

    for (std::size_t chunk_id = 0; chunk_id < chunk_count; ++chunk_id)
    {
        const std::size_t offset = chunk_id * chunk_size;
        const std::size_t remaining = total_count - offset;
        const std::size_t current_count = std::min(chunk_size, remaining);
        const std::size_t current_bytes = current_count * sizeof(float);
        const std::size_t stream_id = chunk_id % streams.size();
        const int chunk_grid = static_cast<int>(
            (current_count + threads_per_block - 1) /
            threads_per_block);

        CUDA_CHECK(cudaMemcpyAsync(chunk_inputs[stream_id].data(),
                                   h_input + offset,
                                   current_bytes,
                                   cudaMemcpyHostToDevice,
                                   streams[stream_id].get()));

        simpleKernel<<<chunk_grid,
                       threads_per_block,
                       0,
                       streams[stream_id].get()>>>(
            chunk_inputs[stream_id].data(),
            chunk_outputs[stream_id].data(),
            static_cast<int>(current_count));
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(h_output + offset,
                                   chunk_outputs[stream_id].data(),
                                   current_bytes,
                                   cudaMemcpyDeviceToHost,
                                   streams[stream_id].get()));
    }

    // One event per stream creates one completion point for the whole pipeline.
    // Stream ordering guarantees that each event follows every chunk assigned
    // to that stream.
    for (std::size_t stream_id = 0; stream_id < streams.size(); ++stream_id)
    {
        completion_events[stream_id].record(streams[stream_id].get());
    }
}

void wait_for_pipeline(
    const std::vector<CudaCompletionEvent>& completion_events)
{
    for (const CudaCompletionEvent& event : completion_events)
    {
        event.synchronize();
    }
}

BenchmarkResult run_benchmark(const float* h_input,
                              float* h_output,
                              const std::vector<float>& reference,
                              int stream_count,
                              std::size_t chunk_size,
                              int threads_per_block,
                              int warmup_iterations,
                              int benchmark_iterations)
{
    std::vector<CudaStream> streams;
    std::vector<CudaCompletionEvent> completion_events;
    std::vector<DeviceBuffer<float>> chunk_inputs;
    std::vector<DeviceBuffer<float>> chunk_outputs;

    // 提前预留出空间，避免空间不足发生不必要的移动，但是这个操作可有可无
    streams.reserve(static_cast<std::size_t>(stream_count));
    completion_events.reserve(static_cast<std::size_t>(stream_count));
    chunk_inputs.reserve(static_cast<std::size_t>(stream_count));
    chunk_outputs.reserve(static_cast<std::size_t>(stream_count));

    for (int stream_id = 0; stream_id < stream_count; ++stream_id)
    {
        streams.emplace_back();
        completion_events.emplace_back();
        chunk_inputs.emplace_back(chunk_size);
        chunk_outputs.emplace_back(chunk_size);
    }

    const std::size_t total_count = reference.size();
    const std::size_t chunk_count =
        (total_count + chunk_size - 1) / chunk_size;

    std::fill(h_output, h_output + total_count, -1.0F);
    submit_pipeline(h_input,
                    h_output,
                    total_count,
                    chunk_size,
                    threads_per_block,
                    streams,
                    chunk_inputs,
                    chunk_outputs,
                    completion_events);
    wait_for_pipeline(completion_events);

    const std::vector<float> actual(h_output, h_output + total_count);
    const CompareResult comparison =
        compare_results(actual, reference, 1.0e-6F, 1.0e-5F);

    for (int iteration = 0; iteration < warmup_iterations; ++iteration)
    {
        submit_pipeline(h_input,
                        h_output,
                        total_count,
                        chunk_size,
                        threads_per_block,
                        streams,
                        chunk_inputs,
                        chunk_outputs,
                        completion_events);
        wait_for_pipeline(completion_events);
    }

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(benchmark_iterations));

    for (int iteration = 0; iteration < benchmark_iterations; ++iteration)
    {
        const auto start = std::chrono::steady_clock::now();

        submit_pipeline(h_input,
                        h_output,
                        total_count,
                        chunk_size,
                        threads_per_block,
                        streams,
                        chunk_inputs,
                        chunk_outputs,
                        completion_events);
        wait_for_pipeline(completion_events);

        const auto stop = std::chrono::steady_clock::now();
        samples.push_back(
            std::chrono::duration<double, std::milli>(stop - start).count());
    }

    return {benchmark_median(samples), chunk_count, comparison};
}

CompareResult run_profile_case(const float* h_input,
                               float* h_output,
                               const std::vector<float>& reference,
                               int stream_count,
                               std::size_t chunk_size,
                               int threads_per_block)
{
    std::vector<CudaStream> streams;
    std::vector<CudaCompletionEvent> completion_events;
    std::vector<DeviceBuffer<float>> chunk_inputs;
    std::vector<DeviceBuffer<float>> chunk_outputs;

    const std::size_t resource_count =
        static_cast<std::size_t>(stream_count);
    streams.reserve(resource_count);
    completion_events.reserve(resource_count);
    chunk_inputs.reserve(resource_count);
    chunk_outputs.reserve(resource_count);

    for (int stream_id = 0; stream_id < stream_count; ++stream_id)
    {
        streams.emplace_back();
        completion_events.emplace_back();
        chunk_inputs.emplace_back(chunk_size);
        chunk_outputs.emplace_back(chunk_size);
    }

    const std::size_t total_count = reference.size();

    // Initialize the CUDA context and modules before collection starts.
    submit_pipeline(h_input,
                    h_output,
                    total_count,
                    chunk_size,
                    threads_per_block,
                    streams,
                    chunk_inputs,
                    chunk_outputs,
                    completion_events);
    wait_for_pipeline(completion_events);

    std::fill(h_output, h_output + total_count, -1.0F);

    CUDA_CHECK(cudaProfilerStart());
    submit_pipeline(h_input,
                    h_output,
                    total_count,
                    chunk_size,
                    threads_per_block,
                    streams,
                    chunk_inputs,
                    chunk_outputs,
                    completion_events);
    wait_for_pipeline(completion_events);
    CUDA_CHECK(cudaProfilerStop());

    const std::vector<float> actual(h_output, h_output + total_count);
    return compare_results(actual, reference, 1.0e-6F, 1.0e-5F);
}
} // namespace

int main(int argc, char* argv[])
{
    constexpr int n = 16'777'219;
    constexpr int threads_per_block = 256;
    constexpr int warmup_iterations = 20;
    constexpr int benchmark_iterations = 50;

    const std::array<int, 3> stream_counts{1, 2, 4};
    const std::array<std::size_t, 3> chunk_sizes{
        65'536,
        262'144,
        1'048'576};

    const std::size_t total_count = static_cast<std::size_t>(n);
    std::vector<float> pageable_input(total_count);
    std::vector<float> reference(total_count);
    PinnedBuffer<float> h_input(total_count);
    PinnedBuffer<float> h_output(total_count);

    init_random(pageable_input, 20260803);
    std::copy(pageable_input.begin(), pageable_input.end(), h_input.data());
    for (std::size_t index = 0; index < total_count; ++index)
    {
        reference[index] = pageable_input[index] * 2.0F + 1.0F;
    }

    const bool profile_mode =
        argc == 2 && std::string_view(argv[1]) == "--profile";

    if (profile_mode)
    {
        constexpr int profile_stream_count = 4;
        constexpr std::size_t profile_chunk_size = 1'048'576;

        const CompareResult comparison =
            run_profile_case(h_input.data(),
                             h_output.data(),
                             reference,
                             profile_stream_count,
                             profile_chunk_size,
                             threads_per_block);

        std::cout << "mode=profile_stream_pipeline"
                  << ", streams=" << profile_stream_count
                  << ", chunk_size=" << profile_chunk_size
                  << ", size=" << n
                  << ", correct="
                  << (comparison.passed ? "PASS" : "FAIL")
                  << ", max_abs_error=" << comparison.max_abs_error
                  << ", max_rel_error=" << comparison.max_rel_error
                  << '\n';

        return comparison.passed ? 0 : 1;
    }

    bool all_passed = true;

    std::cout
        << "streams,chunk_size,N,block,warmup,iters,chunks,"
        << "e2e_ms_median,speedup_vs_1stream,correct,"
        << "max_abs_error,max_rel_error,first_mismatch\n";

    for (const std::size_t chunk_size : chunk_sizes)
    {
        double one_stream_ms = 0.0;

        for (const int stream_count : stream_counts)
        {
            const BenchmarkResult result =
                run_benchmark(h_input.data(),
                              h_output.data(),
                              reference,
                              stream_count,
                              chunk_size,
                              threads_per_block,
                              warmup_iterations,
                              benchmark_iterations);

            if (stream_count == 1)
            {
                one_stream_ms = result.e2e_ms;
            }

            const double speedup = one_stream_ms / result.e2e_ms;
            /*
            std::fixed 让浮点数用普通小数形式输出
            std::setprecision(3)设置浮点数的显示精度。
            当它和 std::fixed 一起使用时，表示“小数点后保留几位”：四舍五入
            std::scientific
            让浮点数使用科学计数法输出
            */
            std::cout << stream_count << ',' << chunk_size << ',' << n << ','
                      << threads_per_block << ',' << warmup_iterations << ','
                      << benchmark_iterations << ',' << result.chunk_count << ','
                      << std::fixed << std::setprecision(6) << result.e2e_ms << ','
                      << std::setprecision(3) << speedup << ','
                      << (result.comparison.passed ? "PASS" : "FAIL") << ','
                      << std::scientific << result.comparison.max_abs_error << ','
                      << result.comparison.max_rel_error << ',';

            if (result.comparison.passed)
            {
                std::cout << "none";
            }
            else
            {
                std::cout << result.comparison.first_mismatch;
            }
            std::cout << '\n';

            all_passed = result.comparison.passed && all_passed;
        }
    }

    return all_passed ? 0 : 1;
}
