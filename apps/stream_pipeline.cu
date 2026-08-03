#include "common/cuda_check.h"
#include "common/compare.h"
#include "common/cuda_event.h"
#include "common/cuda_stream.h"
#include "common/device_buffer.h"
#include "common/init_data.h"
#include "common/pinned_buffer.h"
#include "stream_pipeline.h"

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <array>

int main(){
    int N = 1000003;
    size_t bytes = N * sizeof(float);
    std::vector<float> h_input(N);
    std::vector<float> h_output(N);
    std::vector<float> reference(N);

    init_random(h_input, 20260801);

    for (int i = 0; i < N; i++){
        reference[i] = h_input[i] * 2.0f + 1.0f;
    }

    PinnedBuffer<float> pinned_input(static_cast<std::size_t>(N));
    PinnedBuffer<float> pinned_output(static_cast<std::size_t>(N));
    // 指针本身不能改变，但它指向的数据可以改变
    float* const h_input_pinned = pinned_input.data();
    float* const h_output_pinned = pinned_output.data();

    std::copy(h_input.begin(), h_input.end(), h_input_pinned);

    DeviceBuffer<float> d_input(N);
    DeviceBuffer<float> d_output(N);

    CUDA_CHECK(cudaMemcpy(d_input.data(), h_input.data(), bytes, cudaMemcpyHostToDevice));

    int threads_per_block = 256;
    int grid = (N + threads_per_block - 1) / threads_per_block;

    simpleKernel<<<grid, threads_per_block>>>(d_input.data(), d_output.data(), N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output.data(), bytes, cudaMemcpyDeviceToHost));

    const CompareResult compare_sim = compare_results(h_output, reference, 1e-6f, 1e-5f);

    std::cout<<
    "mode=serial_pageable" <<
    ", size=" << N <<
    ", correct=" << (compare_sim.passed ? "PASS" : "FAIL") <<
    ", max_abs_error=" << compare_sim.max_abs_error <<
    ", max_rel_error=" << compare_sim.max_rel_error <<
    std::endl;

    // 以下代码为使用页锁定内存
    CUDA_CHECK(cudaMemcpy(
        d_input.data(),
        h_input_pinned,
        bytes,
        cudaMemcpyHostToDevice
    ));

    simpleKernel<<<grid, threads_per_block>>>(
        d_input.data(),
        d_output.data(),
        N
    );

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(
        h_output_pinned,
        d_output.data(),
        bytes,
        cudaMemcpyDeviceToHost
    ));
    // 把数组h_output_pinned的首地址和末尾地址传过去
    /*
    std::vector<float> pinned_result(N);

    for (int i = 0; i < N; ++i) {
        pinned_result[i] = h_output_pinned[i];
    }
    h_output_pinned是 float*类型指针，指向数组的首地址
    因为类型是 float*，编译器会自动按 sizeof(float) 移动
    */
    const std::vector<float> pinned_result(
        h_output_pinned,
        h_output_pinned + N
    );

    const CompareResult pinned_comparison = compare_results(
        pinned_result,
        reference,
        1.0e-6F,
        1.0e-5F
    );

    std::cout
        << "mode=serial_pinned"
        << ", size=" << N
        << ", correct="
        << (pinned_comparison.passed ? "PASS" : "FAIL")
        << ", max_abs_error=" << pinned_comparison.max_abs_error
        << ", max_rel_error=" << pinned_comparison.max_rel_error
        << '\n';

    // 同一个 Stream 永远严格按顺序执行
    //以下代码为单stream异步链
    CudaStream stream;
    CudaCompletionEvent completion_event;

    // 清除旧结果
    std::fill(
        h_output_pinned,
        h_output_pinned + N,
        -1.0f
    );
    // 与CPU中memset类似，将某一块内存的值填成0
    // cudaMemsetAsync 和 cudaMemcpyAsync 允许 CPU 在 GPU 工作完成前返回。
    // cudaMemsetAsync的时候，CPU只是告诉了GPU执行，但是具体什么时候执行取决于stream
    // 如果使用cudaMemset，并没有stream参数，它默认在 Default Stream中执行
    CUDA_CHECK(cudaMemsetAsync(
        d_output.data(),
        0,
        bytes,
        // 把这个 memset 放进哪条任务队列？
        stream.get()
    ));

    CUDA_CHECK(cudaMemcpyAsync(
        d_input.data(),
        h_input_pinned,
        bytes,
        cudaMemcpyHostToDevice,
        stream.get()
    ));
    simpleKernel<<<
        grid,
        threads_per_block,
        0,
        stream.get()
    >>>(
        d_input.data(),
        d_output.data(),
        N
    );

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(
        h_output_pinned,
        d_output.data(),
        bytes,
        cudaMemcpyDeviceToHost,
        stream.get()
    ));

    completion_event.record(stream.get());
    completion_event.synchronize();

    const std::vector<float> async_result(
        h_output_pinned,
        h_output_pinned + N
    );

    const CompareResult async_comparison = compare_results(
        async_result,
        reference,
        1.0e-6F,
        1.0e-5F
    );

    std::cout
        << "mode=async_pinned"
        << ", streams=1"
        << ", size=" << N
        << ", correct="
        << (async_comparison.passed ? "PASS" : "FAIL")
        << ", max_abs_error=" << async_comparison.max_abs_error
        << ", max_rel_error=" << async_comparison.max_rel_error
        << '\n';

    // 2-stream
    const int stream_count = 2;
    const std::size_t chunk_size = 262'144;
    // 创建两个流
    std::array<CudaStream, stream_count> streams;

    std::array<DeviceBuffer<float>, stream_count> chunk_inputs{
        DeviceBuffer<float>(chunk_size),
        DeviceBuffer<float>(chunk_size)
    };
    std::array<DeviceBuffer<float>, stream_count> chunk_outputs{
        DeviceBuffer<float>(chunk_size),
        DeviceBuffer<float>(chunk_size)
    };

    const std::size_t total_count = static_cast<std::size_t>(N);
    const std::size_t chunk_count = (total_count + chunk_size - 1) / chunk_size;

    // 防止读取上一次单 Stream 实验留下的正确结果
    std::fill(
        h_output_pinned,
        h_output_pinned + N,
        -1.0f
    );

    for (std::size_t chunk_id = 0; chunk_id < chunk_count; chunk_id++){
        // 偏移量，用于计算元素的起始位置
        const std::size_t offset = chunk_id * chunk_size;
        // 最后一个chunk中的剩余元素
        const std::size_t remaining = total_count - offset;
        // 计算完与之前的相比较
        const std::size_t current_count = std::min(chunk_size, remaining);
        // 把每个chunk放在哪个stream中
        const std::size_t stream_id = chunk_id % streams.size();
        // 每一个chunk分配多少内存
        const std::size_t current_bytes = current_count * sizeof(float);
        // 每一个chunk分配多少block
        const int chunk_grid = static_cast<int>(
            (current_count + threads_per_block - 1) /
            threads_per_block
        );

        CUDA_CHECK(cudaMemcpyAsync(
            // 每个chunk的下标都是从0开始的
            chunk_inputs[stream_id].data(),
            // h_input_pinned + offset是用于计算对应位置，确定把原数组的哪些元素搬入对应的chunk中
            h_input_pinned + offset,
            current_bytes,
            cudaMemcpyHostToDevice,
            streams[stream_id].get()
        ));

        const int kernel_count = static_cast<int>(current_count);

        simpleKernel<<<chunk_grid, threads_per_block, 0, streams[stream_id].get()>>>(chunk_inputs[stream_id].data(), chunk_outputs[stream_id].data(), kernel_count);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpyAsync(
            h_output_pinned + offset,
            chunk_outputs[stream_id].data(),
            current_bytes,
            cudaMemcpyDeviceToHost,
            streams[stream_id].get()
        ));

        std::cout
        << "chunk_id=" << chunk_id
        << ", offset=" << offset
        << ", current_count=" << current_count
        << ", stream_id=" << stream_id
        << ", current_bytes= " << current_bytes
        << ", chunk_grid= " << chunk_grid
        << '\n';
    }

    // stream的同步
    for (const CudaStream& chunk_stream : streams)
    {
        chunk_stream.synchronize();
    }

    const std::vector<float> compare2stream_result(
        h_output_pinned,
        h_output_pinned + N
    );

    const CompareResult compare2stream_comparison =
        compare_results(
            compare2stream_result,
            reference,
            1.0e-6F,
            1.0e-5F
        );

    std::cout
        << "mode=2stream_copy"
        << ", streams=2"
        << ", size=" << N
        << ", correct="
        << (compare2stream_comparison.passed ? "PASS" : "FAIL")
        << ", max_abs_error="
        << compare2stream_comparison.max_abs_error
        << ", max_rel_error="
        << compare2stream_comparison.max_rel_error
        << '\n';

    return (
        compare_sim.passed &&
        pinned_comparison.passed &&
        async_comparison.passed &&
        compare2stream_comparison.passed
    ) ? 0 : 1;
}
