#include "common/cuda_check.h"
#include "common/compare.h"
#include "common/device_buffer.h"
#include "common/init_data.h"

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <algorithm>

__global__ void simpleKernel(
    const float* d_input,
    float* d_output,
    int n
){
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < n){
        d_output[index] = d_input[index] * 2.0f + 1.0f;
    }
}

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

    float* h_input_pinned = nullptr;
    float* h_output_pinned = nullptr;

    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_input_pinned), bytes));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_output_pinned), bytes));

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
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(
        // CUDA，请把创建好的 Stream 句柄写到这里。
        &stream,
        // 表示这条 Stream 不会与默认 Stream 建立隐式同步关系
        cudaStreamNonBlocking
    ));

    cudaEvent_t completion_event = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(
        &completion_event,
        // 此事件不用于计时
        cudaEventDisableTiming
    ));

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
        stream
    ));

    CUDA_CHECK(cudaMemcpyAsync(
        d_input.data(),
        h_input_pinned,
        bytes,
        cudaMemcpyHostToDevice,
        stream
    ));
    simpleKernel<<<
        grid,
        threads_per_block,
        0,
        stream
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
        stream
    ));

    CUDA_CHECK(cudaEventRecord(
        completion_event,
        stream
    ));

    CUDA_CHECK(cudaEventSynchronize(
        completion_event
    ));

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

    CUDA_CHECK(cudaEventDestroy(completion_event));
    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFreeHost(h_input_pinned));
    CUDA_CHECK(cudaFreeHost(h_output_pinned));

    return (
        compare_sim.passed &&
        pinned_comparison.passed &&
        async_comparison.passed
    ) ? 0 : 1;
}
