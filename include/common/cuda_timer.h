#pragma once

#include "common/cuda_check.h"
#include <cuda_runtime.h>

class CudaEventTimer
{
public:
    CudaEventTimer(){
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    CudaEventTimer(const CudaEventTimer&) = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;
    void start(cudaStream_t stream = nullptr){
        CUDA_CHECK(cudaEventRecord(start_, stream));
    }
    void stop(cudaStream_t stream = nullptr){
        CUDA_CHECK(cudaEventRecord(stop_, stream));
    }
    float elapsed_ms(){
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(
            &milliseconds,
            start_,
            stop_
        ));
        return milliseconds;
    }

    ~CudaEventTimer() noexcept {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};
