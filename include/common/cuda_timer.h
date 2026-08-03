#pragma once

#include "common/cuda_check.h"

#include <cuda_runtime.h>

class CudaEventTimer
{
  public:
    CudaEventTimer()
    {
        CUDA_CHECK(cudaEventCreate(&start_));
        try
        {
            CUDA_CHECK(cudaEventCreate(&stop_));
        }
        catch (...)
        {
            (void)cudaEventDestroy(start_);
            start_ = nullptr;
            throw;
        }
    }

    CudaEventTimer(const CudaEventTimer&) = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;

    void start(cudaStream_t stream = nullptr)
    {
        CUDA_CHECK(cudaEventRecord(start_, stream));
    }

    void stop(cudaStream_t stream = nullptr)
    {
        CUDA_CHECK(cudaEventRecord(stop_, stream));
    }

    float elapsed_ms()
    {
        CUDA_CHECK(cudaEventSynchronize(stop_));

        float milliseconds = 0.0F;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_, stop_));
        return milliseconds;
    }

    ~CudaEventTimer() noexcept
    {
        if (start_ != nullptr)
        {
            (void)cudaEventDestroy(start_);
        }
        if (stop_ != nullptr)
        {
            (void)cudaEventDestroy(stop_);
        }
    }

  private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};
