#pragma once

#include "common/cuda_check.h"

#include <cuda_runtime.h>

#include <utility>

class CudaCompletionEvent
{
  public:
    CudaCompletionEvent()
    {
        CUDA_CHECK(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
    }

    CudaCompletionEvent(const CudaCompletionEvent&) = delete;
    CudaCompletionEvent& operator=(const CudaCompletionEvent&) = delete;

    CudaCompletionEvent(CudaCompletionEvent&& other) noexcept
        : event_(std::exchange(other.event_, nullptr))
    {
    }

    CudaCompletionEvent& operator=(CudaCompletionEvent&&) = delete;

    ~CudaCompletionEvent() noexcept
    {
        if (event_ != nullptr)
        {
            (void)cudaEventDestroy(event_);
        }
    }

    void record(cudaStream_t stream) const
    {
        CUDA_CHECK(cudaEventRecord(event_, stream));
    }

    void synchronize() const
    {
        CUDA_CHECK(cudaEventSynchronize(event_));
    }

    cudaEvent_t get() const noexcept
    {
        return event_;
    }

  private:
    cudaEvent_t event_ = nullptr;
};
