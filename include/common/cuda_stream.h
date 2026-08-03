#pragma once

#include "common/cuda_check.h"

#include <cuda_runtime.h>

#include <utility>

class CudaStream
{
  public:
    explicit CudaStream(unsigned int flags = cudaStreamNonBlocking)
    {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, flags));
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    CudaStream(CudaStream&& other) noexcept
        : stream_(std::exchange(other.stream_, nullptr))
    {
    }

    CudaStream& operator=(CudaStream&&) = delete;

    ~CudaStream() noexcept
    {
        if (stream_ != nullptr)
        {
            (void)cudaStreamDestroy(stream_);
        }
    }
    // 相当于拿到自己的句柄 ，因为cudaStream_t stream_ 是私有变量，外部无法访问
    cudaStream_t get() const noexcept
    {
        return stream_;
    }

    void synchronize() const
    {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

  private:
    cudaStream_t stream_ = nullptr;
};
