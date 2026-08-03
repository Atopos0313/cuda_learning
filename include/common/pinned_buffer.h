#pragma once

#include "common/cuda_check.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

template <typename T> class PinnedBuffer
{
  public:
    explicit PinnedBuffer(std::size_t count) : size_(count)
    {
        if (size_ != 0)
        {
            CUDA_CHECK(cudaMallocHost(&ptr_,
                                      size_ * sizeof(T)));
        }
    }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    PinnedBuffer(PinnedBuffer&& other) noexcept
        : ptr_(std::exchange(other.ptr_, nullptr)),
          size_(std::exchange(other.size_, 0))
    {
    }

    PinnedBuffer& operator=(PinnedBuffer&&) = delete;

    ~PinnedBuffer() noexcept
    {
        if (ptr_ != nullptr)
        {
            // (void)是为了忽略返回值，因为析构函数最好别抛异常
            (void)cudaFreeHost(ptr_);
        }
    }

    T* data() noexcept
    {
        return ptr_;
    }

    const T* data() const noexcept
    {
        return ptr_;
    }

    std::size_t size() const noexcept
    {
        return size_;
    }

  private:
    T* ptr_ = nullptr;
    std::size_t size_ = 0;
};
