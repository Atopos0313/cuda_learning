#pragma once

#include "common/cuda_check.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

template <typename T> class DeviceBuffer
{
  public:
    // explicit prevents an integer from being converted into a buffer by
    // accident. count is an element count, not a byte count.
    explicit DeviceBuffer(std::size_t count) : size_(count)
    {
        if (count == 0)
        {
            return;
        }

        CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(std::exchange(other.ptr_, nullptr)), size_(std::exchange(other.size_, 0))
    {
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        if (this != &other)
        {
            release();
            ptr_ = std::exchange(other.ptr_, nullptr);
            size_ = std::exchange(other.size_, 0);
        }

        return *this;
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

    ~DeviceBuffer() noexcept
    {
        release();
    }

  private:
    void release() noexcept
    {
        if (ptr_ != nullptr)
        {
            // A destructor must not throw while another exception is already
            // unwinding. Normal CUDA operations are checked at their call site.
            (void)cudaFree(ptr_);
            ptr_ = nullptr;
            size_ = 0;
        }
    }

    T* ptr_ = nullptr;
    std::size_t size_ = 0;
};
