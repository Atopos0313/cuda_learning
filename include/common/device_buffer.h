#pragma once

#include "common/cuda_check.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

template <typename T>
class DeviceBuffer
{
public:
// explicit 禁止编译器把整数自动转换成 DeviceBuffer,: size_(count)相当于构造的时候直接把size_初始化为count
    explicit DeviceBuffer(std::size_t count)
        : size_(count)
    {
        if (count == 0) {
            return;
        }

        CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(std::exchange(other.ptr_, nullptr)),
          size_(std::exchange(other.size_, 0))
    {
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        if (this != &other) {
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
        if (ptr_ != nullptr) {
            // Destructors cannot report failures by throwing.
            (void)cudaFree(ptr_);
            ptr_ = nullptr;
            size_ = 0;
        }
    }

    T* ptr_ = nullptr;
    std::size_t size_ = 0;
};
