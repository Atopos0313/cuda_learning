#pragma once

#include <stdexcept>

// Returns ceil(value / divisor) for a non-negative value without using
// floating-point arithmetic. Empty workloads produce zero blocks.
inline int ceil_div(int value, int divisor)
{
    if (divisor <= 0)
    {
        throw std::invalid_argument("divisor must be positive");
    }

    if (value <= 0)
    {
        return 0;
    }

    return 1 + (value - 1) / divisor;
}

inline void validate_threads_per_block(int threads_per_block)
{
    constexpr int kMaximumThreadsPerBlock = 1024;

    if (threads_per_block <= 0 || threads_per_block > kMaximumThreadsPerBlock)
    {
        throw std::invalid_argument("threads_per_block must be in [1, 1024]");
    }
}
