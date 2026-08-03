#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

template <typename T> T benchmark_median(std::vector<T> values)
{
    if (values.empty())
    {
        throw std::invalid_argument("benchmark_median requires at least one sample");
    }

    std::sort(values.begin(), values.end());

    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 == 0)
    {
        return (values[middle - 1] + values[middle]) / static_cast<T>(2);
    }

    return values[middle];
}
