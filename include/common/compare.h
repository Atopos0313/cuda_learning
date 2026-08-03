#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <vector>

struct CompareResult
{
    bool passed = true;
    // The largest size_t value is a sentinel meaning "no mismatch found".
    std::size_t first_mismatch = std::numeric_limits<std::size_t>::max();

    float max_abs_error = 0.0F;
    float max_rel_error = 0.0F;
};

inline CompareResult compare_results(const std::vector<float>& actual,
                                     const std::vector<float>& expected,
                                     float atol,
                                     float rtol)
{
    CompareResult result;

    if (actual.size() != expected.size())
    {
        result.passed = false;
        result.first_mismatch = 0;
        result.max_abs_error = std::numeric_limits<float>::infinity();
        result.max_rel_error = std::numeric_limits<float>::infinity();
        return result;
    }

    for (std::size_t index = 0; index < actual.size(); ++index)
    {
        if (!std::isfinite(actual[index]) || !std::isfinite(expected[index]))
        {
            if (result.passed)
            {
                result.first_mismatch = index;
            }

            result.passed = false;
            result.max_abs_error = std::numeric_limits<float>::infinity();
            result.max_rel_error = std::numeric_limits<float>::infinity();
            continue;
        }

        if (actual[index] == expected[index])
        {
            continue;
        }

        const float absolute_error = std::fabs(actual[index] - expected[index]);

        const float denominator = std::max(std::fabs(expected[index]), 1.0e-12F);

        const float relative_error = absolute_error / denominator;

        result.max_abs_error = std::max(result.max_abs_error, absolute_error);

        result.max_rel_error = std::max(result.max_rel_error, relative_error);

        const float tolerance = atol + rtol * std::fabs(expected[index]);

        if (absolute_error > tolerance)
        {
            if (result.passed)
            {
                result.first_mismatch = index;
            }

            result.passed = false;
        }
    }

    return result;
}

// Compatibility wrapper used by the early-day applications.
inline bool check_result(const std::vector<float>& actual,
                         const std::vector<float>& expected,
                         float eps)
{
    const CompareResult result = compare_results(actual, expected, eps, 0.0F);

    if (!result.passed)
    {
        std::cout << "fail at index " << result.first_mismatch
                  << ", max abs error = " << result.max_abs_error << '\n';
    }

    return result.passed;
}
