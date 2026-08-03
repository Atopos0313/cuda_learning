#include "common/compare.h"
#include "common/device_buffer.h"
#include "common/launch_config.h"

#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

static_assert(!std::is_copy_constructible_v<DeviceBuffer<float>>,
              "DeviceBuffer must not be copy constructible");
static_assert(!std::is_copy_assignable_v<DeviceBuffer<float>>,
              "DeviceBuffer must not be copy assignable");
static_assert(std::is_nothrow_move_constructible_v<DeviceBuffer<float>>,
              "DeviceBuffer must support noexcept move construction");
static_assert(std::is_nothrow_move_assignable_v<DeviceBuffer<float>>,
              "DeviceBuffer must support noexcept move assignment");

int main()
{
    bool all_passed = true;

    const CompareResult exact =
        compare_results({1.0F, 2.0F}, {1.0F, 2.0F}, 1.0e-6F, 1.0e-5F);
    all_passed = exact.passed && all_passed;

    const CompareResult nan_result = compare_results(
        {std::numeric_limits<float>::quiet_NaN()}, {1.0F}, 1.0e-6F, 1.0e-5F);
    all_passed = !nan_result.passed && all_passed;

    const CompareResult infinity_result =
        compare_results({std::numeric_limits<float>::infinity()},
                        {std::numeric_limits<float>::infinity()},
                        1.0e-6F,
                        1.0e-5F);
    all_passed = !infinity_result.passed && all_passed;

    const CompareResult size_result =
        compare_results({1.0F}, {1.0F, 2.0F}, 1.0e-6F, 1.0e-5F);
    all_passed = !size_result.passed && all_passed;

    all_passed = ceil_div(0, 256) == 0 && all_passed;
    all_passed = ceil_div(1, 256) == 1 && all_passed;
    all_passed = ceil_div(1000, 256) == 4 && all_passed;

    bool invalid_divisor_reported = false;
    try
    {
        (void)ceil_div(1, 0);
    }
    catch (const std::invalid_argument&)
    {
        invalid_divisor_reported = true;
    }

    all_passed = invalid_divisor_reported && all_passed;
    return all_passed ? 0 : 1;
}
