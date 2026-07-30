#include "common/compare.h"
#include "common/device_buffer.h"

#include <limits>
#include <type_traits>
#include <vector>

static_assert(
    !std::is_copy_constructible_v<DeviceBuffer<float>>,
    "DeviceBuffer must not be copy constructible"
);

static_assert(
    !std::is_copy_assignable_v<DeviceBuffer<float>>,
    "DeviceBuffer must not be copy assignable"
);

static_assert(
    std::is_nothrow_move_constructible_v<DeviceBuffer<float>>,
    "DeviceBuffer must support noexcept move construction"
);

static_assert(
    std::is_nothrow_move_assignable_v<DeviceBuffer<float>>,
    "DeviceBuffer must support noexcept move assignment"
);

int main()
{
    const std::vector<float> expected = {1.0F};
    const std::vector<float> actual = {
        std::numeric_limits<float>::quiet_NaN()
    };

    const CompareResult result = compare_results(
        actual,
        expected,
        1.0e-6F,
        1.0e-5F
    );

    return result.passed ? 1 : 0;
}
