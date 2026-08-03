#include "common/compare.h"
#include "common/init_data.h"
#include "matrix_add.h"

#include <cstddef>
#include <iostream>
#include <vector>

namespace
{
struct MatrixShape
{
    int height;
    int width;
};

bool run_matrix_test(int height, int width)
{
    const std::size_t element_count =
        static_cast<std::size_t>(height) * static_cast<std::size_t>(width);

    std::vector<float> h_a(element_count);
    std::vector<float> h_b(element_count);
    std::vector<float> h_c(element_count);
    std::vector<float> expected(element_count);

    init_random(h_a, 1);
    init_random(h_b, 2);

    matrix_add(h_a.data(), h_b.data(), h_c.data(), height, width);

    for (std::size_t index = 0; index < element_count; ++index)
    {
        expected[index] = h_a[index] + h_b[index];
    }

    return check_result(h_c, expected, 1.0e-6F);
}
} // namespace

int main()
{
    const std::vector<MatrixShape> test_shapes = {
        {0, 5}, {1, 1}, {3, 5}, {17, 19}, {100, 1000}};

    bool all_passed = true;

    for (const MatrixShape shape : test_shapes)
    {
        const bool passed = run_matrix_test(shape.height, shape.width);

        std::cout << "height=" << shape.height << ", width=" << shape.width
                  << ", correct=" << (passed ? "PASS" : "FAIL") << '\n';

        all_passed = passed && all_passed;
    }

    return all_passed ? 0 : 1;
}
