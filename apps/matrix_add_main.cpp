#include <cstddef>
#include <iostream>
#include <vector>

#include "common/compare.h"
#include "common/init_data.h"
#include "matrix_add.h"

bool run_matrix_test(int height, int width)
{
    const std::size_t element_count =
        static_cast<std::size_t>(height) *
        static_cast<std::size_t>(width);

    std::vector<float> h_A(element_count);
    std::vector<float> h_B(element_count);
    std::vector<float> h_C(element_count);
    std::vector<float> cpu_C(element_count);

    init_random(h_A, 1);
    init_random(h_B, 2);

    matrix_add(
        h_A.data(),
        h_B.data(),
        h_C.data(),
        height,
        width
    );

    for (int i = 0; i < element_count; ++i) {
        cpu_C[i] = h_A[i] + h_B[i];
    }

    return check_result(h_C, cpu_C, 1e-6f);
}

struct MatrixShape
{
    int height;
    int width;
};

int main()
{
    const std::vector<MatrixShape> test_shapes = {
        {0, 5},
        {1, 1},
        {3, 5},
        {17, 19},
        {100, 1000}
    };

    bool all_pass = true;

    // MatrixShape   元素的数据类型
    // shape         当前取出来的元素
    // test_shapes   被遍历的容器
    // const         不允许修改当前的 shape

    for (const MatrixShape shape : test_shapes) {
        std::cout
            << "testing height = " << shape.height
            << ", width = " << shape.width
            << std::endl;

        const bool ok =
            run_matrix_test(shape.height, shape.width);

        std::cout
            << (ok ? "PASS" : "FAIL")
            << std::endl;

        if (!ok) {
            all_pass = false;
        }
    }

    return all_pass ? 0 : 1;
}
