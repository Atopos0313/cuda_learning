#include "common/compare.h"
#include "common/init_data.h"
#include "vector_add.h"

#include <cstddef>
#include <iostream>
#include <vector>

namespace
{
bool run_test(int n, int threads_per_block)
{
    std::vector<float> h_a(static_cast<std::size_t>(n));
    std::vector<float> h_b(static_cast<std::size_t>(n));
    std::vector<float> h_c(static_cast<std::size_t>(n));
    std::vector<float> expected(static_cast<std::size_t>(n));

    init_random(h_a, 1);
    init_random(h_b, 2);

    vector_add(h_a.data(), h_b.data(), h_c.data(), n, threads_per_block);

    for (std::size_t index = 0; index < expected.size(); ++index)
    {
        expected[index] = h_a[index] + h_b[index];
    }

    return check_result(h_c, expected, 1.0e-6F);
}
} // namespace

int main()
{
    const std::vector<int> test_sizes = {0, 1, 1000, 100003};
    const std::vector<int> block_sizes = {64, 128, 256};

    bool all_passed = true;

    for (const int block_size : block_sizes)
    {
        for (const int n : test_sizes)
        {
            const bool passed = run_test(n, block_size);

            std::cout << "N=" << n << ", block=" << block_size
                      << ", correct=" << (passed ? "PASS" : "FAIL") << '\n';

            all_passed = passed && all_passed;
        }
    }

    return all_passed ? 0 : 1;
}
